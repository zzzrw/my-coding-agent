from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from coding_agent.runtime.models import Message
from coding_agent.tools.models import ToolResult

from .models import SessionHeader, SessionMessage, SessionRecord, SessionSummary


class SessionStore:
    def __init__(
        self,
        path: Path,
        header: SessionHeader,
        records: list[SessionRecord],
        notice: str | None = None,
    ) -> None:
        self._path = path
        self._header = header
        self._records = records
        self._notice = notice
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        session_id: str | None = None,
        workspace: str,
        model: str,
        context_window: int,
        title: str = "New session",
    ) -> SessionStore:
        root.mkdir(parents=True, exist_ok=True)
        sid = session_id or uuid.uuid4().hex
        now = datetime.now(UTC)
        header = SessionHeader(
            session_id=sid,
            workspace=workspace,
            model=model,
            title=title,
            created_at=now,
            updated_at=now,
            context_window=context_window,
        )
        path = root / f"{sid}.jsonl"
        if path.exists():
            raise FileExistsError(f"session already exists: {sid}")
        with path.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(header.model_dump(mode="json"), ensure_ascii=True) + "\n"
            )
            stream.flush()
            import os

            os.fsync(stream.fileno())
        return cls(path, header, [])

    @classmethod
    def open(cls, root: Path, session_id: str) -> SessionStore:
        if Path(session_id).name != session_id or session_id in {"", ".", ".."}:
            raise ValueError("invalid session id")
        path = root / f"{session_id}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("empty session file")
        try:
            header = SessionHeader.model_validate(json.loads(lines[0]))
        except Exception as exc:
            raise ValueError("invalid session header") from exc
        records: list[SessionRecord] = []
        notice = None
        for index, line in enumerate(lines[1:], 1):
            try:
                records.append(SessionRecord.model_validate(json.loads(line)))
            except Exception as exc:
                if index == len(lines) - 1:
                    notice = f"corrupt final session record ignored: {exc}"
                    break
                raise ValueError(f"invalid session record at line {index + 1}") from exc
        for expected, record in enumerate(records):
            parent = records[expected - 1].id if expected else None
            if record.seq != expected or record.parent_id != parent:
                raise ValueError("invalid session record sequence or parent chain")
        return cls(path, header, records, notice)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session_id(self) -> str:
        return self._header.session_id

    @property
    def header(self) -> SessionHeader:
        if not self._records:
            return self._header
        latest = max(record.timestamp for record in self._records)
        if latest <= self._header.updated_at:
            return self._header
        return self._header.model_copy(update={"updated_at": latest})

    @property
    def load_notice(self) -> str | None:
        return self._notice

    def append(self, record: SessionRecord) -> None:
        with self._lock:
            if self._notice is not None:
                lines = [self._header.model_dump(mode="json")] + [
                    item.model_dump(mode="json") for item in self._records
                ]
                self._path.write_text(
                    "\n".join(json.dumps(item, ensure_ascii=True) for item in lines)
                    + "\n",
                    encoding="utf-8",
                )
                self._notice = None
            expected = len(self._records)
            if record.seq != expected:
                raise ValueError(f"record seq must be {expected}")
            parent = self._records[-1].id if self._records else None
            if record.parent_id != parent:
                raise ValueError("record parent_id does not match active leaf")
            self._records.append(record)
            self._header = self._header.model_copy(
                update={"updated_at": record.timestamp}
            )
            import os

            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(record.model_dump(mode="json"), ensure_ascii=True) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())

    def append_new(
        self,
        record_type: str,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
        turn_id: str | None = None,
    ) -> SessionRecord:
        record = SessionRecord(
            id=uuid.uuid4().hex,
            seq=len(self._records),
            timestamp=datetime.now(UTC),
            type=record_type,
            payload=payload,
            parent_id=self._records[-1].id if self._records else None,
            run_id=run_id,
            turn_id=turn_id,
        )
        self.append(record)
        return record

    def records(self) -> list[SessionRecord]:
        return list(self._records)

    def has_interrupted_turn(self) -> bool:
        open_turns: set[str] = set()
        for record in self._records:
            tid = record.turn_id or record.payload.get("turn_id")
            if record.type == "turn_start" and tid:
                open_turns.add(tid)
            elif record.type == "turn_end" and tid:
                open_turns.discard(tid)
        return bool(open_turns)

    def project_messages(
        self, *, include_open_turn: bool = False
    ) -> list[SessionMessage]:
        open_turns: set[str] = set()
        for record in self._records:
            tid = record.turn_id or record.payload.get("turn_id")
            if record.type == "turn_start" and tid:
                open_turns.add(tid)
            elif record.type == "turn_end" and tid:
                open_turns.discard(tid)
        projected: list[SessionMessage] = []
        pending_calls: dict[str, tuple[int, str | None]] = {}
        completed_calls: set[str] = set()
        active_assistant: tuple[int, str | None] | None = None
        for record in self._records:
            tid = record.turn_id or record.payload.get("turn_id")
            if not include_open_turn and tid in open_turns:
                continue
            if (
                record.type == "assistant_message"
                and record.payload.get("complete") is True
            ):
                msg = Message.model_validate(record.payload["message"])
                idx = len(projected)
                projected.append(
                    SessionMessage(
                        record_id=record.id,
                        turn_id=record.turn_id,
                        seq=record.seq,
                        message=msg,
                    )
                )
                active_assistant = (idx, tid)
                for call in msg.tool_calls:
                    pending_calls[call.id] = (idx, tid)
            elif record.type == "user_message":
                msg = Message.model_validate(record.payload["message"])
                projected.append(
                    SessionMessage(
                        record_id=record.id,
                        turn_id=record.turn_id,
                        seq=record.seq,
                        message=msg,
                    )
                )
                active_assistant = None
            elif record.type == "tool_result":
                result = ToolResult.model_validate(record.payload["result"])
                if (
                    result.tool_call_id not in pending_calls
                    or result.tool_call_id in completed_calls
                ):
                    raise ValueError(
                        f"tool_result does not match preceding tool call: {result.tool_call_id}"
                    )
                assistant_idx, assistant_turn = pending_calls[result.tool_call_id]
                if assistant_turn != tid or active_assistant != (
                    assistant_idx,
                    assistant_turn,
                ):
                    raise ValueError(
                        f"tool_result is not in assistant turn: {result.tool_call_id}"
                    )
                completed_calls.add(result.tool_call_id)
                projected.append(
                    SessionMessage(
                        record_id=record.id,
                        turn_id=record.turn_id,
                        seq=record.seq,
                        message=Message(
                            role="tool",
                            content=result.content,
                            tool_call_id=result.tool_call_id,
                            name=result.tool_name,
                        ),
                    )
                )
            elif record.type not in {"tool_call", "turn_start", "turn_end"}:
                active_assistant = None
        dangling = {
            call_id for call_id in pending_calls if call_id not in completed_calls
        }
        if dangling:
            projected = [
                item
                for item in projected
                if not (
                    item.message.role == "assistant"
                    and any(call.id in dangling for call in item.message.tool_calls)
                )
            ]
        return projected

    @classmethod
    def list_sessions(cls, root: Path) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for path in root.glob("*.jsonl"):
            try:
                store = cls.open(root, path.stem)
            except ValueError:
                continue
            title = store.header.title
            if title == "New session":
                for record in store.records():
                    if record.type == "user_message":
                        try:
                            title = (
                                Message.model_validate(
                                    record.payload["message"]
                                ).content
                                or title
                            )
                        except (KeyError, ValidationError):
                            title = store.header.title
                        break
            status = "interrupted" if store.has_interrupted_turn() else "idle"
            summaries.append(
                SessionSummary(
                    id=store.session_id,
                    workspace=store.header.workspace,
                    created_at=store.header.created_at,
                    updated_at=store.header.updated_at,
                    title=title[:80],
                    last_status=status,
                )
            )
        return sorted(summaries, key=lambda s: s.updated_at, reverse=True)
