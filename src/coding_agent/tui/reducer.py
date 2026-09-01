from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import TypeAdapter

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.session.models import ApprovalRequest, SessionMessage
from coding_agent.tui.state import TranscriptItem, TuiState


def reduce(state: TuiState, event: RuntimeEvent) -> TuiState:
    """Apply one runtime event without mutating the input snapshot."""
    if _is_stale_run_event(state, event):
        return state
    payload = event.payload
    updates: dict[str, Any] = {}
    transcript = [row.model_copy(deep=True) for row in state.transcript]
    pending_approval = (
        state.pending_approval.model_copy(deep=True)
        if state.pending_approval is not None
        else None
    )

    if event.type == "run_started":
        updates.update(
            status="running",
            active_run_id=event.run_id,
            active_turn_id=event.turn_id,
            session_id=_optional_str(payload.get("session_id"), state.session_id),
            model=_optional_str(payload.get("model"), state.model),
            policy=_policy(payload.get("policy"), state.policy),
            pending_approval=None,
            active_tool_call_id=None,
        )

    elif event.type == "run_finished":
        if _event_matches_active_run(state, event):
            outcome = payload.get("outcome", {})
            reason = outcome if isinstance(outcome, str) else None
            if isinstance(outcome, dict):
                reason = outcome.get("reason")
            status = _terminal_status(reason)
            updates.update(
                status=status,
                active_run_id=None,
                active_turn_id=None,
                active_tool_call_id=None,
                pending_approval=None,
            )

    elif event.type == "run_error":
        if _event_matches_active_run(state, event):
            updates.update(
                status="error",
                active_run_id=None,
                active_turn_id=None,
                active_tool_call_id=None,
                pending_approval=None,
            )
            _append_system(
                transcript, payload.get("message", "runtime error"), level="error"
            )

    elif event.type == "user_message":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id:
            _append_or_update(
                transcript,
                TranscriptItem(
                    kind="user",
                    item_id=message_id,
                    text=_text(payload.get("text")),
                ),
            )

    elif event.type == "assistant_started":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id:
            _append_or_update(
                transcript,
                TranscriptItem(kind="assistant", item_id=message_id),
            )

    elif event.type == "assistant_delta":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id:
            index = _find_assistant(transcript, message_id)
            if index is None:
                transcript.append(
                    TranscriptItem(
                        kind="assistant",
                        item_id=message_id,
                        text=_text(payload.get("text")),
                    )
                )
            else:
                row = transcript[index]
                transcript[index] = row.model_copy(
                    update={"text": row.text + _text(payload.get("text"))}
                )

    elif event.type == "assistant_finished":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id is not None and _find_assistant(transcript, message_id) is None:
            transcript.append(TranscriptItem(kind="assistant", item_id=message_id))

    elif event.type == "tool_started":
        call_id = _non_empty_str(payload.get("tool_call_id"))
        if call_id:
            tool_name = _optional_str(payload.get("tool_name"), None)
            row = TranscriptItem(
                kind="tool",
                item_id=call_id,
                tool_call_id=call_id,
                tool_name=tool_name,
                text=_arguments_text(payload.get("arguments")),
                tool_status="running",
                command=_command_text(payload.get("arguments"), tool_name),
            )
            _append_or_update(transcript, row, tool_call_id=call_id)
            updates.update(active_tool_call_id=call_id, status="running")

    elif event.type == "tool_finished":
        call_id = _non_empty_str(payload.get("tool_call_id"))
        if call_id:
            ok = payload.get("ok") is True
            error = _text(payload.get("error"))
            text = _text(payload.get("content"))
            status = _tool_status(payload.get("status"), ok, error, text)
            metadata = _metadata(payload.get("metadata"))
            index = _find_tool(transcript, call_id)
            replacement = TranscriptItem(
                kind="tool",
                item_id=call_id,
                tool_call_id=call_id,
                tool_name=_optional_str(payload.get("tool_name"), None),
                text=text or error,
                tool_status=status,
                elapsed_seconds=_float_or_none(metadata.get("elapsed_seconds")),
                truncated=_bool_or_none(metadata.get("truncated")),
                exit_code=_int_or_none(metadata.get("exit_code")),
            )
            if index is None:
                transcript.append(replacement)
            else:
                old = transcript[index]
                transcript[index] = old.model_copy(
                    update={
                        "tool_name": replacement.tool_name or old.tool_name,
                        "text": replacement.text,
                        "tool_status": status,
                        "command": replacement.command or old.command,
                        "elapsed_seconds": replacement.elapsed_seconds,
                        "truncated": replacement.truncated,
                        "exit_code": replacement.exit_code,
                    }
                )
            if (
                state.active_tool_call_id == call_id
                or updates.get("active_tool_call_id") == call_id
            ):
                updates["active_tool_call_id"] = None

    elif event.type == "approval_requested":
        if _event_matches_active_run(state, event):
            request = _approval(payload.get("request"))
            if request is not None:
                updates.update(pending_approval=request, status="waiting_approval")

    elif event.type == "approval_resolved":
        if _event_matches_active_run(state, event):
            request_id = _non_empty_str(payload.get("request_id"))
            if (
                state.pending_approval is not None
                and request_id == state.pending_approval.request_id
            ):
                updates["pending_approval"] = None
                if payload.get("status") == "cancelled":
                    updates["status"] = "aborted"
                elif payload.get("status") in {"approved", "denied"}:
                    updates["status"] = "running"

    elif event.type == "status_changed":
        status = payload.get("status")
        if status in {"idle", "running", "waiting_approval", "error", "aborted"}:
            updates["status"] = status

    elif event.type == "context_updated":
        updates.update(
            context_used=_int(payload.get("used_tokens"), state.context_used),
            context_window=_int(payload.get("context_window"), state.context_window),
            context_estimated=bool(payload.get("estimated", state.context_estimated)),
        )

    elif event.type == "session_loaded":
        updates.update(
            session_id=_optional_str(payload.get("session_id"), state.session_id),
            workspace=_optional_str(payload.get("workspace"), state.workspace),
            model=_optional_str(payload.get("model"), state.model),
            context_used=0,
            context_window=_int(payload.get("context_window"), 0),
            context_estimated=False,
            policy="default",
            status="idle",
            active_run_id=None,
            active_turn_id=None,
            active_tool_call_id=None,
            pending_approval=None,
            transcript=_projected_transcript(payload.get("history"))
            + [row for row in state.transcript if row.kind == "local_command"],
        )

    elif event.type == "policy_changed":
        updates["policy"] = _policy(payload.get("policy"), state.policy)

    elif event.type == "notice":
        command = _non_empty_str(payload.get("command"))
        if command:
            transcript.append(
                TranscriptItem(
                    kind="local_command",
                    item_id=_next_suffixed_id(
                        transcript, "command-", kind="local_command"
                    ),
                    text=command,
                )
            )
        message = payload.get("message")
        if message is not None:
            level = payload.get("level", "notice")
            _append_system(
                transcript,
                message,
                level=level if level in {"notice", "error"} else "notice",
            )

    if event.type != "session_loaded":
        updates["transcript"] = transcript
    if "pending_approval" not in updates:
        updates["pending_approval"] = pending_approval
    return state.model_copy(update=updates)


def _is_stale_run_event(state: TuiState, event: RuntimeEvent) -> bool:
    """Reject any run-scoped event from an older run, not just terminal events.

    Unscoped events (``run_id is None``), such as local notices, always pass.
    """
    return (
        event.run_id is not None
        and state.active_run_id is not None
        and event.run_id != state.active_run_id
    )


def _event_matches_active_run(state: TuiState, event: RuntimeEvent) -> bool:
    """Accept unscoped lifecycle events and events for the active run only."""
    return (
        event.run_id is None
        or state.active_run_id is None
        or event.run_id == state.active_run_id
    )


def _projected_transcript(value: object) -> list[TranscriptItem]:
    if not isinstance(value, list):
        return []
    transcript: list[TranscriptItem] = []
    for item in value:
        try:
            if isinstance(item, dict):
                raw_tool_status = item.get("tool_status")
                if raw_tool_status is None:
                    tool_status = "success"
                else:
                    try:
                        tool_status = TypeAdapter(
                            Literal["running", "success", "error", "cancelled"]
                        ).validate_python(raw_tool_status, strict=True)
                    except (TypeError, ValueError):
                        continue
                command = item.get("command")
                metadata = item.get("metadata")
                if command is not None and not isinstance(command, str):
                    continue
                if metadata is not None and not isinstance(metadata, dict):
                    continue
                validated_item = {
                    key: value
                    for key, value in item.items()
                    if key not in ("tool_status", "command", "metadata")
                }
            else:
                tool_status = "success"
                command = None
                metadata = None
                validated_item = item
            session_message = SessionMessage.model_validate(validated_item)
        except (TypeError, ValueError):
            continue
        message = session_message.message
        if message.role not in {"user", "assistant", "tool"}:
            continue
        if message.role == "tool":
            call_id = _non_empty_str(message.tool_call_id)
            if call_id is None:
                continue
            transcript.append(
                TranscriptItem(
                    kind="tool",
                    item_id=session_message.record_id,
                    tool_call_id=call_id,
                    tool_name=_optional_str(message.name, None),
                    text=_text(message.content),
                    tool_status=tool_status,
                    command=command,
                    elapsed_seconds=_float_or_none(
                        metadata.get("elapsed_seconds") if metadata else None
                    ),
                    truncated=_bool_or_none(
                        metadata.get("truncated") if metadata else None
                    ),
                    exit_code=_int_or_none(
                        metadata.get("exit_code") if metadata else None
                    ),
                )
            )
        else:
            transcript.append(
                TranscriptItem(
                    kind=message.role,
                    item_id=session_message.record_id,
                    text=_text(message.content),
                )
            )
    return transcript


def _append_or_update(
    transcript: list[TranscriptItem],
    item: TranscriptItem,
    *,
    tool_call_id: str | None = None,
) -> None:
    if tool_call_id:
        index = _find_tool(transcript, tool_call_id)
    else:
        # Match on kind + item_id so re-applied events (e.g. a repeated
        # user_message) update the existing row instead of duplicating it.
        index = next(
            (
                i
                for i, row in enumerate(transcript)
                if row.kind == item.kind and row.item_id == item.item_id
            ),
            None,
        )
    if index is None:
        transcript.append(item)
    else:
        transcript[index] = transcript[index].model_copy(
            update=item.model_dump(exclude_unset=True)
        )


def _append_system(
    transcript: list[TranscriptItem], message: object, *, level: str = "notice"
) -> None:
    transcript.append(
        TranscriptItem(
            kind="system",
            item_id=_next_suffixed_id(transcript, "system-", kind="system"),
            text=_text(message),
            level=level,
        )
    )


def _next_suffixed_id(
    transcript: list[TranscriptItem], prefix: str, *, kind: str
) -> str:
    """Return the next unique ``prefix<number>`` id for rows of ``kind``.

    Session resets keep old local_command/system rows while rebuilding the
    transcript, so a length-based id could collide with a retained row.
    Counting existing ids of the same kind keeps ids unique across resets.
    """
    used = [
        int(row.item_id.removeprefix(prefix))
        for row in transcript
        if row.kind == kind and row.item_id.startswith(prefix)
    ]
    return f"{prefix}{max(used, default=-1) + 1}"


def _find_assistant(transcript: list[TranscriptItem], message_id: str) -> int | None:
    return next(
        (
            index
            for index, row in enumerate(transcript)
            if row.kind == "assistant" and row.item_id == message_id
        ),
        None,
    )


def _find_tool(transcript: list[TranscriptItem], call_id: str) -> int | None:
    return next(
        (
            index
            for index, row in enumerate(transcript)
            if row.kind == "tool" and row.tool_call_id == call_id
        ),
        None,
    )


def _approval(value: object) -> ApprovalRequest | None:
    if isinstance(value, ApprovalRequest):
        return value.model_copy(deep=True)
    if isinstance(value, dict):
        try:
            return ApprovalRequest.model_validate(value).model_copy(deep=True)
        except (TypeError, ValueError):
            return None
    return None


def _terminal_status(reason: object) -> str:
    if reason == "aborted":
        return "aborted"
    if reason in {"provider_error", "session_error"}:
        return "error"
    return "idle"


def _tool_status(value: object, ok: bool, error: str, text: str) -> str:
    if value == "cancelled":
        return "cancelled"
    if value == "error":
        return "error"
    if ok:
        return "success"
    return _tool_error_status(error, text)


def _tool_error_status(error: str, text: str) -> str:
    combined = f"{error} {text}".lower()
    return "cancelled" if re.search(r"\bcancelled\b", combined) else "error"


def _arguments_text(value: object) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _command_text(arguments: object, tool_name: str | None) -> str | None:
    """Derive a compact tool-row command label from the call arguments."""
    if not isinstance(arguments, dict):
        return None
    if tool_name == "run_command":
        command = arguments.get("command")
        return command if isinstance(command, str) and command.strip() else None
    pairs = [f"{key}={value}" for key, value in arguments.items()]
    return ", ".join(pairs) if pairs else None


def _metadata(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text(value: object) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _optional_str(value: object, fallback: str | None) -> str | None:
    value = _non_empty_str(value)
    return value if value is not None else fallback


def _int(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _policy(value: object, fallback: str) -> str:
    return value if value in {"default", "workspace", "full"} else fallback
