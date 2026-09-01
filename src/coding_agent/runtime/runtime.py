from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal

from coding_agent.context.policy import ContextPolicy
from coding_agent.llm.openai_compatible import redact_secrets
from coding_agent.llm.protocol import LLMProvider
from coding_agent.policy.approval import ApprovalPolicy, PermissionMode
from coding_agent.policy.memory import Scope
from coding_agent.runtime.events import EventSink, RuntimeEvent
from coding_agent.runtime.models import Message, RuntimeStatus, ToolCall, TurnOutcome
from coding_agent.runtime.runner import AgentRunner
from coding_agent.session.models import ApprovalRequest, SessionRecord, SessionSummary
from coding_agent.session.store import SessionStore
from coding_agent.tools.filesystem import _atomic_write
from coding_agent.tools.models import ToolResult


def _projected_tool_status(result: ToolResult) -> str:
    if result.ok:
        return "success"
    if result.error and "cancelled" in result.error.lower():
        return "cancelled"
    return "error"


def _projected_command(call: ToolCall | None) -> str | None:
    """Derive the compact tool-row command label from a persisted tool call."""
    if call is None or not call.arguments:
        return None
    if call.name == "run_command":
        command = call.arguments.get("command")
        return command if isinstance(command, str) and command.strip() else None
    pairs = [f"{key}={value}" for key, value in call.arguments.items()]
    return ", ".join(pairs) if pairs else None


_SUMMARY_MAX_CHARS = 4000


def _find_user_message_record(
    records: list[SessionRecord], message_id: str
) -> tuple[int, SessionRecord] | None:
    """Locate the ``user_message`` record for ``user-<turn_id>``, or None."""
    prefix = "user-"
    if not message_id.startswith(prefix):
        return None
    turn_id = message_id[len(prefix) :]
    for index, record in enumerate(records):
        if record.type == "user_message" and record.turn_id == turn_id:
            return index, record
    return None


async def _default_summarize(
    provider: LLMProvider, messages: list[Message], *, model: str
) -> str | None:
    """Collect a model-generated summary via ``provider.stream`` with no tools.

    Reuses the existing provider protocol unchanged; any failure or empty
    output returns None so the caller falls back to silent truncation.
    """
    try:
        stream = provider.stream(messages, [], model=model, signal=asyncio.Event())
        parts: list[str] = []
        async for event in stream:
            if event.type == "text_delta" and event.text:
                parts.append(event.text)
            elif event.type == "error":
                return None
        text = "".join(parts).strip()
        if not text:
            return None
        if len(text) > _SUMMARY_MAX_CHARS:
            text = text[:_SUMMARY_MAX_CHARS]
        return text
    except Exception:  # noqa: BLE001 - summarization is best-effort
        return None


class _ApprovalBroker:
    def __init__(
        self,
        publish: Callable[[RuntimeEvent], Awaitable[None]],
        set_status: Callable[[str], None],
    ) -> None:
        self._publish = publish
        self._set_status = set_status
        self.pending: ApprovalRequest | None = None
        self._future: asyncio.Future[str] | None = None
        self.last_feedback: str | None = None

    async def request(
        self, request: ApprovalRequest
    ) -> Literal["approve", "deny", "cancelled"]:
        if self.pending is not None:
            return "cancelled"
        self.pending = request
        self._set_status("waiting_approval")
        self._future = asyncio.get_running_loop().create_future()
        await self._publish(
            RuntimeEvent(
                type="approval_requested",
                run_id=request.run_id,
                payload={"request": request},
            )
        )
        try:
            return await self._future  # type: ignore[return-value]
        finally:
            self.pending = None
            self._future = None
            self._set_status("running")

    async def resolve(
        self,
        request_id: str,
        decision: Literal["approve", "deny"],
        remember: Scope = "once",
        feedback: str | None = None,
    ) -> dict:
        request = self.pending
        future = self._future
        if (
            request is None
            or request.request_id != request_id
            or future is None
            or future.done()
            or request.status != "pending"
        ):
            raise RuntimeError("approval not pending")
        status = "approved" if decision == "approve" else "denied"
        request = request.model_copy(update={"status": status})
        self.pending = request
        self.last_feedback = feedback
        future.set_result(decision)
        await self._publish(
            RuntimeEvent(
                type="approval_resolved",
                run_id=request.run_id,
                payload={
                    "request_id": request_id,
                    "decision": decision,
                    "status": status,
                    "remember": remember,
                    "feedback": feedback,
                },
            )
        )
        return {
            "request_id": request_id,
            "tool_name": request.tool_name,
            "decision": decision,
            "scope": remember,
            "feedback": feedback,
            "tool_call_id": request.tool_call_id,
            "run_id": request.run_id,
        }

    def cancel_all(self) -> None:
        future = self._future
        request = self.pending
        if (
            future is None
            or future.done()
            or request is None
            or request.status != "pending"
        ):
            return
        request = request.model_copy(update={"status": "cancelled"})
        self.pending = request
        future.set_result("cancelled")
        asyncio.create_task(
            self._publish(
                RuntimeEvent(
                    type="approval_resolved",
                    run_id=request.run_id,
                    payload={
                        "request_id": request.request_id,
                        "decision": "deny",
                        "status": "cancelled",
                    },
                )
            )
        )


class AgentRuntime:
    def __init__(
        self,
        *,
        store: SessionStore,
        runner_factory: Callable[
            [SessionStore, ContextPolicy, _ApprovalBroker], AgentRunner
        ],
        context_policy_factory: Callable[[], ContextPolicy],
        approval_policy: ApprovalPolicy,
        system_prompt: Message,
        model: str,
        permission_mode: PermissionMode = "default",
        summarizer: Callable[[list[Message]], Awaitable[str]] | None = None,
    ) -> None:
        self.store = store
        self._runner_factory = runner_factory
        self._context_policy_factory = context_policy_factory
        self._approval_policy = approval_policy
        self._system_prompt = system_prompt
        self._model = model
        self._permission_mode = permission_mode
        self._summarizer = summarizer
        self._subscribers: list[EventSink] = []
        self._status = RuntimeStatus()
        self._last_outcome: TurnOutcome | None = None
        self._task: asyncio.Task[None] | None = None
        self._submit_in_progress = False
        self._operation_in_progress = False
        self._signal: asyncio.Event | None = None
        self._run_id: str | None = None
        self._turn_id: str | None = None
        self._forced_aborts: set[str] = set()
        self._terminal_events: set[str] = set()
        self._broker = _ApprovalBroker(self._publish, self._set_status)
        self._runner = self._make_runner()

    def _make_runner(self) -> AgentRunner:
        runner = self._runner_factory(
            self.store, self._context_policy_factory(), self._broker
        )
        # The runtime owns subscriber fan-out; runners must not bypass it.
        runner.event_sink = self._publish
        runner.permission_mode = self._permission_mode
        return runner

    @property
    def session_id(self) -> str:
        return self.store.session_id

    @property
    def permission_mode(self) -> PermissionMode:
        return self._permission_mode

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def last_outcome(self) -> TurnOutcome | None:
        return self._last_outcome

    def subscribe(self, sink: EventSink) -> Callable[[], None]:
        self._subscribers.append(sink)

        def unsubscribe() -> None:
            if sink in self._subscribers:
                self._subscribers.remove(sink)

        return unsubscribe

    async def _publish(self, event: RuntimeEvent) -> None:
        if event.type == "context_updated":
            self._status = self._status.model_copy(
                update={
                    "context_used": event.payload.get(
                        "used_tokens", self._status.context_used
                    ),
                    "context_window": event.payload.get(
                        "context_window", self._status.context_window
                    ),
                    "context_estimated": event.payload.get(
                        "estimated", self._status.context_estimated
                    ),
                }
            )
        for sink in list(self._subscribers):
            try:
                await sink(event)
            except Exception:  # noqa: BLE001, S112 - sink isolation is intentional
                continue

    async def _emit(self, kind: str, *, turn_id: str | None = None, **payload) -> None:
        await self._publish(
            RuntimeEvent(
                type=kind,
                run_id=self._run_id,
                turn_id=turn_id if turn_id is not None else self._turn_id,
                payload=payload,
            )
        )

    def _set_status(self, status: str) -> None:
        self._status = self._status.model_copy(update={"status": status})

    def _ensure_no_active_run(self) -> None:
        if (
            self._operation_in_progress
            or self._submit_in_progress
            or (self._task is not None and not self._task.done())
        ):
            raise RuntimeError("active run")

    async def _reserve_operation(self) -> None:
        self._ensure_no_active_run()
        self._operation_in_progress = True

    def _release_operation(self) -> None:
        self._operation_in_progress = False

    async def submit(self, prompt: str) -> str:
        await self._reserve_operation()
        self._submit_in_progress = True
        run_id: str | None = None
        turn_id: str | None = None
        try:
            run_id, turn_id = uuid.uuid4().hex, uuid.uuid4().hex
            self._run_id = run_id
            self._turn_id = turn_id
            self._signal = asyncio.Event()
            self._status = RuntimeStatus(
                status="running", run_id=run_id, turn_id=turn_id
            )
            self.store.append_new(
                "turn_start", {"turn_id": turn_id}, run_id=run_id, turn_id=turn_id
            )
            self.store.append_new(
                "user_message",
                {"message": Message(role="user", content=prompt)},
                run_id=run_id,
                turn_id=turn_id,
            )
            await self._emit(
                "run_started",
                session_id=self.session_id,
                model=self._model,
                policy=self._permission_mode,
                turn_id=turn_id,
            )
            await self._emit(
                "user_message",
                message_id=f"user-{turn_id}",
                text=prompt,
                turn_id=turn_id,
            )
            self._task = asyncio.create_task(
                self._run(prompt, run_id, turn_id, self._signal)
            )
            self._release_operation()
            return run_id
        except BaseException:
            if run_id is not None and turn_id is not None:
                with contextlib.suppress(Exception):
                    self.store.append_new(
                        "turn_end",
                        {"reason": "aborted"},
                        run_id=run_id,
                        turn_id=turn_id,
                    )
            self._status = RuntimeStatus(
                context_window=self.store.header.context_window
            )
            self._run_id = None
            self._turn_id = None
            self._signal = None
            self._submit_in_progress = False
            self._release_operation()
            raise

    async def _run(
        self, prompt: str, run_id: str, turn_id: str, signal: asyncio.Event
    ) -> None:
        turn_closed = False
        try:
            outcome = await self._runner.run_turn(
                prompt, run_id=run_id, turn_id=turn_id, signal=signal
            )
            if run_id in self._forced_aborts:
                return
            self._last_outcome = outcome
            self._status = RuntimeStatus(
                status="aborted"
                if outcome.reason == "aborted"
                else ("error" if outcome.reason.endswith("error") else "idle"),
                usage=outcome.usage,
                run_id=run_id,
                turn_id=turn_id,
                context_used=self._status.context_used,
                context_window=self._status.context_window,
                context_estimated=self._status.context_estimated,
            )
            self.store.append_new(
                "turn_end",
                {"reason": outcome.reason, "outcome": outcome.model_dump()},
                run_id=run_id,
                turn_id=turn_id,
            )
            turn_closed = True
            self._terminal_events.add(run_id)
            await self._emit(
                "run_finished", outcome=outcome.model_dump(), steps=outcome.steps
            )
        except asyncio.CancelledError:
            if not turn_closed and run_id not in self._forced_aborts:
                with contextlib.suppress(Exception):
                    self.store.append_new(
                        "turn_end",
                        {"reason": "aborted"},
                        run_id=run_id,
                        turn_id=turn_id,
                    )
            raise
        except Exception as exc:  # noqa: BLE001 - runtime errors become events
            if run_id in self._forced_aborts:
                return
            self._status = RuntimeStatus(status="error", run_id=run_id, turn_id=turn_id)
            message = redact_secrets(str(exc))
            with contextlib.suppress(Exception):
                self.store.append_new(
                    "turn_end",
                    {"reason": "runtime_error", "error": message},
                    run_id=run_id,
                    turn_id=turn_id,
                )
            turn_closed = True
            await self._emit(
                "run_error", code="runtime_error", message=message, recoverable=False
            )
        finally:
            if self._task is asyncio.current_task():
                self._task = None
                self._submit_in_progress = False
                self._signal = None
                self._run_id = None
                self._turn_id = None
                if self._status.status != "error":
                    self._status = self._status.model_copy(
                        update={
                            "status": "idle"
                            if self._status.status != "aborted"
                            else "aborted"
                        }
                    )
            self._forced_aborts.discard(run_id)

    async def abort(self, run_id: str) -> None:
        if self._task is None or self._task.done() or run_id != self._run_id:
            return
        if self._signal:
            self._signal.set()
        self._broker.cancel_all()
        task = self._task
        turn_id = self._turn_id
        try:
            await asyncio.wait_for(asyncio.shield(task), 5.0)
        except TimeoutError:
            self._forced_aborts.add(run_id)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), 1.0)
            if run_id not in self._terminal_events:
                with contextlib.suppress(Exception):
                    self.store.append_new(
                        "turn_end",
                        {"reason": "aborted"},
                        run_id=run_id,
                        turn_id=turn_id,
                    )
                self._terminal_events.add(run_id)
                await self._publish(
                    RuntimeEvent(
                        type="run_finished",
                        run_id=run_id,
                        turn_id=turn_id,
                        payload={
                            "outcome": TurnOutcome(
                                reason="aborted", steps=0
                            ).model_dump(),
                            "steps": 0,
                        },
                    )
                )
            self._status = self._status.model_copy(update={"status": "aborted"})
            if self._task is task:
                self._task = None
                self._submit_in_progress = False
                self._signal = None
                self._run_id = None
                self._turn_id = None
        self._status = self._status.model_copy(update={"status": "aborted"})

    async def resolve_approval(
        self,
        request_id: str,
        decision: Literal["approve", "deny"],
        remember: Scope = "once",
        feedback: str | None = None,
    ) -> None:
        resolved = await self._broker.resolve(
            request_id, decision, remember=remember, feedback=feedback
        )
        self.store.append_new(
            "approval",
            {
                "request_id": resolved["request_id"],
                "tool_name": resolved["tool_name"],
                "decision": resolved["decision"],
                "scope": resolved["scope"],
                "feedback": resolved["feedback"],
                "tool_call_id": resolved["tool_call_id"],
            },
            run_id=resolved.get("run_id") or self._run_id,
            turn_id=self._turn_id,
        )

    async def set_permission(self, mode: PermissionMode) -> None:
        await self._reserve_operation()
        try:
            previous_mode = self._permission_mode
            self._permission_mode = mode
            self._runner.permission_mode = mode
            self.store.append_new("policy_changed", {"policy": mode})
            await self._emit(
                "policy_changed", policy=mode, previous_policy=previous_mode
            )
        finally:
            self._release_operation()

    async def new_session(self) -> str:
        await self._reserve_operation()
        try:
            self.store = SessionStore.create(
                self.store.path.parent,
                workspace=self.store.header.workspace,
                model=self._model,
                context_window=self.store.header.context_window,
            )
            self._permission_mode = "default"
            self._last_outcome = None
            self._status = RuntimeStatus(
                context_window=self.store.header.context_window
            )
            self._runner = self._make_runner()
            await self._publish(
                RuntimeEvent(
                    type="session_loaded",
                    payload={
                        "session_id": self.session_id,
                        "workspace": self.store.header.workspace,
                        "model": self.store.header.model,
                        "context_window": self.store.header.context_window,
                        "history": [],
                    },
                )
            )
            return self.session_id
        finally:
            self._release_operation()

    async def list_sessions(self) -> list[SessionSummary]:
        return SessionStore.list_sessions(self.store.path.parent)

    async def resume(self, session_id: str) -> None:
        await self._reserve_operation()
        try:
            self.store = SessionStore.open(self.store.path.parent, session_id)
            self.store.mark_open_final_turn_interrupted()
            self._model = self.store.header.model
            self._permission_mode = "default"
            self._last_outcome = None
            self._status = RuntimeStatus(
                context_window=self.store.header.context_window
            )
            self._runner = self._make_runner()
            history = [
                item.model_dump(mode="json")
                for item in self.store.project_messages(include_open_turn=False)
            ]
            tool_results = {
                result.tool_call_id: result
                for record in self.store.records()
                if record.type == "tool_result"
                for result in [ToolResult.model_validate(record.payload["result"])]
            }
            tool_calls = {
                call.id: call
                for record in self.store.records()
                if record.type == "tool_call"
                for call in [ToolCall.model_validate(record.payload["tool_call"])]
            }
            for item in history:
                message = item.get("message", {})
                if message.get("role") != "tool":
                    continue
                tool_call_id = message.get("tool_call_id")
                result = tool_results.get(tool_call_id)
                call = tool_calls.get(tool_call_id)
                if result is not None:
                    item["tool_status"] = _projected_tool_status(result)
                    if result.metadata:
                        item["metadata"] = result.metadata
                command = _projected_command(call)
                if command:
                    item["command"] = command
            await self._publish(
                RuntimeEvent(
                    type="session_loaded",
                    payload={
                        "session_id": session_id,
                        "workspace": self.store.header.workspace,
                        "model": self.store.header.model,
                        "context_window": self.store.header.context_window,
                        "history": history,
                    },
                )
            )
            if self.store.load_notice:
                await self._publish(
                    RuntimeEvent(
                        type="notice",
                        payload={"level": "warning", "message": self.store.load_notice},
                    )
                )
        finally:
            self._release_operation()

    async def fork_at(self, message_id: str) -> str:
        """Fork the current session at a past user message.

        Creates a new session whose persisted records end at that user
        message, swaps the runtime onto it, and returns the prompt text so the
        TUI can refill the composer. The original session is untouched.
        """
        await self._reserve_operation()
        try:
            found = _find_user_message_record(self.store.records(), message_id)
            if found is None:
                raise ValueError("message not found in session history")
            index, record = found
            prompt = Message.model_validate(record.payload["message"]).content or ""
            records = self.store.records()
            # End the fork at the selected message. If the turn is closed by an
            # immediately following turn_end, keep it so the turn stays closed
            # and the user message still projects into the new session history.
            end = index + 1
            if (
                end < len(records)
                and records[end].type == "turn_end"
                and records[end].turn_id == record.turn_id
            ):
                end += 1
            prefix = records[:end]
            new_store = SessionStore.create(
                self.store.path.parent,
                workspace=self.store.header.workspace,
                model=self._model,
                context_window=self.store.header.context_window,
                title=f"Fork of {self.store.session_id[:8]}",
            )
            for item in prefix:
                new_store.append(item)
            self.store = new_store
            self._permission_mode = "default"
            self._last_outcome = None
            self._status = RuntimeStatus(context_window=new_store.header.context_window)
            self._runner = self._make_runner()
            await self._publish(
                RuntimeEvent(
                    type="session_loaded",
                    payload={
                        "session_id": self.session_id,
                        "workspace": new_store.header.workspace,
                        "model": new_store.header.model,
                        "context_window": new_store.header.context_window,
                        "history": [
                            item.model_dump(mode="json")
                            for item in new_store.project_messages(
                                include_open_turn=False
                            )
                        ],
                    },
                )
            )
            return prompt
        finally:
            self._release_operation()

    async def compact(self) -> None:
        await self._reserve_operation()
        try:
            await self._compact()
        finally:
            self._release_operation()

    async def _compact(self) -> None:
        history = self.store.project_messages(include_open_turn=True)
        policy = self._context_policy_factory()
        view = policy.prepare(
            history,
            system_prompt=self._system_prompt,
            context_window=self.store.header.context_window,
            usage=None,
            force=True,
        )
        await self._publish(
            RuntimeEvent(
                type="context_updated",
                payload={
                    "used_tokens": view.used_tokens,
                    "context_window": view.context_window,
                    "estimated": view.estimated,
                    "compacted": view.compacted,
                },
            )
        )
        if not view.compacted:
            await self._publish(
                RuntimeEvent(
                    type="notice",
                    payload={"level": "info", "message": "nothing to compact"},
                )
            )
            return
        turns = list(dict.fromkeys(item.turn_id for item in history if item.turn_id))
        removed = turns[: view.removed_turns]
        retained = turns[view.removed_turns :]
        tokens_before = sum(
            max(1, len(item.message.content or "") // 4) for item in [*history]
        ) + max(1, len(self._system_prompt.content or "") // 4)
        payload: dict = {
            "strategy": "turn_truncate",
            "removed_turn_ids": removed,
            "retained_turn_ids": retained,
            "tokens_before": tokens_before,
            "tokens_after": view.used_tokens,
            "forced": True,
        }
        if removed:
            removed_messages = [
                item.message for item in history if item.turn_id in removed
            ]
            summary = await self._summarize(removed_messages)
            if summary:
                payload["summary"] = summary
        self.store.append_new("compaction", payload)

    async def _summarize(self, messages: list[Message]) -> str | None:
        """Summarize dropped messages, returning None when unavailable/failed."""
        if self._summarizer is not None:
            try:
                return await self._summarizer(messages)
            except Exception:  # noqa: BLE001 - summarization is best-effort
                return None
        provider = getattr(self._runner, "provider", None)
        if provider is None:
            return None
        return await _default_summarize(provider, messages, model=self._model)

    async def undo(self) -> None:
        """Restore the most recent file mutation from the executor's journal.

        Pops the latest ``(path, original)`` snapshot recorded by the executor
        and restores the prior content (or unlinks the file when it did not
        exist), then emits a local notice so the transcript shows an undo row.
        An empty journal is a no-op that still emits a notice.
        """
        await self._reserve_operation()
        try:
            journal = getattr(self._runner.executor, "journal", None)
            entry = journal.pop() if journal is not None else None
            if entry is None:
                await self._emit("notice", level="info", message="nothing to undo")
                return
            path, original = entry
            if original is None:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
            else:
                _atomic_write(path, original)
            await self._emit(
                "notice",
                command=f"undo {path}",
                message=f"undid write to {path}",
            )
        finally:
            self._release_operation()
