from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from typing import Literal

from coding_agent.context.policy import ContextPolicy
from coding_agent.policy.approval import ApprovalPolicy, PermissionMode
from coding_agent.runtime.events import EventSink, RuntimeEvent
from coding_agent.runtime.models import Message, RuntimeStatus, TurnOutcome
from coding_agent.runtime.runner import AgentRunner
from coding_agent.session.models import ApprovalRequest, SessionSummary
from coding_agent.session.store import SessionStore


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
        self, request_id: str, decision: Literal["approve", "deny"]
    ) -> None:
        request = self.pending
        if (
            request is None
            or request.request_id != request_id
            or self._future is None
            or self._future.done()
            or request.status != "pending"
        ):
            raise RuntimeError("approval not pending")
        status = "approved" if decision == "approve" else "denied"
        request = request.model_copy(update={"status": status})
        self.pending = request
        await self._publish(
            RuntimeEvent(
                type="approval_resolved",
                run_id=request.run_id,
                payload={
                    "request_id": request_id,
                    "decision": decision,
                    "status": status,
                },
            )
        )
        if not self._future.done():
            self._future.set_result(decision)

    def cancel_all(self) -> None:
        if self._future is None or self._future.done() or self.pending is None:
            return
        request = self.pending.model_copy(update={"status": "cancelled"})
        self.pending = request
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
        self._future.set_result("cancelled")


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
    ) -> None:
        self.store = store
        self._runner_factory = runner_factory
        self._context_policy_factory = context_policy_factory
        self._approval_policy = approval_policy
        self._system_prompt = system_prompt
        self._model = model
        self._permission_mode = permission_mode
        self._subscribers: list[EventSink] = []
        self._status = RuntimeStatus()
        self._last_outcome: TurnOutcome | None = None
        self._task: asyncio.Task[None] | None = None
        self._signal: asyncio.Event | None = None
        self._run_id: str | None = None
        self._broker = _ApprovalBroker(self._publish, self._set_status)
        self._runner = self._make_runner()

    def _make_runner(self) -> AgentRunner:
        return self._runner_factory(
            self.store, self._context_policy_factory(), self._broker
        )

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
        for sink in list(self._subscribers):
            try:
                await sink(event)
            except Exception:  # noqa: BLE001, S112 - sink isolation is intentional
                continue

    async def _emit(self, kind: str, **payload) -> None:
        await self._publish(
            RuntimeEvent(type=kind, run_id=self._run_id, payload=payload)
        )

    def _set_status(self, status: str) -> None:
        self._status = self._status.model_copy(update={"status": status})

    async def submit(self, prompt: str) -> str:
        if self._task is not None and not self._task.done():
            raise RuntimeError("active run")
        run_id, turn_id = uuid.uuid4().hex, uuid.uuid4().hex
        self._run_id = run_id
        self._signal = asyncio.Event()
        self._status = RuntimeStatus(status="running", run_id=run_id, turn_id=turn_id)
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
        )
        self._task = asyncio.create_task(
            self._run(prompt, run_id, turn_id, self._signal)
        )
        return run_id

    async def _run(
        self, prompt: str, run_id: str, turn_id: str, signal: asyncio.Event
    ) -> None:
        try:
            outcome = await self._runner.run_turn(
                prompt, run_id=run_id, turn_id=turn_id, signal=signal
            )
            self._last_outcome = outcome
            self._status = RuntimeStatus(
                status="aborted"
                if outcome.reason == "aborted"
                else ("error" if outcome.reason.endswith("error") else "idle"),
                usage=outcome.usage,
                run_id=run_id,
                turn_id=turn_id,
            )
            self.store.append_new(
                "turn_end",
                {"reason": outcome.reason, "outcome": outcome.model_dump()},
                run_id=run_id,
                turn_id=turn_id,
            )
            await self._emit(
                "run_finished", outcome=outcome.model_dump(), steps=outcome.steps
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - runtime errors become events
            self._status = RuntimeStatus(status="error", run_id=run_id, turn_id=turn_id)
            await self._emit(
                "run_error", code="runtime_error", message=str(exc), recoverable=False
            )
        finally:
            self._task = None
            self._signal = None
            self._run_id = None
            if self._status.status != "error":
                self._status = self._status.model_copy(
                    update={
                        "status": "idle"
                        if self._status.status != "aborted"
                        else "aborted"
                    }
                )

    async def abort(self, run_id: str) -> None:
        if self._task is None or self._task.done() or run_id != self._run_id:
            return
        if self._signal:
            self._signal.set()
        self._broker.cancel_all()
        task = self._task
        try:
            await asyncio.wait_for(asyncio.shield(task), 5.0)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._status = self._status.model_copy(update={"status": "idle"})

    async def resolve_approval(
        self, request_id: str, decision: Literal["approve", "deny"]
    ) -> None:
        await self._broker.resolve(request_id, decision)

    async def set_permission(self, mode: PermissionMode) -> None:
        self._permission_mode = mode
        await self._emit("policy_changed", policy=mode)

    async def new_session(self) -> str:
        if self._task is not None and not self._task.done():
            raise RuntimeError("active run")
        self.store = SessionStore.create(
            self.store.path.parent,
            workspace=self.store.header.workspace,
            model=self._model,
            context_window=self.store.header.context_window,
        )
        self._permission_mode = "default"
        self._runner = self._make_runner()
        await self._publish(
            RuntimeEvent(type="session_loaded", payload={"session_id": self.session_id})
        )
        return self.session_id

    async def list_sessions(self) -> list[SessionSummary]:
        return SessionStore.list_sessions(self.store.path.parent)

    async def resume(self, session_id: str) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError("active run")
        self.store = SessionStore.open(self.store.path.parent, session_id)
        self._model = self.store.header.model
        self._permission_mode = "default"
        self._runner = self._make_runner()
        await self._publish(
            RuntimeEvent(type="session_loaded", payload={"session_id": session_id})
        )
        if self.store.load_notice:
            await self._publish(
                RuntimeEvent(
                    type="notice",
                    payload={"level": "warning", "message": self.store.load_notice},
                )
            )

    async def compact(self) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError("active run")
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
        self.store.append_new(
            "compaction",
            {
                "strategy": "turn_truncate",
                "removed_turn_ids": removed,
                "retained_turn_ids": retained,
                "tokens_before": policy.estimate_tokens(
                    [self._system_prompt] + [i.message for i in history]
                ),
                "tokens_after": view.used_tokens,
                "forced": True,
            },
        )
