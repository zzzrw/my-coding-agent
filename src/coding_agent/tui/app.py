from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any, ClassVar

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.commands import SUPPORTED_COMMANDS, parse_command
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import TuiState, initial_state
from coding_agent.tui.widgets import (
    ApprovalScreen,
    CommandComposer,
    PermissionFullScreen,
    PermissionModeScreen,
    SessionSelector,
    StatusLine,
    SubmitTextArea,
    TranscriptView,
)

_COMPACTING_CONFLICTS = frozenset(
    {"new", "session", "resume", "compact", "permission", "clear"}
)


class _RuntimeBridge:
    """Bounded event sink and loop-owned reducer bridge."""

    def __init__(self, app: CodingAgentApp, *, maxsize: int = 256) -> None:
        self.app = app
        self.queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=maxsize)
        self._coalesced: dict[tuple[int, str], RuntimeEvent] = {}
        self._coalesced_order: list[tuple[int, str]] = []
        self._max_coalesced = max(maxsize, 2)
        self._generation = 0
        self._publish_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._stopped = False
        self._task: asyncio.Task[None] | None = None
        self._queued_delta_event_id: str | None = None
        self._last_delta: RuntimeEvent | None = None

    def start(self) -> None:
        self._stopped = False
        self._task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        self._stopped = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def publish(self, event: RuntimeEvent) -> None:
        async with self._publish_lock:
            await self._publish_locked(event)

    def _buffer_delta(self, key: tuple[int, str], event: RuntimeEvent) -> None:
        existing = self._coalesced.get(key)
        if existing is not None:
            old_text = existing.payload.get("text", "")
            new_text = event.payload.get("text", "")
            if isinstance(old_text, str) and isinstance(new_text, str):
                event = event.model_copy(
                    update={
                        "payload": {
                            **event.payload,
                            "text": old_text + new_text,
                        }
                    }
                )
        else:
            self._coalesced_order.append(key)
        self._coalesced[key] = event

    def _pop_coalesced(self) -> RuntimeEvent:
        key = self._coalesced_order.pop(0)
        return self._coalesced.pop(key)

    async def _publish_locked(self, event: RuntimeEvent) -> None:
        if self._stopped:
            return
        if event.type == "assistant_delta":
            message_id = event.payload.get("message_id")
            if isinstance(message_id, str) and message_id:
                key = (self._generation, message_id)
                # A newer delta never enters the queue ahead of a buffered one.
                if self._coalesced and not self.queue.full():
                    self.queue.put_nowait(self._pop_coalesced())
                if self._coalesced or self.queue.full():
                    # Bound the overflow buffer: flush the oldest coalesced
                    # delta to the queue before buffering a new distinct one.
                    if len(self._coalesced) >= self._max_coalesced:
                        await self.queue.put(self._pop_coalesced())
                    self._buffer_delta(key, event)
                    self._wake.set()
                    return
                self.queue.put_nowait(event)
            else:
                # Unidentified deltas cannot be coalesced without losing text.
                await self.queue.put(event)
        else:
            # Control events wait for capacity and every earlier coalesced
            # delta; neither buffering nor draining may let control overtake.
            while self._coalesced:
                await self.queue.put(self._pop_coalesced())
            await self.queue.put(event)
            self._generation += 1
        self._wake.set()

    async def _drain(self) -> None:
        while not self._stopped:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except TimeoutError:
                if self._coalesced:
                    event = self._pop_coalesced()
                else:
                    continue
            try:
                if event.type not in {"assistant_delta", "run_started"}:
                    pending = [
                        key for key in self._coalesced if key[0] < self._generation
                    ]
                    for key in pending:
                        coalesced = self._coalesced.pop(key)
                        self._coalesced_order.remove(key)
                        self.app._apply_event(coalesced)
                self.app._apply_event(event)
            except Exception as exc:  # noqa: BLE001 - bridge errors are visible
                self.app._show_error(f"TUI error: {exc}")


class CodingAgentApp(App[None]):
    """Three-region Textual shell around an injected runtime."""

    CSS = """
    Screen { layout: vertical; }
    #transcript { height: 1fr; width: 1fr; }
    #composer { height: 4; width: 1fr; }
    #composer-input { height: 4; width: 1fr; }
    #composer > #command-palette { width: 1fr; display: none; height: auto; max-height: 8; overlay: screen; constrain: none inside; border: tall $border-blurred; background: $surface; }
    #statusline { height: 1; width: 1fr; }
    ApprovalScreen, SessionSelector, PermissionModeScreen { align: center middle; }
    #approval, #permission-full, #permission-mode, #session-selector { width: 76; height: auto; padding: 1 2; border: round $accent; background: $surface; }
    #approval-details, #permission-full-details { height: auto; margin-bottom: 1; }
    #approve, #deny, #permission-full-approve, #permission-full-cancel { width: 1fr; margin: 0 1; }
    #session-options, #permission-mode-options { height: auto; max-height: 18; }
    #session-selector-title, #permission-mode-title { margin-bottom: 1; text-style: bold; }
    """
    BINDINGS: ClassVar = [Binding("ctrl+c", "interrupt", "Abort", priority=True)]

    def __init__(
        self,
        *,
        runtime: Any,
        initial_state: TuiState | None = None,
        queue_size: int = 256,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.state = initial_state or _state_from_runtime(runtime)
        self._queue_size = queue_size
        self._bridge: _RuntimeBridge | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._approval_request_id: str | None = None
        self._submit_scheduled = False
        self._submit_settled = asyncio.Event()
        self._submit_settled.set()
        self._shutdown_requested = False
        self._shutdown_abort_started = False
        self._abort_tasks: dict[str, asyncio.Task[None]] = {}
        self._submitted_run_id: str | None = None

    def compose(self) -> ComposeResult:
        yield TranscriptView(id="transcript")
        yield CommandComposer(
            self.state.input_text,
            id="composer",
        )
        yield StatusLine(self.state, id="statusline")

    async def on_mount(self) -> None:
        self._bridge = _RuntimeBridge(self, maxsize=self._queue_size)
        self._unsubscribe = self.runtime.subscribe(self._bridge.publish)
        self._bridge.start()
        await self._refresh_widgets()
        self.query_one("#composer-input", SubmitTextArea).focus()

    async def on_unmount(self, event: events.Unmount) -> None:
        del event
        await self._submit_settled.wait()
        await self._settle_runtime_before_teardown()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._bridge is not None:
            await self._bridge.stop()
            self._bridge = None

    async def on_submit_text_area_submitted(
        self, event: SubmitTextArea.Submitted
    ) -> None:
        event.stop()
        prompt = event.text.strip()
        if not prompt:
            return
        event.text_area.text = ""
        self.state = self.state.model_copy(update={"input_text": ""})
        command = parse_command(prompt)
        if command is not None:
            self._apply_event(
                RuntimeEvent(
                    type="notice",
                    payload={"command": prompt},
                )
            )
            self._dispatch_command(command.name, command.args)
            return
        if self.state.compacting:
            self._show_notice("compaction in progress")
            return
        if (
            self.state.status not in {"idle", "aborted", "error"}
            or self._submit_scheduled
        ):
            self._show_notice("A run is already active")
            return
        self._submit_scheduled = True
        self._submit_settled.clear()
        self._submitted_run_id = None
        self.run_worker(
            self._submit_prompt(prompt),
            name="submit-prompt",
            group="runtime",
            exit_on_error=False,
        )

    async def _submit_prompt(self, prompt: str) -> None:
        try:
            self._submitted_run_id = await self.runtime.submit(prompt)
        except Exception as exc:  # noqa: BLE001 - runtime errors become rows
            self._show_notice(str(exc), level="error")
        finally:
            self._submit_scheduled = False
            self._submit_settled.set()

    def _dispatch_command(self, name: str, args: list[str]) -> None:
        if name == "exit":
            name = "quit"
        if name in {
            "new",
            "session",
            "resume",
            "compact",
        } and self.state.status not in {"idle", "aborted", "error"}:
            self._show_notice("A run is already active")
            return
        if self.state.compacting and name in _COMPACTING_CONFLICTS:
            self._show_notice("compaction in progress")
            return
        if name not in SUPPORTED_COMMANDS:
            self._show_notice(f"unknown command: /{name}", level="error")
        elif name == "help":
            if args:
                self._show_notice("usage: /help")
                return
            self._show_notice(
                "commands: "
                + ", ".join(f"/{item}" for item in sorted(SUPPORTED_COMMANDS))
            )
        elif name == "context":
            if args:
                self._show_notice("usage: /context")
                return
            self._show_notice(self._context_notice())
        elif name == "clear":
            if args:
                self._show_notice("usage: /clear")
                return
            self.state = self.state.model_copy(
                update={
                    "transcript": [self.state.transcript[-1]]
                    if self.state.transcript
                    and self.state.transcript[-1].kind == "local_command"
                    else []
                }
            )
            self.call_after_refresh(self._refresh_widgets)
        elif name == "quit":
            if args:
                self._show_notice("usage: /quit")
                return
            self._request_shutdown()
        elif name == "permission":
            self._dispatch_permission(args)
        elif name == "resume":
            self._dispatch_resume(args)
        elif name == "session":
            if args:
                self._show_notice("usage: /session")
                return
            self.run_worker(
                self._open_session_selector(),
                name="list-sessions",
                group="runtime",
                exit_on_error=False,
            )
        elif name == "new":
            if args:
                self._show_notice("usage: /new")
                return
            self.run_worker(
                self._runtime_action("new_session"),
                name="new-session",
                group="runtime",
                exit_on_error=False,
            )
        elif name == "compact":
            if args:
                self._show_notice("usage: /compact")
                return
            self._start_compact()

    def _request_shutdown(self) -> None:
        """Abort any runtime-owned turn before Textual tears down the UI."""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._dismiss_transient_screens()
        self.run_worker(
            self._shutdown_then_exit(),
            name="shutdown-runtime",
            group="runtime",
            exit_on_error=False,
        )

    async def _shutdown_then_exit(self) -> None:
        await self._submit_settled.wait()
        await self._settle_runtime_before_teardown()
        self.exit()

    async def _settle_runtime_before_teardown(self) -> None:
        if self._shutdown_abort_started:
            return
        self._shutdown_abort_started = True
        run_id = self._shutdown_run_id()
        if run_id is not None:
            await self._abort_run(run_id)
        self._dismiss_transient_screens()

    def _shutdown_run_id(self) -> str | None:
        if self.state.pending_approval is not None:
            return self.state.pending_approval.run_id or self._submitted_run_id
        return self.state.active_run_id or self._submitted_run_id

    def _dismiss_transient_screens(self) -> None:
        if not self._screen_stack:
            self._approval_request_id = None
            return
        if isinstance(
            self.screen,
            (
                ApprovalScreen,
                PermissionFullScreen,
                PermissionModeScreen,
                SessionSelector,
            ),
        ):
            self.screen.dismiss(None)
        self._approval_request_id = None

    def _dispatch_permission(self, args: list[str]) -> None:
        if not args:
            self._show_notice(f"permission: {self.state.policy}")
            self.push_screen(
                PermissionModeScreen(self.state.policy),
                callback=self._permission_mode_decision,
            )
            return
        if len(args) != 1 or args[0] not in {"default", "workspace", "full"}:
            self._show_notice("usage: /permission default|workspace|full")
            return
        if args[0] == "full":
            self.push_screen(
                PermissionFullScreen(), callback=self._permission_full_decision
            )
            return
        self.run_worker(
            self._set_permission(args[0]),
            name="set-permission",
            group="runtime",
            exit_on_error=False,
        )

    def _permission_mode_decision(self, mode: str | None) -> None:
        if mode == "full":
            self.push_screen(
                PermissionFullScreen(), callback=self._permission_full_decision
            )
        elif mode in {"default", "workspace"}:
            self.run_worker(
                self._set_permission(mode),
                name="set-permission",
                group="runtime",
                exit_on_error=False,
            )

    def _permission_full_decision(self, approved: bool | None) -> None:
        if approved:
            self.run_worker(
                self._set_permission("full"),
                name="set-permission",
                group="runtime",
                exit_on_error=False,
            )

    def _dispatch_resume(self, args: list[str]) -> None:
        if len(args) != 1 or not args[0]:
            self._show_notice("usage: /resume <id-or-unique-prefix>")
            return
        self.run_worker(
            self._resume_prefix(args[0]),
            name="resume-session",
            group="runtime",
            exit_on_error=False,
        )

    async def _runtime_action(self, method: str) -> None:
        try:
            await getattr(self.runtime, method)()
        except Exception as exc:  # noqa: BLE001 - runtime failures are notices
            self._show_notice(str(exc), level="error")

    def _start_compact(self) -> None:
        self.state = self.state.model_copy(update={"compacting": True})
        self._show_notice("Compacting context...")
        self.run_worker(
            self._compact(),
            name="compact",
            group="runtime",
            exit_on_error=False,
        )

    async def _compact(self) -> None:
        try:
            await self.runtime.compact()
            self._show_notice("context compacted")
        except Exception as exc:  # noqa: BLE001 - runtime failures are notices
            self._show_notice(str(exc), level="error")
        finally:
            self.state = self.state.model_copy(update={"compacting": False})
            self.call_after_refresh(self._refresh_widgets)

    async def _set_permission(self, mode: str) -> None:
        try:
            await self.runtime.set_permission(mode)
        except Exception as exc:  # noqa: BLE001
            self._show_notice(str(exc), level="error")

    async def _resume_prefix(self, prefix: str) -> None:
        try:
            sessions = await self.runtime.list_sessions()
            matches = [summary for summary in sessions if summary.id.startswith(prefix)]
            if not matches:
                self._show_notice(f"session not found: {prefix}")
            elif len(matches) > 1:
                self._show_notice(f"ambiguous session prefix: {prefix}")
            elif self.state.status not in {"idle", "aborted", "error"}:
                self._show_notice("A run is already active")
            else:
                await self.runtime.resume(matches[0].id)
        except Exception as exc:  # noqa: BLE001
            self._show_notice(str(exc), level="error")

    async def _open_session_selector(self) -> None:
        if self.state.status not in {"idle", "aborted", "error"}:
            self._show_notice("A run is already active")
            return
        try:
            sessions = list(await self.runtime.list_sessions())
            sessions.sort(key=lambda summary: summary.updated_at, reverse=True)
            if not sessions:
                self._show_notice("no sessions found")
                return
            self.push_screen(SessionSelector(sessions), callback=self._session_selected)
        except Exception as exc:  # noqa: BLE001
            self._show_notice(str(exc), level="error")

    def _session_selected(self, session_id: str | None) -> None:
        if session_id:
            self.run_worker(
                self._resume_selected(session_id),
                name="resume-session",
                group="runtime",
                exit_on_error=False,
            )

    async def _resume_selected(self, session_id: str) -> None:
        try:
            if self.state.status not in {"idle", "aborted", "error"}:
                self._show_notice("A run is already active")
                return
            await self.runtime.resume(session_id)
        except Exception as exc:  # noqa: BLE001
            self._show_notice(str(exc), level="error")

    def _context_notice(self) -> str:
        window = self.state.context_window
        remaining = max(0, window - self.state.context_used) if window else "?"
        label = "estimated" if self.state.context_estimated else "configured"
        return (
            f"context used {self.state.context_used}, remaining {remaining}, "
            f"window {window or '?'} ({label})"
        )

    async def action_interrupt(self) -> None:
        if self._submit_scheduled:
            self._request_shutdown()
            return
        if self.state.pending_approval is not None:
            run_id = self.state.pending_approval.run_id or self.state.active_run_id
            self._dismiss_approval_screen()
            if run_id:
                self.run_worker(
                    self._abort_run(run_id),
                    name="abort-run",
                    group="runtime",
                    exit_on_error=False,
                )
            return
        run_id = self.state.active_run_id or self._submitted_run_id
        if (
            self.state.status in {"running", "waiting_approval"} or run_id is not None
        ) and run_id:
            self.run_worker(
                self._abort_run(run_id),
                name="abort-run",
                group="runtime",
                exit_on_error=False,
            )
        elif self.state.status == "idle":
            composer = self.query_one("#composer-input", SubmitTextArea)
            if composer.text:
                composer.text = ""
                self.state = self.state.model_copy(update={"input_text": ""})
            else:
                self._request_shutdown()

    async def _abort_run(self, run_id: str) -> None:
        task = self._abort_tasks.get(run_id)
        if task is None:
            task = asyncio.create_task(self._request_runtime_abort(run_id))
            self._abort_tasks[run_id] = task
        await task

    async def _request_runtime_abort(self, run_id: str) -> None:
        try:
            await self.runtime.abort(run_id)
        except Exception as exc:  # noqa: BLE001 - abort failures become notices
            self._show_notice(str(exc), level="error")

    def _apply_event(self, event: RuntimeEvent) -> None:
        # A terminal event from an older run must not settle the currently
        # active run or clear its approval state.
        if (
            event.type in {"run_finished", "run_error"}
            and event.run_id is not None
            and self.state.active_run_id is not None
            and event.run_id != self.state.active_run_id
        ):
            return
        if (
            event.type in {"run_finished", "run_error"}
            and event.run_id is not None
            and self.state.pending_approval is not None
            and self.state.pending_approval.run_id is not None
            and event.run_id != self.state.pending_approval.run_id
        ):
            return
        previous = self.state
        self.state = reduce(self.state, event)
        lifecycle_event_applied = (
            event.type not in {"run_finished", "run_error"} or self.state != previous
        )
        if (
            previous.pending_approval is not None
            and self.state.pending_approval is None
        ) or (
            lifecycle_event_applied
            and previous.pending_approval is not None
            and event.type in {"run_finished", "run_error"}
        ):
            self._dismiss_approval_screen()
            self._approval_request_id = None
        if (
            event.type == "approval_requested"
            and self.state.pending_approval is not None
        ):
            request = self.state.pending_approval
            if (
                previous.pending_approval is None
                or previous.pending_approval.request_id != request.request_id
            ):
                self._approval_request_id = request.request_id
                self.push_screen(
                    ApprovalScreen(request),
                    callback=self._approval_decision,
                )
        if event.type in {"run_finished", "run_error"} and not self._shutdown_requested:
            self._submitted_run_id = None
        self.call_after_refresh(self._refresh_widgets)

    def _dismiss_approval_screen(self) -> None:
        if self._approval_request_id is None:
            return
        screen = self.screen
        if isinstance(screen, ApprovalScreen):
            screen.dismiss(None)

    def _approval_decision(self, decision: str | None) -> None:
        request_id = self._approval_request_id
        self._approval_request_id = None
        if request_id is not None and decision in {"approve", "deny"}:
            self.run_worker(
                self._resolve_approval(request_id, decision),
                name="resolve-approval",
                group="runtime",
                exit_on_error=False,
            )

    async def _resolve_approval(self, request_id: str, decision: str) -> None:
        try:
            await self.runtime.resolve_approval(request_id, decision)
        except Exception as exc:  # noqa: BLE001 - approval failures become notices
            self._show_notice(str(exc), level="error")

    async def _refresh_widgets(self) -> None:
        if not self.is_mounted:
            return
        transcript = self.query_one("#transcript", TranscriptView)
        at_end = transcript.is_vertical_scroll_end
        await transcript.render_state(self.state.transcript)
        if at_end:
            transcript.scroll_end(animate=False)
        self.query_one("#statusline", StatusLine).render_state(
            self.state, getattr(self.runtime, "status", None)
        )

    def on_resize(self, event: events.Resize) -> None:
        if self.is_mounted:
            self.query_one("#statusline", StatusLine).render_state(
                self.state, getattr(self.runtime, "status", None)
            )

    def _show_notice(self, message: str, *, level: str = "notice") -> None:
        self._apply_event(
            RuntimeEvent(type="notice", payload={"message": message, "level": level})
        )

    def _show_error(self, message: str) -> None:
        self._apply_event(RuntimeEvent(type="run_error", payload={"message": message}))


def _state_from_runtime(runtime: Any) -> TuiState:
    workspace = getattr(runtime, "workspace", None)
    model = getattr(runtime, "model", None)
    store = getattr(runtime, "store", None)
    header = getattr(store, "header", None)
    workspace = workspace or getattr(header, "workspace", ".")
    model = model or getattr(header, "model", "unknown")
    session_id = getattr(runtime, "session_id", None)
    context_window = getattr(header, "context_window", 0)
    runtime_status = getattr(runtime, "status", None)
    status = getattr(runtime_status, "status", "idle")
    context_used = getattr(runtime_status, "context_used", 0)
    status_window = getattr(runtime_status, "context_window", None)
    context_estimated = getattr(runtime_status, "context_estimated", False)
    policy = getattr(runtime, "permission_mode", "default")
    if policy not in {"default", "workspace", "full"}:
        policy = "default"
    if status not in {"idle", "running", "waiting_approval", "error", "aborted"}:
        status = "idle"
    return initial_state(
        workspace=str(workspace),
        model=str(model),
        session_id=session_id if isinstance(session_id, str) else None,
        context_window=(
            context_window
            if isinstance(context_window, int) and context_window > 0
            else status_window
            if isinstance(status_window, int) and status_window > 0
            else 0
        ),
        policy=policy,
    ).model_copy(
        update={
            "status": status,
            "context_used": context_used
            if isinstance(context_used, int) and context_used >= 0
            else 0,
            "context_estimated": bool(context_estimated),
        }
    )
