from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.skills.discovery import discover_skills
from coding_agent.tui.commands import SUPPORTED_COMMANDS, parse_command
from coding_agent.tui.reducer import _command_text, reduce
from coding_agent.tui.state import TuiState, initial_state
from coding_agent.tui.widgets import (
    ApprovalScreen,
    CommandComposer,
    HelpScreen,
    HistoryScreen,
    PermissionFullScreen,
    PermissionModeScreen,
    RewindPicker,
    SessionSelector,
    SkillsScreen,
    StatusLine,
    SubmitTextArea,
    TranscriptRow,
    TranscriptView,
    _relative_time,
    detect_git_branch,
)

_COMPACTING_CONFLICTS = frozenset(
    {"new", "session", "resume", "compact", "permission", "clear"}
)

_PROMPT_HISTORY_MAX = 50
"""Maximum number of submitted prompts recalled in the composer."""

_INBOX_MAX_ROWS = 20
"""Maximum number of records shown in the call-history inbox."""


class _PromptHistory(list[str]):
    """A prompt list capped at ``_PROMPT_HISTORY_MAX``, dropping oldest first."""

    def __init__(self, maxlen: int = _PROMPT_HISTORY_MAX) -> None:
        super().__init__()
        self.maxlen = maxlen

    def append(self, item: str) -> None:
        super().append(item)
        if len(self) > self.maxlen:
            del self[: len(self) - self.maxlen]

    def extend(self, items: Iterable[str]) -> None:
        for item in items:
            self.append(item)


def _inbox_compact_args(arguments: object, tool_name: str | None = None) -> str:
    """One-line, length-capped summary of a tool call's arguments.

    Mirrors the reducer's tool-row label: ``run_command`` shows its ``command``
    and other tools render ``key=value`` pairs, always excluding the oversized
    payload keys (``content``/``old_text``/``new_text``) and capping the label
    at ~160 chars, so a huge ``write_file`` body never leaks into an inbox row.
    """
    if not isinstance(arguments, Mapping) or not arguments:
        return ""
    return _command_text(dict(arguments), tool_name) or ""


def _inbox_result_status(result: object) -> str:
    """Status label for a persisted tool result: success/error/cancelled."""
    if getattr(result, "ok", None):
        return "success"
    error = getattr(result, "error", None)
    if isinstance(error, str) and "cancelled" in error.lower():
        return "cancelled"
    return "error"


def _format_inbox_record(record: object) -> str | None:
    """One-line inbox summary for a tool/approval session record.

    Returns ``None`` for record types the inbox does not surface. Rows carry a
    compact ``MM-DD HH:MM`` timestamp prefix and read as:
    ``tool run_command ls -la``, ``result write_file (success)``, and
    ``approve write_file``.
    """
    stamp = getattr(record, "timestamp", None)
    prefix = ""
    if isinstance(stamp, datetime):
        prefix = stamp.astimezone().strftime("%m-%d %H:%M") + "  "
    payload = getattr(record, "payload", None)
    if not isinstance(payload, dict):
        return None
    record_type = getattr(record, "type", None)
    if record_type == "tool_call":
        call = payload.get("tool_call")
        name = getattr(call, "name", "") or ""
        args = _inbox_compact_args(getattr(call, "arguments", None), name)
        summary = f"tool {name}"
        if args:
            summary += f" {args}"
        return f"{prefix}{summary}"
    if record_type == "tool_result":
        result = payload.get("result")
        name = getattr(result, "tool_name", "") or ""
        status = _inbox_result_status(result)
        return f"{prefix}result {name} ({status})"
    if record_type == "approval":
        decision = str(payload.get("decision", "") or "")
        tool = str(payload.get("tool_name", "") or "")
        summary = f"{decision} {tool}".strip()
        return f"{prefix}{summary}" if summary else None
    return None


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
        if event.type in {"assistant_delta", "tool_output_delta"}:
            stream_id = event.payload.get("message_id") or event.payload.get(
                "tool_call_id"
            )
            if isinstance(stream_id, str) and stream_id:
                key = (self._generation, stream_id)
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
                if event.type not in {
                    "assistant_delta",
                    "tool_output_delta",
                    "run_started",
                }:
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
    #transcript .row { margin: 0 0 1 0; width: 1fr; }
    #transcript .row.row-user { width: 1fr; background: #2b2b2e; padding: 0 1; margin: 1 0 1 0; }
    #transcript .row.row-local_command { color: $text-muted; }
    #composer { height: 4; width: 1fr; }
    #composer-input { height: 4; width: 1fr; }
    #composer > #command-palette { width: 1fr; display: none; height: auto; max-height: 8; overlay: screen; constrain: none inside; border: tall $border-blurred; background: $surface; }
    #statusline { height: 1; width: 1fr; }
    ApprovalScreen, HelpScreen, HistoryScreen, SessionSelector, PermissionModeScreen, RewindPicker, SkillsScreen { align: center middle; }
    #approval, #permission-full, #permission-mode, #session-selector, #help, #history, #rewind-picker, #skills { width: 76; height: auto; padding: 1 2; border: round $accent; background: $surface; }
    #help, #history, #skills { max-height: 80%; overflow-y: auto; }
    #rewind-picker-options { height: auto; max-height: 18; }
    #rewind-picker-title { margin-bottom: 1; text-style: bold; }
    #approval-details, #permission-full-details { height: auto; margin-bottom: 1; }
    #approval-diff { border: round $accent; max-height: 12; overflow-y: auto; margin-bottom: 1; }
    #approval-remember-label { margin-bottom: 1; }
    #approval-remember, #approval-feedback { margin-bottom: 1; }
    #approve, #deny, #permission-full-approve, #permission-full-cancel { width: 1fr; margin: 0 1; }
    #session-options, #permission-mode-options { height: auto; max-height: 18; }
    #session-selector-title, #permission-mode-title { margin-bottom: 1; text-style: bold; }
    #session-toggle { margin: 1 0 0 0; width: auto; }
    #session-search { margin: 0 0 1 0; }
    #session-hint { margin-top: 1; color: $text-muted; }
    """
    BINDINGS: ClassVar = [
        Binding("ctrl+c", "interrupt", "Abort", priority=True),
        Binding("?", "open_help", "Help"),
    ]

    def __init__(
        self,
        *,
        runtime: Any,
        initial_state: TuiState | None = None,
        queue_size: int = 256,
        branch_detector: Callable[[str], str | None] | None = None,
        skills_user_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.state = initial_state or _state_from_runtime(runtime)
        self._queue_size = queue_size
        self._branch_detector = branch_detector or detect_git_branch
        self._skills_user_root = skills_user_root
        self._git_branch_detections = 0
        self._follow_bottom = False
        self._last_tail_id: str | None = None
        self._scroll_settle_scheduled = False
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
        self._exit_armed: bool = False
        self._refresh_in_progress = False
        self._refresh_pending = False
        self._spinner_interval: Any | None = None
        self.prompt_history: list[str] = _PromptHistory()
        self._history_index: int | None = None
        self._history_draft = ""

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
        self._sync_spinner_timer()
        self._schedule_git_branch_detection(self.state.workspace)
        self.query_one("#composer-input", SubmitTextArea).focus()

    def _schedule_git_branch_detection(self, workspace: str) -> None:
        self.run_worker(
            self._detect_git_branch(workspace),
            name="detect-git-branch",
            group="runtime",
            exit_on_error=False,
        )

    async def _detect_git_branch(self, workspace: str) -> None:
        try:
            branch = await asyncio.to_thread(self._branch_detector, workspace)
            if self.state.workspace == workspace and self.state.git_branch != branch:
                self.state = self.state.model_copy(update={"git_branch": branch})
                self.call_after_refresh(self._refresh_widgets)
        finally:
            self._git_branch_detections += 1

    async def on_unmount(self, event: events.Unmount) -> None:
        del event
        if self._spinner_interval is not None:
            self._spinner_interval.stop()
            self._spinner_interval = None
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
            self._record_history(prompt)
        except Exception as exc:  # noqa: BLE001 - runtime errors become rows
            self._show_notice(str(exc), level="error")
        finally:
            self._submit_scheduled = False
            self._submit_settled.set()

    def _record_history(self, prompt: str) -> None:
        """Append a submitted prompt to the history ring and reset navigation."""
        self.prompt_history.append(prompt)
        self._history_index = None
        self._history_draft = ""

    def on_submit_text_area_composer_history_requested(
        self, event: SubmitTextArea.ComposerHistoryRequested
    ) -> None:
        event.stop()
        self._history_recall(event.offset)

    def _history_recall(self, offset: int) -> None:
        """Recall composer prompt history: ``-1`` older, ``+1`` newer.

        The first ``-1`` stashes the current draft text; navigating ``+1`` past
        the newest entry restores it and resets ``_history_index``. No-op when
        history is empty, so the arrows are inert until something was submitted.
        """
        if not self.prompt_history:
            return
        composer = self.query_one("#composer-input", SubmitTextArea)
        if offset < 0:
            if self._history_index is None:
                self._history_draft = composer.text
                self._history_index = len(self.prompt_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            composer.text = self.prompt_history[self._history_index]
            return
        if self._history_index is None:
            return
        if self._history_index >= len(self.prompt_history) - 1:
            self._history_index = None
            composer.text = self._history_draft
        else:
            self._history_index += 1
            composer.text = self.prompt_history[self._history_index]

    def on_submit_text_area_rewind_requested(self, event) -> None:
        event.stop()
        if self.state.status != "idle":
            self._show_notice("A run is active; cannot rewind")
            return
        rows = self._rewind_rows()
        if not rows:
            self._show_notice("No earlier user messages to rewind to")
            return
        self.push_screen(RewindPicker(rows), callback=self._rewind_selected)

    def _rewind_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for item in self.state.transcript:
            if item.kind == "user":
                preview = item.text[:60] or "(empty prompt)"
                rows.append((item.item_id, preview, _relative_time(item.timestamp)))
        return rows

    async def _rewind_selected(self, message_id: str | None) -> None:
        if message_id:
            self.run_worker(
                self._fork_from_message(message_id),
                name="rewind-fork",
                group="runtime",
                exit_on_error=False,
            )

    async def _fork_from_message(self, message_id: str) -> None:
        try:
            prompt = await self.runtime.fork_at(message_id)
            composer = self.query_one("#composer-input", SubmitTextArea)
            composer.text = prompt
        except Exception as exc:  # noqa: BLE001 - user-facing fork errors
            self._show_notice(f"rewind failed: {exc}", level="error")

    def action_open_help(self) -> None:
        """Push the help overlay (bound to ``?`` and the ``/help`` command)."""
        if isinstance(self.screen, HelpScreen):
            return
        self.push_screen(HelpScreen(skills=self._skills_catalog()))

    def _skills_catalog(self):
        """Discover the installed skills for the current session workspace."""
        return discover_skills(
            Path(self.state.workspace), user_root=self._skills_user_root
        )

    def _inbox_rows(self) -> list[str]:
        """Recent tool/approval summaries from ``store.records()``.

        Filters to ``tool_call`` (name + compact args), ``tool_result``
        (status), and ``approval`` (decision + tool) records, sorts them newest
        first, and caps the list at ``_INBOX_MAX_ROWS``.
        """
        store = getattr(self.runtime, "store", None)
        records = getattr(store, "records", None)
        if records is None:
            return []
        rows: list[tuple[datetime, str]] = []
        for record in records():
            row = _format_inbox_record(record)
            if row is not None:
                rows.append((record.timestamp, row))
        rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in rows[:_INBOX_MAX_ROWS]]

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
            self.action_open_help()
        elif name == "inbox":
            if args:
                self._show_notice("usage: /inbox")
                return
            self.push_screen(HistoryScreen(self._inbox_rows()))
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
        elif name == "undo":
            if args:
                self._show_notice("usage: /undo")
                return
            self.run_worker(
                self._runtime_action("undo"),
                name="undo",
                group="runtime",
                exit_on_error=False,
            )
        elif name == "skills":
            if args:
                self._show_notice("usage: /skills")
                return
            self.push_screen(SkillsScreen(self._skills_catalog()))

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
                HelpScreen,
                HistoryScreen,
                PermissionFullScreen,
                PermissionModeScreen,
                SessionSelector,
                SkillsScreen,
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
        if not args:
            # Bare /resume opens the same discoverable session picker as
            # /session so a session can be chosen interactively.
            self.run_worker(
                self._open_session_selector(),
                name="list-sessions",
                group="runtime",
                exit_on_error=False,
            )
            return
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
            return
        self._show_notice(f"permission mode changed to {mode}")

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
            self.push_screen(
                SessionSelector(sessions, workspace=self.state.workspace),
                callback=self._session_selected,
            )
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
        label = "estimated" if self.state.context_estimated else "configured"
        return (
            f"context used {self.state.context_used}, window {window or '?'} ({label})"
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
            if self._exit_armed:
                self._request_shutdown()
            else:
                self._exit_armed = True
                self._show_notice("Press ctrl+c again to exit", level="notice")

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
        if event.type == "run_started":
            self._exit_armed = False
        previous = self.state
        self.state = reduce(self.state, event)
        if (
            event.type == "session_loaded"
            and self.state.workspace != previous.workspace
        ):
            self._schedule_git_branch_detection(self.state.workspace)
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
                    ApprovalScreen(request, workspace=self.state.workspace),
                    callback=self._approval_decision,
                )
        if event.type in {"run_finished", "run_error"} and not self._shutdown_requested:
            self._submitted_run_id = None
        self._sync_spinner_timer()
        self.call_after_refresh(self._refresh_widgets)

    def _sync_spinner_timer(self) -> None:
        """Keep the statusline animation timer in step with the run status.

        Starts the interval when a run becomes active (running or waiting on
        approval) and stops it as soon as the run leaves those states. Idempotent
        so it is safe to call from any ``_apply_event``.
        """
        active = self.state.status in {"running", "waiting_approval"}
        if active and self._spinner_interval is None and self.is_mounted:
            self._spinner_interval = self.set_interval(
                0.2, self._tick_spinner, name="statusline-spinner"
            )
        elif not active and self._spinner_interval is not None:
            self._spinner_interval.stop()
            self._spinner_interval = None

    def _tick_spinner(self) -> None:
        """Advance the statusline spinner frame and refresh elapsed time."""
        self.state = self.state.model_copy(
            update={"spinner_frame": self.state.spinner_frame + 1}
        )
        if self.is_mounted:
            self.query_one("#statusline", StatusLine).render_state(
                self.state, getattr(self.runtime, "status", None)
            )
            self.query_one("#transcript", TranscriptView).update_pending(
                self.state.spinner_frame, time.monotonic()
            )

    def _dismiss_approval_screen(self) -> None:
        if self._approval_request_id is None:
            return
        screen = self.screen
        if isinstance(screen, ApprovalScreen):
            screen.dismiss(None)

    def _approval_decision(self, result) -> None:
        """Resolve a pending approval with the screen's remember/feedback.

        ``result`` is ``(decision, remember, feedback)`` from ``ApprovalScreen``,
        or ``None`` when the modal was dismissed (interrupt/shutdown).
        """
        request_id = self._approval_request_id
        self._approval_request_id = None
        if (
            request_id is None
            or not isinstance(result, tuple)
            or len(result) != 3
            or result[0] not in {"approve", "deny"}
        ):
            return
        decision, remember, feedback = result
        self.run_worker(
            self._resolve_approval(request_id, decision, remember, feedback),
            name="resolve-approval",
            group="runtime",
            exit_on_error=False,
        )

    async def _resolve_approval(
        self,
        request_id: str,
        decision: str,
        remember: str = "once",
        feedback: str | None = None,
    ) -> None:
        try:
            await self.runtime.resolve_approval(
                request_id, decision, remember=remember, feedback=feedback
            )
        except Exception as exc:  # noqa: BLE001 - approval failures become notices
            self._show_notice(str(exc), level="error")

    async def _refresh_widgets(self) -> None:
        # Serialize renders: concurrent calls used to interleave
        # remove_children()/mount_all() on the transcript and raise
        # DuplicateIds. If a render is already running, mark a re-run and let
        # the running render schedule it once it finishes.
        if self._refresh_in_progress:
            self._refresh_pending = True
            return
        self._refresh_in_progress = True
        self._refresh_pending = False
        try:
            if not self.is_mounted:
                return
            transcript = self.query_one("#transcript", TranscriptView)
            at_end = transcript.is_vertical_scroll_end
            tail = self.state.transcript[-1] if self.state.transcript else None
            tail_changed = tail is not None and tail.item_id != self._last_tail_id
            # Keep following the newest row not only when the view is already at
            # the bottom, but also when a row was appended while we were at the
            # bottom on the previous refresh. Local commands append a row and may
            # open an overlay that shrinks the viewport before the scheduled
            # refresh runs, so the pre-render at_end check alone would leave the
            # fresh row just under the fold.
            follow = at_end or (self._follow_bottom and tail_changed)
            await transcript.render_state(
                self.state.transcript,
                spinner_frame=self.state.spinner_frame,
                now=time.monotonic(),
            )
            if follow:
                transcript.scroll_end(animate=False)
                # scroll_end right after mounting can run before the widget has
                # re-laid out, so it uses a stale max_scroll_y and a freshly
                # appended row (e.g. a local command) is left just under the
                # fold. Re-run once the layout settles to land on the true end.
                self._schedule_scroll_settle()
            self._follow_bottom = at_end
            self._last_tail_id = tail.item_id if tail is not None else None
            self.query_one("#statusline", StatusLine).render_state(
                self.state, getattr(self.runtime, "status", None)
            )
        finally:
            self._refresh_in_progress = False
            if self._refresh_pending:
                self.call_after_refresh(self._refresh_widgets)

    def _schedule_scroll_settle(self) -> None:
        """Re-scroll to the transcript end once a pending layout settles.

        Coalesced so a burst of deltas schedules at most one follow-up.
        """
        if self._scroll_settle_scheduled:
            return
        self._scroll_settle_scheduled = True
        self.set_timer(0.05, self._settle_scroll_end)

    def _settle_scroll_end(self) -> None:
        self._scroll_settle_scheduled = False
        if not self.is_mounted:
            return
        transcript = self.query_one("#transcript", TranscriptView)
        if not transcript.is_vertical_scroll_end:
            transcript.scroll_end(animate=False)

    def on_resize(self, event: events.Resize) -> None:
        if self.is_mounted:
            self.query_one("#statusline", StatusLine).render_state(
                self.state, getattr(self.runtime, "status", None)
            )

    def on_transcript_row_tool_row_clicked(
        self, event: TranscriptRow.ToolRowClicked
    ) -> None:
        """Expand/collapse a clicked tool row and re-render from state."""
        event.stop()
        transcript = [
            row.model_copy(update={"expanded": not row.expanded})
            if row.kind == "tool" and row.item_id == event.item_id
            else row
            for row in self.state.transcript
        ]
        self.state = self.state.model_copy(update={"transcript": transcript})
        self.call_after_refresh(self._refresh_widgets)

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
    policy = getattr(runtime, "permission_mode", "workspace")
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
