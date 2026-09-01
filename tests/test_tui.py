from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from textual.containers import VerticalScroll
from textual.widgets import OptionList, Static, TextArea

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import Message
from coding_agent.session.models import ApprovalRequest, SessionMessage, SessionSummary
from coding_agent.tui.app import (
    ApprovalScreen,
    CodingAgentApp,
    PermissionFullScreen,
    _RuntimeBridge,
)
from coding_agent.tui.state import TranscriptItem, TuiState, initial_state
from coding_agent.tui.widgets import CommandComposer, TranscriptRow, format_statusline


class FakeRuntime:
    def __init__(self) -> None:
        self.subscribers: list[Callable[[RuntimeEvent], Awaitable[None]]] = []
        self.submitted: list[str] = []
        self.aborted: list[str] = []
        self.resolved: list[tuple[str, str]] = []
        self.permissions: list[str] = []
        self.new_sessions = 0
        self.compactions = 0
        self.resumes: list[str] = []
        now = datetime.now(UTC)
        self.sessions = [
            SessionSummary(
                id="s2-newest-long-id",
                workspace="/tmp/project",
                created_at=now - timedelta(hours=2),
                updated_at=now,
                title="Newest session",
                last_status="idle",
            ),
            SessionSummary(
                id="s1-older-long-id",
                workspace="/tmp/project",
                created_at=now - timedelta(hours=3),
                updated_at=now - timedelta(hours=1),
                title="Older session",
                last_status="idle",
            ),
        ]
        self.status = type("RuntimeStatus", (), {"usage": None})()

    def subscribe(
        self, sink: Callable[[RuntimeEvent], Awaitable[None]]
    ) -> Callable[[], None]:
        self.subscribers.append(sink)

        def unsubscribe() -> None:
            if sink in self.subscribers:
                self.subscribers.remove(sink)

        return unsubscribe

    async def emit(self, event: RuntimeEvent) -> None:
        for sink in list(self.subscribers):
            await sink(event)

    async def submit(self, prompt: str) -> str:
        self.submitted.append(prompt)
        return "run-1"

    async def abort(self, run_id: str) -> None:
        self.aborted.append(run_id)

    async def resolve_approval(
        self,
        request_id: str,
        decision: str,
        remember: str = "once",
        feedback: str | None = None,
    ) -> None:
        del remember, feedback
        self.resolved.append((request_id, decision))

    async def set_permission(self, mode: str) -> None:
        self.permissions.append(mode)
        await self.emit(RuntimeEvent(type="policy_changed", payload={"policy": mode}))

    async def new_session(self) -> str:
        self.new_sessions += 1
        await self.emit(
            RuntimeEvent(type="session_loaded", payload={"session_id": "new-id"})
        )
        return "new-id"

    async def list_sessions(self) -> list[SessionSummary]:
        return self.sessions

    async def resume(self, session_id: str) -> None:
        self.resumes.append(session_id)
        await self.emit(
            RuntimeEvent(type="session_loaded", payload={"session_id": session_id})
        )

    async def compact(self) -> None:
        self.compactions += 1


class FailingCommandRuntime(FakeRuntime):
    async def new_session(self) -> str:
        raise RuntimeError("cannot create session")


def make_state(**updates: object) -> TuiState:
    state = initial_state("/tmp/project", "fake-model", context_window=1000)
    return state.model_copy(update=updates)


@pytest.mark.asyncio
async def test_shell_has_exact_three_regions_in_order() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        children = list(pilot.app.screen.children)
        assert [child.id for child in children] == [
            "transcript",
            "composer",
            "statusline",
        ]
        assert isinstance(pilot.app.query_one("#transcript"), VerticalScroll)
        assert isinstance(pilot.app.query_one("#composer"), CommandComposer)
        assert isinstance(pilot.app.query_one("#composer-input"), TextArea)
        assert isinstance(pilot.app.query_one("#statusline"), Static)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["aborted", "error"])
async def test_terminal_state_allows_follow_up_prompt(status: str) -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state(status=status))

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "follow up"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.submitted == ["follow up"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["aborted", "error"])
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/new", "new"),
        ("/resume s2-newest", "resume"),
        ("/compact", "compact"),
    ],
)
async def test_terminal_state_allows_recovery_command(
    status: str, command: str, expected: str
) -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state(status=status))

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = command
        await pilot.press("enter")
        await pilot.pause()

    if expected == "new":
        assert runtime.new_sessions == 1
    elif expected == "resume":
        assert runtime.resumes == ["s2-newest-long-id"]
    else:
        assert runtime.compactions == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["aborted", "error"])
async def test_terminal_state_allows_opening_session_selector(status: str) -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state(status=status))

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/session"
        await pilot.press("enter")
        await pilot.pause()

        assert pilot.app.screen.query_one("OptionList", OptionList)


@pytest.mark.asyncio
async def test_aborted_run_allows_a_follow_up_prompt() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(
                type="run_finished",
                run_id="run-aborted",
                payload={"outcome": {"reason": "aborted"}},
            )
        )
        await pilot.pause()
        assert app.state.status == "aborted"

        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "follow up"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.submitted == ["follow up"]


@pytest.mark.asyncio
async def test_bridge_flushes_all_coalesced_deltas_before_a_control_event() -> None:
    class BridgeTarget:
        def __init__(self) -> None:
            self.events: list[RuntimeEvent] = []

        def _apply_event(self, event: RuntimeEvent) -> None:
            self.events.append(event)

        def _show_error(self, message: str) -> None:
            raise AssertionError(message)

    target = BridgeTarget()
    bridge = _RuntimeBridge(target, maxsize=1)
    await bridge.publish(RuntimeEvent(type="run_started", run_id="run-1"))
    await bridge.publish(
        RuntimeEvent(
            type="assistant_delta",
            run_id="run-1",
            payload={"message_id": "m-a", "text": "A"},
        )
    )
    await bridge.publish(
        RuntimeEvent(
            type="assistant_delta",
            run_id="run-1",
            payload={"message_id": "m-b", "text": "B"},
        )
    )
    control = asyncio.create_task(
        bridge.publish(RuntimeEvent(type="run_finished", run_id="run-1"))
    )
    bridge.start()
    await asyncio.wait_for(control, timeout=1)
    await asyncio.sleep(0.15)
    await bridge.stop()

    assert [event.type for event in target.events] == [
        "run_started",
        "assistant_delta",
        "assistant_delta",
        "run_finished",
    ]
    assert [
        event.payload["text"]
        for event in target.events
        if event.type == "assistant_delta"
    ] == ["A", "B"]


@pytest.mark.asyncio
async def test_rapid_second_enter_is_rejected_before_blocking_submit_returns() -> None:
    class BlockingSubmitRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.submit_started = asyncio.Event()
            self.release_submit = asyncio.Event()

        async def submit(self, prompt: str) -> str:
            self.submitted.append(prompt)
            self.submit_started.set()
            await self.release_submit.wait()
            return "run-1"

    runtime = BlockingSubmitRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "first prompt"
        await pilot.press("enter")
        await asyncio.wait_for(runtime.submit_started.wait(), timeout=1)
        composer.text = "second prompt"
        await pilot.press("enter")
        await pilot.pause()

        assert runtime.submitted == ["first prompt"]
        assert any("active" in row.text.lower() for row in app.state.transcript)
        runtime.release_submit.set()


@pytest.mark.asyncio
async def test_external_app_shutdown_waits_for_pending_submit_before_abort() -> None:
    class BlockingSubmitThenRunRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.submit_started = asyncio.Event()
            self.release_submit = asyncio.Event()

        async def submit(self, prompt: str) -> str:
            self.submitted.append(prompt)
            self.submit_started.set()
            await self.release_submit.wait()
            return "run-shutdown"

    runtime = BlockingSubmitThenRunRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "pending prompt"
        await pilot.press("enter")
        await asyncio.wait_for(runtime.submit_started.wait(), timeout=1)
        runtime.release_submit.set()
        await pilot.pause()
        app.state = app.state.model_copy(
            update={"status": "running", "active_run_id": "run-shutdown"}
        )

    assert runtime.aborted == ["run-shutdown"]


@pytest.mark.asyncio
async def test_enter_submits_prompt_only_while_idle() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "  inspect the project  "
        composer.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert runtime.submitted == ["inspect the project"]
        assert composer.text == ""


@pytest.mark.asyncio
async def test_quit_aborts_active_run_and_waits_before_ui_teardown() -> None:
    class BlockingAbortRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.abort_started = asyncio.Event()
            self.release_abort = asyncio.Event()
            self.abort_settled = asyncio.Event()

        async def abort(self, run_id: str) -> None:
            self.aborted.append(run_id)
            self.abort_started.set()
            await self.release_abort.wait()
            self.abort_settled.set()

    runtime = BlockingAbortRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=make_state(status="running", active_run_id="run-quit"),
    )

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/quit"
        await pilot.press("enter")
        await asyncio.wait_for(runtime.abort_started.wait(), timeout=1)
        await pilot.pause()

        assert app.is_running
        assert not runtime.abort_settled.is_set()
        runtime.release_abort.set()
        await pilot.pause()
        await asyncio.wait_for(runtime.abort_settled.wait(), timeout=1)

    assert runtime.aborted == ["run-quit"]


@pytest.mark.asyncio
async def test_idle_quit_exits_without_calling_abort() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/quit"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.aborted == []


@pytest.mark.asyncio
async def test_external_run_finished_dismisses_approval_modal_without_stale_callback() -> (
    None
):
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())
    request = ApprovalRequest(
        request_id="approval-finished",
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="shell",
        risk_level="execute_command",
    )

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(
                type="approval_requested", run_id="run-1", payload={"request": request}
            )
        )
        await pilot.pause()
        await runtime.emit(
            RuntimeEvent(
                type="run_finished",
                run_id="run-1",
                payload={"outcome": {"reason": "aborted"}},
            )
        )
        await pilot.pause()

        assert not isinstance(pilot.app.screen, ApprovalScreen)
        assert app.state.pending_approval is None
        assert app._approval_request_id is None

    assert runtime.resolved == []


@pytest.mark.asyncio
async def test_ctrl_c_aborts_active_run() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=make_state(status="running", active_run_id="run-9"),
    )

    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert runtime.aborted == ["run-9"]


@pytest.mark.asyncio
async def test_aborted_follow_up_after_terminal_event() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(
                type="run_finished",
                run_id="run-aborted",
                payload={"outcome": {"reason": "aborted"}},
            )
        )
        await pilot.pause()
        assert app.state.status == "aborted"

        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "follow up"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.submitted == ["follow up"]


@pytest.mark.asyncio
async def test_unknown_command_is_reported_as_error() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/unknown"
        await pilot.press("enter")
        await pilot.pause()

    assert app.state.transcript[-1].level == "error"
    assert "unknown command" in app.state.transcript[-1].text


@pytest.mark.asyncio
async def test_stale_run_finished_does_not_clear_new_run_approval() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=make_state(
            status="waiting_approval",
            active_run_id="run-new",
            active_turn_id="turn-new",
        ),
    )
    request = ApprovalRequest(
        request_id="approval-new",
        run_id="run-new",
        tool_call_id="call-new",
        tool_name="shell",
        risk_level="execute_command",
    )

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(
                type="approval_requested",
                run_id="run-new",
                payload={"request": request},
            )
        )
        await pilot.pause()
        await runtime.emit(
            RuntimeEvent(
                type="run_finished",
                run_id="run-old",
                payload={"outcome": {"reason": "aborted"}},
            )
        )
        await pilot.pause()

        assert app.state.active_run_id == "run-new"
        assert app.state.pending_approval == request
        assert isinstance(pilot.app.screen, ApprovalScreen)


@pytest.mark.asyncio
async def test_runtime_deltas_render_as_one_assistant_row_and_finish_idle() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(type="run_started", run_id="run-1", turn_id="turn-1")
        )
        await runtime.emit(
            RuntimeEvent(
                type="assistant_started", run_id="run-1", payload={"message_id": "m-1"}
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="assistant_delta",
                run_id="run-1",
                payload={"message_id": "m-1", "text": "hello "},
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="assistant_delta",
                run_id="run-1",
                payload={"message_id": "m-1", "text": "world"},
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="run_finished",
                run_id="run-1",
                payload={"outcome": {"reason": "completed"}},
            )
        )
        await pilot.pause()

        assert app.state.status == "idle"
        assert [
            row.text for row in app.state.transcript if row.kind == "assistant"
        ] == ["hello world"]
        rendered = "\n".join(
            str(row.render()) for row in pilot.app.query("#transcript TranscriptRow")
        )
        assert "hello world" in rendered


@pytest.mark.asyncio
async def test_approval_action_calls_only_runtime_resolution() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())
    request = ApprovalRequest(
        request_id="approval-1",
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="write_file",
        risk_level="mutate_file",
        arguments={"path": "main.py"},
        reason="write requested",
    )

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(
                type="approval_requested",
                run_id="run-1",
                payload={"request": request},
            )
        )
        await pilot.pause()
        assert app.state.pending_approval == request
        assert pilot.app.screen.query_one("#approval")

        await pilot.press("a")
        await pilot.pause()

    assert runtime.resolved == [("approval-1", "approve")]


@pytest.mark.asyncio
async def test_permission_full_requires_confirmation_before_runtime_call() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state(policy="workspace"))

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/permission full"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(pilot.app.screen, PermissionFullScreen)
        assert runtime.permissions == []
        await pilot.press("escape")
        await pilot.pause()

    assert runtime.permissions == []
    assert app.state.policy == "workspace"


@pytest.mark.asyncio
async def test_permission_full_confirmation_calls_only_runtime_set_permission() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/permission full"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.permissions == ["full"]


@pytest.mark.asyncio
async def test_bare_permission_reports_current_mode_without_runtime_call() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state(policy="workspace"))

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/permission"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.permissions == []
    assert "workspace" in app.state.transcript[-1].text


@pytest.mark.asyncio
async def test_commands_are_rejected_while_run_is_active() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=make_state(status="running", active_run_id="run-1"),
    )

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/new"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.new_sessions == 0
    assert "active" in app.state.transcript[-1].text.lower()


@pytest.mark.asyncio
async def test_slash_commands_call_runtime_and_never_submit_command_text() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        for command in ("/permission workspace", "/new", "/compact"):
            composer.text = command
            await pilot.press("enter")
            await pilot.pause()

    assert runtime.permissions == ["workspace"]
    assert runtime.new_sessions == 1
    assert runtime.compactions == 1
    assert runtime.submitted == []


@pytest.mark.asyncio
async def test_command_failure_is_local_notice_without_error_status() -> None:
    runtime = FailingCommandRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/new"
        await pilot.press("enter")
        await pilot.pause()

    assert app.state.status == "idle"
    assert "cannot create session" in app.state.transcript[-1].text


@pytest.mark.asyncio
async def test_resume_renders_reducer_hydrated_history_without_store_access() -> None:
    class StoreForbiddenRuntime(FakeRuntime):
        @property
        def store(self) -> object:
            raise AssertionError("TUI must not read runtime.store")

        async def resume(self, session_id: str) -> None:
            self.resumes.append(session_id)
            history = [
                SessionMessage(
                    record_id="history-user",
                    seq=1,
                    message=Message(role="user", content="restored question"),
                ).model_dump(mode="json"),
                SessionMessage(
                    record_id="history-answer",
                    seq=2,
                    message=Message(role="assistant", content="restored answer"),
                ).model_dump(mode="json"),
            ]
            await self.emit(
                RuntimeEvent(
                    type="session_loaded",
                    payload={"session_id": session_id, "history": history},
                )
            )

    runtime = StoreForbiddenRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/resume s2-newest"
        await pilot.press("enter")
        await pilot.pause()

        rendered = pilot.app.query_one("#transcript").renderable_text
        assert [row.text for row in app.state.transcript] == [
            "restored question",
            "restored answer",
            "/resume s2-newest",
        ]
        assert "restored question" in rendered
        assert "restored answer" in rendered


@pytest.mark.asyncio
async def test_new_session_clears_hydrated_history_rows() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=make_state(
            transcript=[TranscriptItem(kind="assistant", item_id="old", text="old row")]
        ),
    )

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/new"
        await pilot.press("enter")
        await pilot.pause()

        assert [(row.kind, row.text) for row in app.state.transcript] == [
            ("local_command", "/new"),
        ]
        assert "/new" in pilot.app.query_one("#transcript").renderable_text


@pytest.mark.asyncio
async def test_session_selector_resumes_only_selected_session() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/session"
        await pilot.press("enter")
        await pilot.pause()
        options = pilot.app.screen.query_one("OptionList", OptionList)
        options.highlighted = 1
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.resumes == ["s1-older-long-id"]


@pytest.mark.asyncio
async def test_bare_resume_opens_session_selector() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/resume"
        await pilot.press("enter")
        await pilot.pause()
        options = pilot.app.screen.query_one("OptionList", OptionList)
        options.highlighted = 1
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.resumes == ["s1-older-long-id"]


@pytest.mark.asyncio
async def test_resume_ambiguous_prefix_is_local_notice() -> None:
    runtime = FakeRuntime()
    runtime.sessions = [
        runtime.sessions[0].model_copy(update={"id": "abcdef-one"}),
        runtime.sessions[1].model_copy(update={"id": "abcdef-two"}),
    ]
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/resume abcdef"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.resumes == []
    assert any("ambiguous" in row.text.lower() for row in app.state.transcript)


@pytest.mark.asyncio
async def test_ctrl_c_denies_pending_approval() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())
    request = ApprovalRequest(
        request_id="approval-cancel",
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="shell",
        risk_level="execute_command",
        arguments={"command": "pwd"},
        reason="requested",
    )

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(type="approval_requested", payload={"request": request})
        )
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert runtime.aborted == ["run-1"]
    assert runtime.resolved == []


@pytest.mark.asyncio
async def test_external_approval_resolution_dismisses_modal() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())
    request = ApprovalRequest(
        request_id="approval-external",
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="write_file",
        risk_level="mutate_file",
    )

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(
                type="approval_requested",
                run_id="run-1",
                payload={"request": request},
            )
        )
        await pilot.pause()
        assert isinstance(pilot.app.screen, ApprovalScreen)

        await runtime.emit(
            RuntimeEvent(
                type="approval_resolved",
                run_id="run-1",
                payload={
                    "request_id": request.request_id,
                    "decision": "approve",
                    "status": "approved",
                },
            )
        )
        await pilot.pause()

        assert app.state.pending_approval is None
        assert not isinstance(pilot.app.screen, ApprovalScreen)


@pytest.mark.asyncio
async def test_bridge_flushes_coalesced_delta_before_control_event() -> None:
    class BridgeTarget:
        def __init__(self) -> None:
            self.events: list[RuntimeEvent] = []

        def _apply_event(self, event: RuntimeEvent) -> None:
            self.events.append(event)

        def _show_error(self, message: str) -> None:
            raise AssertionError(message)

    target = BridgeTarget()
    bridge = _RuntimeBridge(target, maxsize=1)
    await bridge.publish(RuntimeEvent(type="run_started", run_id="run-1"))
    await bridge.publish(
        RuntimeEvent(
            type="assistant_delta",
            run_id="run-1",
            payload={"message_id": "m-1", "text": "hello"},
        )
    )
    control = asyncio.create_task(
        bridge.publish(RuntimeEvent(type="run_finished", run_id="run-1"))
    )
    await asyncio.sleep(0)
    bridge.start()
    await asyncio.wait_for(control, timeout=1)
    await asyncio.sleep(0.15)
    await bridge.stop()

    assert [event.type for event in target.events] == [
        "run_started",
        "assistant_delta",
        "run_finished",
    ]
    assert target.events[1].payload["text"] == "hello"


@pytest.mark.asyncio
async def test_bridge_preserves_fifo_when_later_delta_finds_queue_capacity() -> None:
    class BridgeTarget:
        def __init__(self) -> None:
            self.events: list[RuntimeEvent] = []

        def _apply_event(self, event: RuntimeEvent) -> None:
            self.events.append(event)

        def _show_error(self, message: str) -> None:
            raise AssertionError(message)

    target = BridgeTarget()
    bridge = _RuntimeBridge(target, maxsize=1)
    first = RuntimeEvent(
        type="assistant_delta",
        run_id="run-1",
        payload={"message_id": "m-1", "text": "first"},
    )
    await bridge.publish(RuntimeEvent(type="run_started", run_id="run-1"))
    await bridge.publish(first)
    assert bridge.queue.get_nowait().type == "run_started"
    assert bridge.queue.empty()

    second = RuntimeEvent(
        type="assistant_delta",
        run_id="run-1",
        payload={"message_id": "m-2", "text": "second"},
    )
    await bridge.publish(second)
    assert bridge.queue.get_nowait().payload["text"] == "first"
    assert bridge._coalesced
    assert (
        bridge._coalesced.pop(next(iter(bridge._coalesced))).payload["text"] == "second"
    )


@pytest.mark.asyncio
async def test_bridge_keeps_post_control_delta_after_control_event() -> None:
    class BridgeTarget:
        def __init__(self) -> None:
            self.events: list[RuntimeEvent] = []

        def _apply_event(self, event: RuntimeEvent) -> None:
            self.events.append(event)

        def _show_error(self, message: str) -> None:
            raise AssertionError(message)

    target = BridgeTarget()
    bridge = _RuntimeBridge(target, maxsize=1)
    await bridge.publish(RuntimeEvent(type="run_started", run_id="run-1"))
    control = asyncio.create_task(
        bridge.publish(RuntimeEvent(type="run_finished", run_id="run-1"))
    )
    await asyncio.sleep(0)
    delta = asyncio.create_task(
        bridge.publish(
            RuntimeEvent(
                type="assistant_delta",
                run_id="run-1",
                payload={"message_id": "m-1", "text": "late"},
            )
        )
    )
    await asyncio.sleep(0)
    bridge.start()
    await asyncio.wait_for(delta, timeout=1)
    await asyncio.wait_for(control, timeout=1)
    await asyncio.sleep(0.15)
    await bridge.stop()

    assert [event.type for event in target.events] == [
        "run_started",
        "run_finished",
        "assistant_delta",
    ]


@pytest.mark.asyncio
async def test_bridge_does_not_merge_deltas_across_control_event() -> None:
    class BridgeTarget:
        def __init__(self) -> None:
            self.events: list[RuntimeEvent] = []

        def _apply_event(self, event: RuntimeEvent) -> None:
            self.events.append(event)

        def _show_error(self, message: str) -> None:
            raise AssertionError(message)

    target = BridgeTarget()
    bridge = _RuntimeBridge(target, maxsize=1)
    await bridge.publish(RuntimeEvent(type="run_started", run_id="run-1"))
    await bridge.publish(
        RuntimeEvent(
            type="assistant_delta",
            run_id="run-1",
            payload={"message_id": "m-1", "text": "before"},
        )
    )
    control = asyncio.create_task(
        bridge.publish(RuntimeEvent(type="run_finished", run_id="run-1"))
    )
    await asyncio.sleep(0)
    after = asyncio.create_task(
        bridge.publish(
            RuntimeEvent(
                type="assistant_delta",
                run_id="run-1",
                payload={"message_id": "m-1", "text": "after"},
            )
        )
    )
    bridge.start()
    await asyncio.wait_for(control, timeout=1)
    await asyncio.wait_for(after, timeout=1)
    await asyncio.sleep(0.15)
    await bridge.stop()

    assert [event.type for event in target.events] == [
        "run_started",
        "assistant_delta",
        "run_finished",
        "assistant_delta",
    ]
    assert [
        event.payload["text"]
        for event in target.events
        if event.type == "assistant_delta"
    ] == [
        "before",
        "after",
    ]


def test_statusline_contains_context_remaining_and_usage() -> None:
    from coding_agent.runtime.models import Usage

    state = make_state(context_used=300, context_window=1000, git_branch=None)
    text = format_statusline(state, usage=Usage(input_tokens=120, output_tokens=45))
    assert "ctx 300/700/1000" in text
    assert "in 120" in text
    assert "out 45" in text
    for width in (0, 1, 10, 40):
        assert len(format_statusline(state, width=width)) <= width


def test_transcript_markup_values_render_as_literal_text() -> None:
    item = TranscriptItem(kind="assistant", item_id="m-1", text="[bold]literal[/bold]")
    row = TranscriptRow(item, index=0)

    assert str(row.render()) == "[bold]literal[/bold]"


@pytest.mark.asyncio
async def test_runtime_error_is_visible_in_system_row() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(
                type="run_error",
                run_id="run-1",
                payload={"message": "provider unavailable"},
            )
        )
        await pilot.pause()

        assert app.state.status == "error"
        assert app.state.transcript[-1].kind == "system"
        assert "provider unavailable" in str(
            pilot.app.query_one("#transcript").renderable_text
        )


@pytest.mark.asyncio
async def test_slash_opens_filtered_command_palette_without_adding_layout_region() -> (
    None
):
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/co"
        await pilot.pause()

        palette = pilot.app.query_one("#command-palette", OptionList)
        assert palette.display is True
        assert palette.option_count == 2
        assert {
            palette.get_option_at_index(index).prompt.split()[0]
            for index in range(palette.option_count)
        } == {"/compact", "/context"}
        assert [child.id for child in pilot.app.screen.children] == [
            "transcript",
            "composer",
            "statusline",
        ]


@pytest.mark.asyncio
async def test_command_palette_is_rendered_on_an_80_by_24_screen() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test(size=(80, 24)) as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/"
        await pilot.pause()

        palette = pilot.app.query_one("#command-palette", OptionList)
        screenshot = app.export_screenshot()
        assert palette.display is True
        assert palette.region.width > 0
        assert palette.region.height > 0
        assert "/compact" in screenshot
        assert "/context" in screenshot
        assert [child.id for child in pilot.app.screen.children] == [
            "transcript",
            "composer",
            "statusline",
        ]


@pytest.mark.asyncio
async def test_palette_escape_closes_and_enter_selects_filtered_command() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/comp"
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert pilot.app.query_one("#command-palette", OptionList).display is False

        composer.text = "/comp"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.compactions == 1
    assert runtime.submitted == []


@pytest.mark.asyncio
async def test_palette_up_and_down_navigate_highlighted_command() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/"
        await pilot.pause()
        palette = pilot.app.query_one("#command-palette", OptionList)
        assert palette.highlighted == 0
        await pilot.press("down")
        assert palette.highlighted == 1
        await pilot.press("up")
        assert palette.highlighted == 0


@pytest.mark.asyncio
async def test_submitted_local_command_is_visible_but_never_submitted_to_runtime() -> (
    None
):
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/context"
        await pilot.press("enter")
        await pilot.pause()
        rendered = pilot.app.query_one("#transcript").renderable_text

    assert runtime.submitted == []
    assert app.state.transcript[0].kind == "local_command"
    assert app.state.transcript[0].text == "/context"
    assert "[notice]" in rendered


@pytest.mark.asyncio
async def test_ctrl_c_during_pending_submit_requests_shutdown_after_submit_settles() -> (
    None
):
    class PendingRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.submit_started = asyncio.Event()
            self.release_submit = asyncio.Event()

        async def submit(self, prompt: str) -> str:
            self.submitted.append(prompt)
            self.submit_started.set()
            await self.release_submit.wait()
            return "pending-run"

    runtime = PendingRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "prompt"
        await pilot.press("enter")
        await asyncio.wait_for(runtime.submit_started.wait(), timeout=1)
        await pilot.press("ctrl+c")
        assert app.is_running
        runtime.release_submit.set()
        await pilot.pause()

    assert runtime.aborted == ["pending-run"]


@pytest.mark.asyncio
async def test_clear_keeps_local_command_visible_before_clearing_prior_rows() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=make_state(
            transcript=[TranscriptItem(kind="assistant", item_id="old", text="old")]
        ),
    )

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/clear"
        await pilot.press("enter")
        await pilot.pause()

    assert [(row.kind, row.text) for row in app.state.transcript] == [
        ("local_command", "/clear"),
    ]


def test_transcript_categories_render_notice_and_error_labels() -> None:
    from coding_agent.tui.reducer import reduce

    state = make_state()
    state = reduce(state, RuntimeEvent(type="notice", payload={"message": "info"}))
    state = reduce(state, RuntimeEvent(type="run_error", payload={"message": "bad"}))

    assert [row.level for row in state.transcript] == ["notice", "error"]
    assert str(TranscriptRow(state.transcript[0], index=0).render()) == "[notice] info"
    assert str(TranscriptRow(state.transcript[1], index=1).render()) == "[error] bad"


@pytest.mark.asyncio
async def test_idle_ctrl_c_clears_text_then_exits_on_empty_composer() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "draft"
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert composer.text == ""
        assert app.is_running
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert not app.is_running


@pytest.mark.asyncio
async def test_idle_ctrl_c_on_empty_composer_requires_two_presses() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        # First ctrl+c on an empty composer arms the confirmation; no exit.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running
        assert app._exit_armed is True
        assert app.state.transcript[-1].text == "Press ctrl+c again to exit"

        # Second ctrl+c exits.
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert not app.is_running


@pytest.mark.asyncio
async def test_shutdown_aborts_run_returned_by_queued_submit_before_run_started_event() -> (
    None
):
    class QueuedSubmitRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.submit_started = asyncio.Event()
            self.release_submit = asyncio.Event()

        async def submit(self, prompt: str) -> str:
            self.submitted.append(prompt)
            self.submit_started.set()
            await self.release_submit.wait()
            return "queued-run"

    runtime = QueuedSubmitRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "queued prompt"
        await pilot.press("enter")
        await asyncio.wait_for(runtime.submit_started.wait(), timeout=1)
        runtime.release_submit.set()
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert runtime.aborted == ["queued-run"]


@pytest.mark.asyncio
async def test_ctrl_c_aborts_returned_run_even_when_composer_has_text() -> None:
    class ReturnedRunRuntime(FakeRuntime):
        async def submit(self, prompt: str) -> str:
            self.submitted.append(prompt)
            return "returned-run"

    runtime = ReturnedRunRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "prompt"
        await pilot.press("enter")
        await pilot.pause()
        composer.text = "draft"
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert runtime.aborted == ["returned-run"]


@pytest.mark.asyncio
async def test_ctrl_c_aborts_known_run_before_delayed_run_started_with_draft() -> None:
    class DelayedRunStartedRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.release_run_started = asyncio.Event()

        async def submit(self, prompt: str) -> str:
            self.submitted.append(prompt)
            self.run_worker = asyncio.create_task(self._emit_run_started())
            return "delayed-run"

        async def _emit_run_started(self) -> None:
            await self.release_run_started.wait()
            await self.emit(RuntimeEvent(type="run_started", run_id="delayed-run"))

    runtime = DelayedRunStartedRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "prompt"
        await pilot.press("enter")
        await pilot.pause()
        assert app.state.status == "idle"
        composer.text = "draft"
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert composer.text == "draft"
    assert runtime.aborted == ["delayed-run"]
    runtime.release_run_started.set()
    await runtime.run_worker


@pytest.mark.asyncio
async def test_quit_waits_for_in_flight_ctrl_c_abort() -> None:
    class BlockingAbortRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.abort_started = asyncio.Event()
            self.release_abort = asyncio.Event()

        async def abort(self, run_id: str) -> None:
            self.aborted.append(run_id)
            self.abort_started.set()
            await self.release_abort.wait()

    runtime = BlockingAbortRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=make_state(status="running", active_run_id="abort-run"),
    )

    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await asyncio.wait_for(runtime.abort_started.wait(), timeout=1)
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/quit"
        await pilot.press("enter")
        await pilot.pause()
        assert app.is_running
        runtime.release_abort.set()
        await pilot.pause()

    assert runtime.aborted == ["abort-run"]
