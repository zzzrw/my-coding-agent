"""W5 interaction & history tests.

Task 1: help overlay — ``CommandSuggestion`` usage text, the ``HelpScreen``
modal, ``/help`` dispatch, and the ``?`` keybinding.
Task 2: composer prompt-history ring with draft preservation.
Task 3: session selector workspace filter with a browse-all toggle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from textual.widgets import OptionList

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.session.models import SessionSummary
from coding_agent.tui.app import CodingAgentApp, HelpScreen
from coding_agent.tui.state import TuiState, initial_state
from coding_agent.tui.widgets import SessionSelector, SubmitTextArea


class FakeRuntime:
    def __init__(self) -> None:
        self.subscribers: list[Callable[[RuntimeEvent], Awaitable[None]]] = []
        self.submitted: list[str] = []
        self.sessions: list[SessionSummary] = []
        self.status = type("RuntimeStatus", (), {"usage": None})()

    def subscribe(
        self, sink: Callable[[RuntimeEvent], Awaitable[None]]
    ) -> Callable[[], None]:
        self.subscribers.append(sink)

        def unsubscribe() -> None:
            if sink in self.subscribers:
                self.subscribers.remove(sink)

        return unsubscribe

    async def submit(self, prompt: str) -> str:
        self.submitted.append(prompt)
        return "run-1"

    async def list_sessions(self) -> list[SessionSummary]:
        return self.sessions


def make_state(**updates: object) -> TuiState:
    state = initial_state("/tmp/project", "fake-model", context_window=1000)
    return state.model_copy(update=updates)


def test_command_suggestions_carry_usage():
    from coding_agent.tui.commands import command_suggestions

    for entry in command_suggestions(""):
        assert hasattr(entry, "usage")
        assert isinstance(entry.usage, str)
    suggestions = command_suggestions("resume")
    assert suggestions and suggestions[0].name == "resume"
    assert suggestions[0].usage


def test_help_screen_composes_commands():
    from coding_agent.tui.commands import SUPPORTED_COMMANDS
    from coding_agent.tui.widgets import HelpScreen

    screen = HelpScreen()
    # composing yields at least one Static overlay
    assert any(w for w in screen.compose())
    rendered = screen.body.plain
    # every supported command name is listed, with usage text
    for name in ("help", "undo", "session", "permission"):
        assert f"/{name}" in rendered
    for name in SUPPORTED_COMMANDS:
        assert f"/{name}" in rendered
    # keybinding legend and permission legend are present
    assert "Ctrl+C" in rendered
    assert "↑/↓" in rendered
    assert "?" in rendered
    assert "default" in rendered


def test_parse_command_help_is_unchanged():
    from coding_agent.tui.commands import Command, parse_command

    assert parse_command("/help") == Command(name="help", args=[])
    assert parse_command("/help session") == Command(name="help", args=["session"])


@pytest.mark.asyncio
async def test_help_command_typed_in_composer_opens_help():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/help"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


@pytest.mark.asyncio
async def test_question_mark_binding_opens_help():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())
    assert any(b.key == "?" and b.action == "open_help" for b in app.BINDINGS)

    async with app.run_test() as pilot:
        app.action_open_help()
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


def test_history_cap_and_recall():
    app = CodingAgentApp(runtime=FakeRuntime(), initial_state=make_state())
    for i in range(60):
        app.prompt_history.append(f"p{i}")
    assert len(app.prompt_history) == 50
    assert app.prompt_history[0] == "p10"
    assert app.prompt_history[-1] == "p59"


@pytest.mark.asyncio
async def test_submit_appends_to_prompt_history():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "hello agent"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.submitted == ["hello agent"]
    assert app.prompt_history == ["hello agent"]
    assert app._history_index is None


@pytest.mark.asyncio
async def test_commands_are_not_recorded_in_history():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/context"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.submitted == []
    assert app.prompt_history == []


@pytest.mark.asyncio
async def test_up_recalls_previous_prompt_and_stashes_draft():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as _:
        composer = app.query_one("#composer-input", SubmitTextArea)
        app.prompt_history.append("first prompt")
        app.prompt_history.append("second prompt")
        composer.text = "in-progress draft"

        app._history_recall(-1)

        assert composer.text == "second prompt"
        assert app._history_index == 1
        assert app._history_draft == "in-progress draft"


@pytest.mark.asyncio
async def test_down_recalls_newer_and_restores_draft():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as _:
        composer = app.query_one("#composer-input", SubmitTextArea)
        app.prompt_history.append("first prompt")
        app.prompt_history.append("second prompt")
        composer.text = "draft"

        app._history_recall(-1)
        assert composer.text == "second prompt"
        app._history_recall(-1)
        assert composer.text == "first prompt"

        app._history_recall(+1)
        assert composer.text == "second prompt"
        app._history_recall(+1)
        assert composer.text == "draft"
        assert app._history_index is None

        # staying past the newest entry is a no-op
        app._history_recall(+1)
        assert composer.text == "draft"


@pytest.mark.asyncio
async def test_history_recall_noop_when_history_empty():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as _:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "draft"

        app._history_recall(-1)
        assert composer.text == "draft"
        assert app._history_index is None
        app._history_recall(+1)
        assert composer.text == "draft"
        assert app._history_index is None


@pytest.mark.asyncio
async def test_up_down_keys_recall_composer_history():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        app.prompt_history.append("first prompt")
        app.prompt_history.append("second prompt")
        composer.text = "draft"

        await pilot.press("up")
        await pilot.pause()
        assert composer.text == "second prompt"
        assert app._history_draft == "draft"

        await pilot.press("up")
        await pilot.pause()
        assert composer.text == "first prompt"

        await pilot.press("down")
        await pilot.pause()
        assert composer.text == "second prompt"

        await pilot.press("down")
        await pilot.pause()
        assert composer.text == "draft"
        assert app._history_index is None


# --- Task 3: session selector workspace filter ---


def _session_summary(
    session_id: str, workspace: str, *, when: datetime
) -> SessionSummary:
    return SessionSummary(
        id=session_id,
        workspace=workspace,
        created_at=when,
        updated_at=when,
        title=f"title-{session_id}",
        last_status="idle",
    )


def test_session_selector_filters_by_workspace():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    sessions = [
        _session_summary("a", "/w1", when=when),
        _session_summary("b", "/w2", when=when),
    ]
    selector = SessionSelector(sessions, workspace="/w1")
    assert [s.id for s in selector.visible_sessions()] == ["a"]
    selector.toggle_filter()
    assert [s.id for s in selector.visible_sessions()] == ["a", "b"]
    selector.toggle_filter()
    assert [s.id for s in selector.visible_sessions()] == ["a"]


def test_session_selector_toggle_label_reflects_current_mode():
    selector = SessionSelector([], workspace="/w1")
    assert selector.toggle_label() == "[browse: /w1]"
    selector.toggle_filter()
    assert selector.toggle_label() == "[browse: all]"
    selector.toggle_filter()
    assert selector.toggle_label() == "[browse: /w1]"


def test_session_selector_without_workspace_shows_all_sessions():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    sessions = [
        _session_summary("a", "/w1", when=when),
        _session_summary("b", "/w2", when=when),
    ]
    selector = SessionSelector(sessions)
    assert [s.id for s in selector.visible_sessions()] == ["a", "b"]


@pytest.mark.asyncio
async def test_session_selector_mounted_toggle_button_switches_scope():
    from textual.widgets import Button

    when = datetime(2026, 1, 1, tzinfo=UTC)
    sessions = [
        _session_summary("a", "/w1", when=when),
        _session_summary("b", "/w2", when=when),
    ]
    app = CodingAgentApp(runtime=FakeRuntime(), initial_state=make_state())

    async with app.run_test() as pilot:
        app.push_screen(SessionSelector(sessions, workspace="/w1"))
        await pilot.pause()
        toggle = app.screen.query_one("#session-toggle", Button)
        assert toggle.id == "session-toggle"
        assert str(toggle.label) == "[browse: /w1]"
        options = app.screen.query_one("#session-options", OptionList)
        assert options.option_count == 1
        # pressing the footer toggle re-filters to every session
        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "[browse: all]"
        assert options.option_count == 2


@pytest.mark.asyncio
async def test_open_session_selector_defaults_to_current_workspace():
    runtime = FakeRuntime()
    now = datetime.now(UTC)
    runtime.sessions = [
        _session_summary("ws-a", "/tmp/project", when=now),
        _session_summary("other-b", "/other", when=now - timedelta(hours=1)),
    ]
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/session"
        await pilot.press("enter")
        await pilot.pause()
        options = app.screen.query_one("#session-options", OptionList)
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "ws-a"
        # the toggle (via its key binding) reveals every session
        app.screen.action_toggle_filter()
        await pilot.pause()
        assert options.option_count == 2
        assert options.get_option_at_index(0).id == "ws-a"
        assert options.get_option_at_index(1).id == "other-b"
