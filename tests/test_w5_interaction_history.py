"""W5 interaction & history tests.

Task 1: help overlay — ``CommandSuggestion`` usage text, the ``HelpScreen``
modal, ``/help`` dispatch, and the ``?`` keybinding.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.app import CodingAgentApp, HelpScreen
from coding_agent.tui.state import TuiState, initial_state
from coding_agent.tui.widgets import SubmitTextArea


class FakeRuntime:
    def __init__(self) -> None:
        self.subscribers: list[Callable[[RuntimeEvent], Awaitable[None]]] = []
        self.status = type("RuntimeStatus", (), {"usage": None})()

    def subscribe(
        self, sink: Callable[[RuntimeEvent], Awaitable[None]]
    ) -> Callable[[], None]:
        self.subscribers.append(sink)

        def unsubscribe() -> None:
            if sink in self.subscribers:
                self.subscribers.remove(sink)

        return unsubscribe


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
