"""W5 interaction & history tests.

Task 1: help overlay — ``CommandSuggestion`` usage text, the ``HelpScreen``
modal, ``/help`` dispatch, and the ``?`` keybinding.
Task 2: composer prompt-history ring with draft preservation.
Task 3: session selector workspace filter with a browse-all toggle.
Task 4: call-history inbox — ``_inbox_rows`` from ``store.records()`` newest
first, the ``HistoryScreen`` modal, and ``/inbox`` dispatch.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from textual.widgets import OptionList

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import ToolCall
from coding_agent.session.models import SessionRecord, SessionSummary
from coding_agent.tools.models import ToolResult
from coding_agent.tui.app import CodingAgentApp, HelpScreen
from coding_agent.tui.state import TuiState, initial_state
from coding_agent.tui.widgets import HistoryScreen, SessionSelector, SubmitTextArea


class FakeStore:
    """Minimal stand-in for ``SessionStore`` exposing ``records()``."""

    def __init__(self, records: list[SessionRecord] | None = None) -> None:
        self._records = list(records) if records else []

    def records(self) -> list[SessionRecord]:
        return list(self._records)


class FakeRuntime:
    def __init__(self) -> None:
        self.subscribers: list[Callable[[RuntimeEvent], Awaitable[None]]] = []
        self.submitted: list[str] = []
        self.sessions: list[SessionSummary] = []
        self.store = FakeStore()
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


# --- Task 4: call-history inbox ---


def _tool_call_record(
    call_id: str, name: str, arguments: dict[str, object], when: datetime
) -> SessionRecord:
    return SessionRecord(
        id=f"call-{call_id}",
        seq=0,
        timestamp=when,
        type="tool_call",
        payload={"tool_call": ToolCall(id=call_id, name=name, arguments=arguments)},
    )


def _tool_result_record(
    call_id: str,
    name: str,
    *,
    ok: bool,
    error: str | None,
    when: datetime,
) -> SessionRecord:
    return SessionRecord(
        id=f"result-{call_id}",
        seq=0,
        timestamp=when,
        type="tool_result",
        payload={
            "result": ToolResult(
                tool_call_id=call_id,
                tool_name=name,
                ok=ok,
                content="",
                error=error,
            )
        },
    )


def _approval_record(decision: str, tool_name: str, when: datetime) -> SessionRecord:
    return SessionRecord(
        id=f"approval-{decision}-{tool_name}",
        seq=0,
        timestamp=when,
        type="approval",
        payload={
            "request_id": "r1",
            "tool_name": tool_name,
            "decision": decision,
            "scope": "once",
            "feedback": None,
            "tool_call_id": "c1",
        },
    )


def fake_runtime_with_records() -> FakeRuntime:
    """A runtime whose store carries tool/approval records at distinct times."""
    runtime = FakeRuntime()
    when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    runtime.store = FakeStore(
        [
            _tool_call_record("c1", "run_command", {"command": "ls -la"}, when=when),
            _tool_call_record(
                "c2",
                "write_file",
                {"path": "a.txt", "content": "x"},
                when=when + timedelta(minutes=1),
            ),
            _tool_result_record(
                "c2",
                "write_file",
                ok=True,
                error=None,
                when=when + timedelta(minutes=2),
            ),
            _approval_record("approve", "write_file", when=when + timedelta(minutes=3)),
            _tool_result_record(
                "c3",
                "run_command",
                ok=False,
                error="cancelled by user",
                when=when + timedelta(minutes=4),
            ),
        ]
    )
    return runtime


def test_parse_command_inbox():
    from coding_agent.tui.commands import parse_command

    assert parse_command("/inbox").name == "inbox"
    assert parse_command("/inbox").args == []


def test_inbox_builds_rows_from_records_newest_first():
    app = CodingAgentApp(
        runtime=fake_runtime_with_records(), initial_state=make_state()
    )
    rows = app._inbox_rows()
    assert rows
    # the newest record (cancelled run_command result) is first
    assert "cancelled" in rows[0]
    assert "run_command" in rows[0]
    # an approval record surfaces as "approve <tool>"
    assert any("approve write_file" in row for row in rows)
    # tool calls carry their name plus compact args
    assert any("write_file" in row and "path=a.txt" in row for row in rows)


def test_inbox_rows_ignores_other_record_types_and_empty_store():
    app = CodingAgentApp(runtime=FakeRuntime(), initial_state=make_state())
    assert app._inbox_rows() == []


def test_inbox_rows_capped_at_twenty():
    runtime = FakeRuntime()
    when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    records = [
        _tool_call_record(
            f"c{i}",
            "run_command",
            {"command": f"cmd {i}"},
            when=when + timedelta(minutes=i),
        )
        for i in range(25)
    ]
    runtime.store = FakeStore(records)
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())
    rows = app._inbox_rows()
    assert len(rows) == 20
    # newest first: the latest record (c24) is at the head
    assert "cmd 24" in rows[0]
    assert "cmd 5" in rows[-1]


def test_inbox_write_file_row_omits_large_body():
    runtime = FakeRuntime()
    when = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    body = "LARGEBODY-" + "x" * 5000
    runtime.store = FakeStore(
        [
            _tool_call_record(
                "c1", "write_file", {"path": "a.txt", "content": body}, when=when
            )
        ]
    )
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())
    rows = app._inbox_rows()
    assert rows and "write_file" in rows[0]
    assert "path=a.txt" in rows[0]
    assert "content" not in rows[0]
    assert "LARGEBODY" not in rows[0]


@pytest.mark.asyncio
async def test_inbox_command_opens_history_screen():
    app = CodingAgentApp(
        runtime=fake_runtime_with_records(), initial_state=make_state()
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/inbox"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, HistoryScreen)
        # the modal renders the rows, including the approval summary
        assert "approve write_file" in app.screen.body.plain


@pytest.mark.asyncio
async def test_history_screen_renders_rows_and_closes():
    from coding_agent.tui.widgets import history_overlay_text

    body = history_overlay_text(["approve write_file", "tool run_command ls -la"])
    assert "Call history" in body.plain
    assert "approve write_file" in body.plain

    app = CodingAgentApp(runtime=FakeRuntime(), initial_state=make_state())
    async with app.run_test() as pilot:
        app.push_screen(HistoryScreen(["one", "two"]))
        await pilot.pause()
        assert app.screen.body.plain.count("\n") >= 2
        app.screen.action_close_history()
        await pilot.pause()
        assert not isinstance(app.screen, HistoryScreen)


def test_history_screen_empty_state():
    from coding_agent.tui.widgets import history_overlay_text

    body = history_overlay_text([])
    assert "no tool calls yet" in body.plain


# --- Task 5: session selector live search + hint + autofocus ---


def test_session_selector_search_filters_by_id_title_and_workspace():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    sessions = [
        _session_summary("aaaa", "/w1", when=when),
        _session_summary("bbbb", "/w2", when=when),
        _session_summary("cccc", "/w1", when=when),
    ]
    selector = SessionSelector(sessions, workspace="/w1")
    assert [s.id for s in selector.visible_sessions()] == ["aaaa", "cccc"]

    selector._query = "aaaa"  # id search narrows the /w1 list to the matching session
    assert [s.id for s in selector.visible_sessions()] == ["aaaa"]
    selector._query = "bbbb"
    assert [
        s.id for s in selector.visible_sessions()
    ] == []  # workspace /w1 filter still applies
    selector.toggle_filter()  # browse all
    assert [s.id for s in selector.visible_sessions()] == ["bbbb"]
    selector._query = "/w2"
    assert [s.id for s in selector.visible_sessions()] == ["bbbb"]


def test_session_selector_search_matches_title():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    sessions = [
        _session_summary("aaa", "/w1", when=when),
        _session_summary("bbb", "/w1", when=when),
    ]
    # _session_summary builds title = "title-<id>"; override to a real title.
    sessions[0] = sessions[0].model_copy(update={"title": "deploy api fix"})
    selector = SessionSelector(sessions, workspace="/w1")
    selector._query = "deploy"
    assert [s.id for s in selector.visible_sessions()] == ["aaa"]


@pytest.mark.asyncio
async def test_session_selector_focuses_list_and_shows_hint():
    when = datetime(2026, 1, 1, tzinfo=UTC)
    app = CodingAgentApp(
        runtime=FakeRuntime(),
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test() as pilot:
        app.push_screen(SessionSelector([_session_summary("a", "/w1", when=when)]))
        await pilot.pause()
        selector = app.screen
        assert selector.query_one("#session-options", OptionList).has_focus
        assert "↑↓ select" in str(selector.query_one("#session-hint").render())
