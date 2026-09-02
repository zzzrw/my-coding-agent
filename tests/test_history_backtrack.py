from datetime import UTC, datetime, timedelta

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import Message, TurnOutcome
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import initial_state
from coding_agent.tui.widgets import RewindPicker, SubmitTextArea, _relative_time


class _FakeRuntime:
    def __init__(self) -> None:
        self.status: str | None = None
        self.forked: list[str] = []

    def subscribe(self, callback):
        return lambda: None

    async def fork_at(self, message_id: str) -> str:
        self.forked.append(message_id)
        return f"restored:{message_id}"


class _NoopRunner:
    async def run_turn(self, prompt, *, run_id, turn_id, signal, usage=None):
        return TurnOutcome(reason="completed", final_text=prompt, steps=1)


def _store_with_turns(root, *, turns: int = 2) -> SessionStore:
    store = SessionStore.create(
        root, workspace=str(root), model="fake", context_window=1000
    )
    for index in range(turns):
        tid = f"t{index + 1}"
        store.append_new(
            "turn_start", {"turn_id": tid}, run_id=f"r{index + 1}", turn_id=tid
        )
        store.append_new(
            "user_message",
            {"message": Message(role="user", content=f"prompt {index + 1}")},
            run_id=f"r{index + 1}",
            turn_id=tid,
        )
        store.append_new(
            "turn_end",
            {"reason": "completed", "turn_id": tid},
            run_id=f"r{index + 1}",
            turn_id=tid,
        )
    return store


def _runtime(store: SessionStore) -> AgentRuntime:
    return AgentRuntime(
        store=store,
        runner_factory=lambda *_: _NoopRunner(),
        context_policy_factory=lambda: TruncatePolicy(1000),
        approval_policy=DefaultApprovalPolicy(),
        system_prompt=Message(role="system", content="s"),
        model="fake",
    )


@pytest.mark.asyncio
async def test_fork_at_creates_truncated_new_session_and_returns_prompt(tmp_path):
    store = _store_with_turns(tmp_path)
    original_path = store.path
    original_text = original_path.read_text(encoding="utf-8")
    runtime = _runtime(store)
    original_session_id = runtime.session_id

    prompt = await runtime.fork_at("user-t1")

    assert prompt == "prompt 1"
    assert runtime.session_id != original_session_id
    assert [r.type for r in runtime.store.records()] == [
        "turn_start",
        "user_message",
        "turn_end",
    ]
    # Original session file is untouched and still holds every record.
    assert original_path.read_text(encoding="utf-8") == original_text
    # New session file is truncated at the fork point.
    new_text = runtime.store.path.read_text(encoding="utf-8")
    assert "prompt 1" in new_text
    assert "prompt 2" not in new_text


@pytest.mark.asyncio
async def test_fork_at_unknown_message_raises(tmp_path):
    store = _store_with_turns(tmp_path)
    runtime = _runtime(store)
    with pytest.raises(ValueError):
        await runtime.fork_at("user-nope")


@pytest.mark.asyncio
async def test_fork_at_latest_message_works_with_open_turn(tmp_path):
    store = _store_with_turns(tmp_path, turns=1)
    runtime = _runtime(store)
    prompt = await runtime.fork_at("user-t1")
    assert prompt == "prompt 1"
    assert [r.type for r in runtime.store.records()] == [
        "turn_start",
        "user_message",
        "turn_end",
    ]


def test_user_message_row_carries_timestamp() -> None:
    stamp = datetime.now(UTC)
    event = RuntimeEvent(
        type="user_message",
        run_id="r",
        payload={"message_id": "user-t1", "text": "hi"},
        timestamp=stamp,
    )
    state = reduce(initial_state(workspace="/tmp/project", model="fake"), event)
    row = state.transcript[-1]
    assert row.kind == "user"
    assert row.timestamp == stamp


def test_relative_time_formats() -> None:
    now = datetime.now(UTC)
    assert _relative_time(None) == "-"
    assert _relative_time(now - timedelta(seconds=30)) == "30s ago"
    assert _relative_time(now - timedelta(minutes=5)) == "5m ago"
    assert _relative_time(now - timedelta(hours=2)) == "2h ago"


@pytest.mark.asyncio
async def test_rewind_picker_selects_message_id() -> None:
    from coding_agent.tui.app import CodingAgentApp

    rows = [("user-1", "first prompt", "2m ago"), ("user-2", "second", "-")]

    class Host(CodingAgentApp):
        def __init__(self) -> None:
            super().__init__(
                runtime=_FakeRuntime(),
                initial_state=initial_state("/tmp/project", "fake"),
                branch_detector=lambda workspace: None,
            )

        def on_mount(self) -> None:
            self.push_screen(RewindPicker(rows), callback=self._picked)

        def _picked(self, value: str | None) -> None:
            self._picked_value = value

    app = Host()
    async with app.run_test() as pilot:
        picker = app.screen
        assert isinstance(picker, RewindPicker)
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app._picked_value == "user-2"


@pytest.mark.asyncio
async def test_double_escape_opens_rewind_picker() -> None:
    from coding_agent.tui.app import CodingAgentApp
    from coding_agent.tui.state import initial_state

    app = CodingAgentApp(
        runtime=_FakeRuntime(),
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test() as pilot:
        # Seed a user message so the picker has a row.
        app._apply_event(
            RuntimeEvent(
                type="user_message",
                run_id="r",
                payload={"message_id": "user-t1", "text": "hello"},
            )
        )
        await pilot.pause()
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = ""
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RewindPicker)


@pytest.mark.asyncio
async def test_rewind_selected_forks_and_refills_composer() -> None:
    from coding_agent.tui.app import CodingAgentApp
    from coding_agent.tui.state import initial_state

    runtime = _FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        await pilot.pause()
        await app._rewind_selected("user-t1")
        await pilot.pause()
        assert runtime.forked == ["user-t1"]
        assert composer.text == "restored:user-t1"


@pytest.mark.asyncio
async def test_rewind_rejected_while_run_active() -> None:
    from coding_agent.tui.app import CodingAgentApp
    from coding_agent.tui.state import initial_state

    runtime = _FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test() as pilot:
        app._apply_event(
            RuntimeEvent(
                type="run_started",
                run_id="r",
                turn_id="t",
                payload={"session_id": "s", "model": "fake", "policy": "default"},
            )
        )
        await pilot.pause()
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = ""
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, RewindPicker)
        assert runtime.forked == []
