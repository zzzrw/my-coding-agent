from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from textual.widgets import OptionList, TextArea

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import Usage
from coding_agent.tui.app import (
    CodingAgentApp,
    PermissionFullScreen,
    _RuntimeBridge,
)
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import TranscriptItem, TuiState, initial_state
from coding_agent.tui.widgets import (
    PermissionModeScreen,
    TranscriptRow,
    format_statusline,
)


def event(event_type: str, payload: dict, *, run_id: str | None = None) -> RuntimeEvent:
    return RuntimeEvent(type=event_type, run_id=run_id, payload=payload)


def make_state(**updates: object) -> TuiState:
    return initial_state("/tmp/project", "fake-model", context_window=1000).model_copy(
        update=updates
    )


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

    async def resolve_approval(self, request_id: str, decision: str) -> None:
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

    async def list_sessions(self) -> list:
        return []

    async def resume(self, session_id: str) -> None:
        self.resumes.append(session_id)
        await self.emit(
            RuntimeEvent(type="session_loaded", payload={"session_id": session_id})
        )

    async def compact(self) -> None:
        self.compactions += 1


# ---------------------------------------------------------------------------
# (1) stale run-scoped event rejection
# ---------------------------------------------------------------------------


def test_reducer_rejects_stale_run_scoped_events_but_keeps_unscoped_notice() -> None:
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("run_started", {}, run_id="run-new"),
    )
    baseline = state

    for stale in (
        RuntimeEvent(
            type="assistant_delta",
            run_id="run-old",
            payload={"message_id": "m-old", "text": "stale text"},
        ),
        RuntimeEvent(
            type="assistant_started",
            run_id="run-old",
            payload={"message_id": "m-old"},
        ),
        RuntimeEvent(
            type="user_message",
            run_id="run-old",
            payload={"message_id": "u-old", "text": "stale user"},
        ),
        RuntimeEvent(
            type="tool_started",
            run_id="run-old",
            payload={"tool_call_id": "c-old", "tool_name": "read_file"},
        ),
        RuntimeEvent(
            type="tool_finished",
            run_id="run-old",
            payload={"tool_call_id": "c-old", "ok": True, "content": "x"},
        ),
        RuntimeEvent(
            type="context_updated",
            run_id="run-old",
            payload={"used_tokens": 999, "context_window": 999, "estimated": True},
        ),
        RuntimeEvent(
            type="run_finished",
            run_id="run-old",
            payload={"outcome": {"reason": "aborted"}},
        ),
    ):
        state = reduce(state, stale)
        assert state == baseline

    # A stale run_started must not hijack the active run either.
    state = reduce(
        state, RuntimeEvent(type="run_started", run_id="run-old", payload={})
    )
    assert state.active_run_id == "run-new"

    # An unscoped local notice is preserved while a run is active.
    state = reduce(state, RuntimeEvent(type="notice", payload={"message": "local"}))
    assert state.transcript[-1].kind == "system"
    assert state.transcript[-1].text == "local"


def test_reducer_accepts_event_for_current_run() -> None:
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("run_started", {}, run_id="run-new"),
    )
    state = reduce(
        state,
        RuntimeEvent(
            type="assistant_delta",
            run_id="run-new",
            payload={"message_id": "m", "text": "hello"},
        ),
    )
    assert [row.text for row in state.transcript if row.kind == "assistant"] == [
        "hello"
    ]


@pytest.mark.asyncio
async def test_stale_delta_does_not_leak_into_new_run_transcript() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        await runtime.emit(
            RuntimeEvent(type="run_started", run_id="run-new", turn_id="turn-new")
        )
        await runtime.emit(
            RuntimeEvent(
                type="assistant_delta",
                run_id="run-old",
                payload={"message_id": "m-old", "text": "stale"},
            )
        )
        await pilot.pause()

    assert not any(row.text == "stale" for row in app.state.transcript)


# ---------------------------------------------------------------------------
# (2) bounded bridge memory under many unique delta message ids
# ---------------------------------------------------------------------------


class BridgeTarget:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    def _apply_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    def _show_error(self, message: str) -> None:
        raise AssertionError(message)


@pytest.mark.asyncio
async def test_bridge_memory_is_bounded_with_unique_delta_ids() -> None:
    target = BridgeTarget()
    bridge = _RuntimeBridge(target, maxsize=4)
    bridge.start()

    await bridge.publish(RuntimeEvent(type="run_started", run_id="run-1"))
    for index in range(500):
        await bridge.publish(
            RuntimeEvent(
                type="assistant_delta",
                run_id="run-1",
                payload={"message_id": f"m-{index}", "text": f"t{index}"},
            )
        )
        assert len(bridge._coalesced) <= bridge._max_coalesced

    await bridge.publish(RuntimeEvent(type="run_finished", run_id="run-1"))

    async def wait_for_finish() -> None:
        while not any(event.type == "run_finished" for event in target.events):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_for_finish(), timeout=5)
    await bridge.stop()

    deltas = [event for event in target.events if event.type == "assistant_delta"]
    assert len(deltas) == 500
    # Every delta was delivered before the control event, and none was dropped.
    assert [event.type for event in target.events] == [
        "run_started"
    ] + ["assistant_delta"] * 500 + ["run_finished"]


# ---------------------------------------------------------------------------
# (3) compact busy state, notices, and rejection of prompts/commands
# ---------------------------------------------------------------------------


class BlockingCompactRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.compact_started = asyncio.Event()
        self.release_compact = asyncio.Event()

    async def compact(self) -> None:
        self.compactions += 1
        self.compact_started.set()
        await self.release_compact.wait()


@pytest.mark.asyncio
async def test_compact_sets_busy_state_and_shows_progress_then_result() -> None:
    runtime = BlockingCompactRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/compact"
        await pilot.press("enter")
        await asyncio.wait_for(runtime.compact_started.wait(), timeout=1)

        assert app.state.compacting is True
        assert any("compact" in row.text.lower() for row in app.state.transcript)

        runtime.release_compact.set()
        await pilot.pause()

    assert app.state.compacting is False
    assert runtime.compactions == 1
    assert any("compacted" in row.text.lower() for row in app.state.transcript)


@pytest.mark.asyncio
async def test_compact_rejects_prompt_and_conflicting_commands() -> None:
    runtime = BlockingCompactRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/compact"
        await pilot.press("enter")
        await asyncio.wait_for(runtime.compact_started.wait(), timeout=1)

        composer.text = "a prompt during compact"
        await pilot.press("enter")
        await pilot.pause()

        composer.text = "/new"
        await pilot.press("enter")
        await pilot.pause()

        assert runtime.submitted == []
        assert runtime.new_sessions == 0
        assert any(
            "compact" in row.text.lower() for row in app.state.transcript
        )

        runtime.release_compact.set()
        await pilot.pause()

    assert runtime.compactions == 1


@pytest.mark.asyncio
async def test_compact_failure_is_visible_error_notice_and_clears_busy() -> None:
    class FailingCompactRuntime(FakeRuntime):
        async def compact(self) -> None:
            self.compactions += 1
            raise RuntimeError("compaction failed")

    runtime = FailingCompactRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/compact"
        await pilot.press("enter")
        await pilot.pause()

    assert app.state.compacting is False
    assert runtime.compactions == 1
    assert app.state.status == "idle"
    assert app.state.transcript[-1].kind == "system"
    assert app.state.transcript[-1].level == "error"
    assert "compaction failed" in app.state.transcript[-1].text


# ---------------------------------------------------------------------------
# (4) distinct transcript row boundaries
# ---------------------------------------------------------------------------


def test_local_command_boundary_is_distinct_from_user() -> None:
    user = TranscriptRow(TranscriptItem(kind="user", item_id="u", text="/help"), index=0)
    command = TranscriptRow(
        TranscriptItem(kind="local_command", item_id="c", text="/help"), index=1
    )

    assert str(user.render()).startswith("> ")
    assert str(command.render()).startswith("$ ")
    assert str(user.render()) != str(command.render())


def test_all_transcript_kinds_render_distinct_boundaries() -> None:
    rendered = {
        kind: str(TranscriptRow(item, index=0).render())
        for kind, item in (
            (
                "user",
                TranscriptItem(kind="user", item_id="u", text="same text"),
            ),
            (
                "assistant",
                TranscriptItem(kind="assistant", item_id="a", text="same text"),
            ),
            (
                "tool",
                TranscriptItem(
                    kind="tool",
                    item_id="t",
                    tool_call_id="t",
                    tool_name="read_file",
                    tool_status="success",
                    text="same text",
                ),
            ),
            (
                "local_command",
                TranscriptItem(kind="local_command", item_id="c", text="same text"),
            ),
            (
                "notice",
                TranscriptItem(kind="system", item_id="s1", level="notice", text="same text"),
            ),
            (
                "error",
                TranscriptItem(kind="system", item_id="s2", level="error", text="same text"),
            ),
        )
    }

    assert rendered["user"].startswith("> ")
    assert rendered["local_command"].startswith("$ ")
    assert rendered["assistant"] == "same text"
    assert rendered["tool"].startswith("[success] ")
    assert rendered["notice"].startswith("[notice] ")
    assert rendered["error"].startswith("[error] ")
    assert len(set(rendered.values())) == 6


# ---------------------------------------------------------------------------
# (5) bare /permission opens a discoverable mode selector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_permission_opens_mode_selector_without_runtime_call() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state(policy="workspace"))

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/permission"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(pilot.app.screen, PermissionModeScreen)
        assert runtime.permissions == []
        options = pilot.app.screen.query_one("OptionList", OptionList)
        assert options.option_count == 3
        assert {
            options.get_option_at_index(index).id
            for index in range(options.option_count)
        } == {"default", "workspace", "full"}


@pytest.mark.asyncio
async def test_permission_selector_workspace_applies_directly() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state(policy="default"))

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/permission"
        await pilot.press("enter")
        await pilot.pause()

        options = pilot.app.screen.query_one("OptionList", OptionList)
        options.highlighted = 1  # "workspace"
        await pilot.press("enter")
        await pilot.pause()

    assert runtime.permissions == ["workspace"]


@pytest.mark.asyncio
async def test_permission_selector_full_still_requires_confirmation() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        composer.text = "/permission"
        await pilot.press("enter")
        await pilot.pause()

        options = pilot.app.screen.query_one("OptionList", OptionList)
        options.highlighted = 2  # "full"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(pilot.app.screen, PermissionFullScreen)
        assert runtime.permissions == []

        await pilot.press("enter")  # approve
        await pilot.pause()

    assert runtime.permissions == ["full"]


# ---------------------------------------------------------------------------
# (6) explicit configured/estimated context label
# ---------------------------------------------------------------------------


def test_statusline_labels_context_configured_explicitly() -> None:
    state = make_state(context_used=300, context_window=1000, git_branch=None)
    text = format_statusline(state, usage=Usage(input_tokens=120, output_tokens=45))

    assert "ctx 300/700/1000" in text
    assert "configured" in text
    assert "estimated" not in text
    for width in (0, 1, 10, 40, 80):
        assert len(format_statusline(state, width=width)) <= width


def test_statusline_labels_context_estimated_explicitly() -> None:
    state = make_state(
        context_used=300, context_window=1000, context_estimated=True, git_branch=None
    )
    text = format_statusline(state)
    assert "estimated" in text
    assert "configured" not in text


def test_context_label_after_resume_and_initial_state() -> None:
    # Initial state has no measurement: labeled configured (exact window).
    initial = format_statusline(initial_state("/tmp/project", "fake", context_window=1000))
    assert "configured" in initial

    # Resumed state resets usage to zero and stays non-estimated.
    resumed = reduce(
        initial_state("/tmp/project", "fake"),
        event(
            "session_loaded",
            {
                "session_id": "s1",
                "workspace": "/tmp/project",
                "model": "fake",
                "context_window": 1000,
                "history": [],
            },
        ),
    )
    text = format_statusline(resumed)
    assert "configured" in text
    assert "estimated" not in text

    # A measured (estimated) update flips the label.
    measured = reduce(
        resumed,
        event("context_updated", {"used_tokens": 42, "context_window": 1000, "estimated": True}),
    )
    assert "estimated" in format_statusline(measured)


def test_statusline_fits_80_columns() -> None:
    state = make_state(
        workspace="/very/long/workspace/path/that/keeps/going",
        model="a-rather-long-model-name",
        context_used=12345,
        context_window=100000,
        context_estimated=True,
        session_id="a-very-long-session-identifier",
    )
    assert len(format_statusline(state, width=80)) <= 80
