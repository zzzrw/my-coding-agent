"""W1 real-time feedback tests: streamed tool output + statusline spinner/elapsed.

Task 1 coverage: the tool output sink threaded through ``ToolContext`` ->
``_ShellTool`` -> ``ToolExecutor``. Task 2 coverage: ``AgentRunner`` emits
``tool_output_delta`` events for streamed tool output. Task 3 coverage: the
TUI reducer accumulates deltas into the tool row and tracks run timing via
``TuiState.run_started_at`` / ``TuiState.spinner_frame``. Task 4 coverage:
the ``_RuntimeBridge`` coalesces ``tool_output_delta`` under the
``(generation, tool_call_id)`` key without reordering control events.
"""

import asyncio

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import ToolCall
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.registry import ToolContext, ToolRegistry
from coding_agent.tools.shell import make_run_command_tool
from coding_agent.tui.app import CodingAgentApp, _RuntimeBridge
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import initial_state
from coding_agent.tui.widgets import SPINNER_FRAMES, format_statusline


class _NoopBroker:
    async def request(self, request):
        return "approve"

    def cancel_all(self):
        pass


async def test_tool_context_carries_on_output():
    collected: list[str] = []

    async def sink(text: str) -> None:
        collected.append(text)

    context = ToolContext(workspace=".", permission_mode="full", on_output=sink)
    assert context.on_output is not None
    await context.on_output("hello")
    assert collected == ["hello"]


async def test_shell_tool_streams_output_chunks(tmp_path):
    collected: list[str] = []

    async def sink(text: str) -> None:
        collected.append(text)

    tool = make_run_command_tool()
    context = ToolContext(
        workspace=str(tmp_path), permission_mode="full", on_output=sink
    )
    signal = asyncio.Event()
    result = await tool.execute(
        {"command": "printf 'one\\ntwo\\n'"}, context=context, signal=signal
    )
    assert result.ok
    assert result.content == "one\ntwo\n"
    assert collected and "".join(collected) == "one\ntwo\n"


async def test_executor_forwards_output_sink(tmp_path):
    collected: list[str] = []

    async def sink(text: str) -> None:
        collected.append(text)

    registry = ToolRegistry()
    registry.register(make_run_command_tool())
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), _NoopBroker())
    call = ToolCall(id="c1", name="run_command", arguments={"command": "printf hi"})
    signal = asyncio.Event()
    result = await executor.execute(
        call,
        run_id="r",
        workspace=tmp_path,
        permission_mode="full",
        signal=signal,
        output_sink=sink,
    )
    assert result.ok
    assert "".join(collected) == "hi"


class _ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    async def stream(self, messages, tools, *, model, signal):
        for event in self.responses.pop(0):
            yield event


async def test_runner_emits_tool_output_delta(tmp_path):
    from coding_agent.context.truncate import TruncatePolicy
    from coding_agent.runtime.events import RuntimeEvent
    from coding_agent.runtime.models import LLMEvent, Message
    from coding_agent.runtime.runner import AgentRunner
    from coding_agent.session.store import SessionStore

    events: list[RuntimeEvent] = []

    async def sink(event: RuntimeEvent) -> None:
        events.append(event)

    # Two provider turns, matching test_runner.py: a tool-call stream followed by
    # a text stream. A later `response_end(finish_reason="stop")` in the same
    # stream would overwrite `tool_calls` and mark the call truncated.
    tool_turn = [
        LLMEvent(
            type="tool_call_start", tool_call_id="call-1", tool_name="run_command"
        ),
        LLMEvent(
            type="tool_call_delta",
            tool_call_id="call-1",
            arguments_delta='{"command": "seq 1 2000"}',
        ),
        LLMEvent(type="response_end", finish_reason="tool_calls"),
    ]
    final_turn = [
        LLMEvent(type="text_delta", text="done"),
        LLMEvent(type="response_end", finish_reason="stop"),
    ]

    registry = ToolRegistry()
    registry.register(make_run_command_tool())
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), _NoopBroker())
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="test",
        context_window=100_000,
    )
    store.append_new("turn_start", {"turn_id": "t1"}, run_id="r1", turn_id="t1")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="run it")},
        run_id="r1",
        turn_id="t1",
    )
    runner = AgentRunner(
        provider=_ScriptedProvider([tool_turn, final_turn]),
        registry=registry,
        executor=executor,
        context_policy=TruncatePolicy(),
        store=store,
        event_sink=sink,
        system_prompt=Message(role="system", content="sys"),
        model="test",
        context_window=100_000,
        permission_mode="full",
    )

    outcome = await runner.run_turn(
        "run it", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    deltas = [e for e in events if e.type == "tool_output_delta"]
    assert deltas, "expected tool_output_delta events"
    assert all(e.payload["tool_call_id"] == "call-1" for e in deltas)
    finished = next(
        e
        for e in events
        if e.type == "tool_finished" and e.payload["tool_call_id"] == "call-1"
    )
    assert "".join(e.payload["text"] for e in deltas) == finished.payload["content"]
    # Deltas arrive between tool_started and tool_finished, in stream order.
    indices = {
        e.type: i
        for i, e in enumerate(events)
        if e.type in {"tool_started", "tool_finished"}
    }
    assert indices["tool_started"] < min(
        i for i, e in enumerate(events) if e.type == "tool_output_delta"
    )
    assert (
        max(i for i, e in enumerate(events) if e.type == "tool_output_delta")
        < indices["tool_finished"]
    )


def _delta_event(run_id: str, turn_id: str, call_id: str, text: str) -> RuntimeEvent:
    return RuntimeEvent(
        type="tool_output_delta",
        run_id=run_id,
        turn_id=turn_id,
        payload={"tool_call_id": call_id, "text": text},
    )


def test_reducer_accumulates_tool_output_delta():
    state = initial_state(".", "test")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_started",
            run_id="r",
            turn_id="t",
            payload={
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "arguments": {"command": "seq 10"},
            },
        ),
    )
    state = reduce(state, _delta_event("r", "t", "c1", "1\n2\n"))
    state = reduce(state, _delta_event("r", "t", "c1", "3\n"))

    tool_rows = [row for row in state.transcript if row.kind == "tool"]
    assert tool_rows and tool_rows[0].text == "1\n2\n3\n"
    # Deltas never flip the tool row out of running, and the command label
    # derived from tool_started is preserved.
    assert tool_rows[0].tool_status == "running"
    assert tool_rows[0].command == "seq 10"


def test_reducer_creates_tool_row_from_delta_when_started_missing():
    state = reduce(initial_state(".", "test"), _delta_event("r", "t", "c1", "out"))

    rows = [row for row in state.transcript if row.kind == "tool"]
    assert len(rows) == 1
    assert rows[0].tool_call_id == "c1"
    assert rows[0].tool_status == "running"
    assert rows[0].text == "out"


def test_run_started_sets_timing_and_finished_clears():
    state = initial_state(".", "test")
    state = reduce(
        state,
        RuntimeEvent(
            type="run_started",
            run_id="r",
            turn_id="t",
            payload={"session_id": "s", "model": "test", "policy": "default"},
        ),
    )
    assert state.run_started_at is not None
    assert state.spinner_frame == 0
    state = reduce(
        state,
        RuntimeEvent(
            type="run_finished",
            run_id="r",
            turn_id="t",
            payload={"outcome": {"reason": "completed"}, "steps": 1},
        ),
    )
    assert state.run_started_at is None


def test_run_error_clears_timing_and_session_loaded_resets():
    state = reduce(
        initial_state(".", "test"),
        RuntimeEvent(
            type="run_started",
            run_id="r",
            turn_id="t",
            payload={"session_id": "s", "model": "test", "policy": "default"},
        ),
    )
    assert state.run_started_at is not None
    state = reduce(
        state,
        RuntimeEvent(
            type="run_error",
            run_id="r",
            turn_id="t",
            payload={"code": "provider_error", "message": "unavailable"},
        ),
    )
    assert state.run_started_at is None

    # A fresh session load resets both the elapsed anchor and the spinner.
    state = state.model_copy(update={"spinner_frame": 4})
    state = reduce(
        state,
        RuntimeEvent(
            type="session_loaded",
            payload={"session_id": "s2", "workspace": ".", "model": "test"},
        ),
    )
    assert state.run_started_at is None
    assert state.spinner_frame == 0


# ---------------------------------------------------------------------------
# Task 4: _RuntimeBridge coalesces tool_output_delta
# ---------------------------------------------------------------------------


async def test_bridge_coalesces_tool_output_delta() -> None:
    applied: list[RuntimeEvent] = []

    class FakeApp:
        def _apply_event(self, event: RuntimeEvent) -> None:
            applied.append(event)

        def _show_error(self, message: str) -> None:
            raise AssertionError(message)

    bridge = _RuntimeBridge(FakeApp(), maxsize=1)
    # Pre-fill the queue with a control event so the streamed deltas for the
    # same call must be buffered rather than entering it; only then does the
    # bridge merge them under the (generation, tool_call_id) key.
    bridge.queue.put_nowait(
        RuntimeEvent(
            type="run_started",
            run_id="r",
            payload={"session_id": "s", "model": "test", "policy": "default"},
        )
    )
    bridge.start()

    await bridge.publish(_delta_event("r", "t", "c1", "ab"))
    await bridge.publish(_delta_event("r", "t", "c1", "cd"))
    await bridge.publish(
        RuntimeEvent(
            type="tool_finished",
            run_id="r",
            turn_id="t",
            payload={"tool_call_id": "c1", "ok": True, "content": "abcd"},
        )
    )

    async def wait_for_finish() -> None:
        while not any(event.type == "tool_finished" for event in applied):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_for_finish(), timeout=5)
    await bridge.stop()

    delta_texts = [
        event.payload["text"] for event in applied if event.type == "tool_output_delta"
    ]
    # Both streamed chunks merge into a single delta, concatenated in stream
    # order, and the control event never overtakes them.
    assert delta_texts == ["abcd"]
    assert applied[-1].type == "tool_finished"


# ---------------------------------------------------------------------------
# Task 5: statusline spinner + elapsed timer
# ---------------------------------------------------------------------------


def test_statusline_renders_spinner_frame_and_elapsed() -> None:
    state = initial_state(".", "test", context_window=10)
    state = state.model_copy(
        update={"status": "running", "run_started_at": 100.0, "spinner_frame": 2}
    )
    text = format_statusline(state, now=105.0)
    rendered = str(text)
    # Frame index 2 renders its spinner glyph alongside the status and a live
    # elapsed timer (5s from run_started_at=100 to now=105).
    assert SPINNER_FRAMES[2] in rendered
    assert "running" in rendered
    assert "⏱" in rendered
    assert "5s" in rendered


def test_statusline_spinner_frame_wraps_around() -> None:
    state = initial_state(".", "test")
    state = state.model_copy(
        update={"status": "running", "run_started_at": 100.0, "spinner_frame": 10}
    )
    text = format_statusline(state, now=105.0)
    # frame index 10 wraps to index 0 of SPINNER_FRAMES.
    assert SPINNER_FRAMES[0] in str(text)


def test_statusline_waiting_approval_keeps_elapsed_but_pauses_spinner() -> None:
    state = initial_state(".", "test")
    state = state.model_copy(
        update={
            "status": "waiting_approval",
            "run_started_at": 100.0,
            "spinner_frame": 2,
        }
    )
    text = format_statusline(state, now=107.0)
    rendered = str(text)
    # Elapsed time keeps ticking while approval is pending...
    assert "waiting_approval" in rendered
    assert "⏱" in rendered
    assert "7s" in rendered
    # ...but the animated frame at the current index is not shown (paused).
    assert SPINNER_FRAMES[2] not in rendered


def test_statusline_running_without_start_shows_zero_elapsed() -> None:
    state = initial_state(".", "test")
    state = state.model_copy(update={"status": "running", "spinner_frame": 0})
    text = format_statusline(state, now=1.0)
    assert "⏱0s" in str(text)


class _FakeAppRuntime:
    """Minimal runtime stub for app-level spinner-timer tests."""

    def __init__(self) -> None:
        self.subscribers = []
        self.workspace = "/tmp/project"
        self.model = "fake"
        self.status = type("S", (), {"usage": None})()

    def subscribe(self, sink):
        self.subscribers.append(sink)
        return lambda: None


async def test_app_spinner_timer_advances_frame_and_stops_on_finish() -> None:
    runtime = _FakeAppRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test(size=(100, 24)):
        # Idle: no spinner interval is running.
        assert app._spinner_interval is None

        app._apply_event(
            RuntimeEvent(
                type="run_started",
                run_id="r",
                turn_id="t",
                payload={"session_id": "s", "model": "fake", "policy": "default"},
            )
        )
        assert app.state.status == "running"
        assert app.state.run_started_at is not None
        assert app._spinner_interval is not None

        # The timer callback advances the spinner frame and refreshes.
        frame_before = app.state.spinner_frame
        app._tick_spinner()
        assert app.state.spinner_frame == frame_before + 1

        # Finishing the run clears the elapsed anchor and stops the timer.
        app._apply_event(
            RuntimeEvent(
                type="run_finished",
                run_id="r",
                turn_id="t",
                payload={"outcome": {"reason": "completed"}, "steps": 1},
            )
        )
        assert app.state.run_started_at is None
        assert app._spinner_interval is None
