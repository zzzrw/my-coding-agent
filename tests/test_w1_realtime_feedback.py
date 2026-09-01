"""W1 real-time feedback tests: streamed tool output + statusline spinner/elapsed.

Task 1 coverage: the tool output sink threaded through ``ToolContext`` ->
``_ShellTool`` -> ``ToolExecutor``. Task 2 coverage: ``AgentRunner`` emits
``tool_output_delta`` events for streamed tool output.
"""

import asyncio

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import ToolCall
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.registry import ToolContext, ToolRegistry
from coding_agent.tools.shell import make_run_command_tool


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
