"""W3 parallelism tests: wave partitioning + parallel tool execution.

Task 1 coverage: ``partition_waves`` batches parallel-safe calls and
distinct-path mutations while serializing same-path mutations; ``AgentRunner``
executes each wave concurrently via ``asyncio.gather``, preserving per-call
event order (tool_call record -> tool_started -> output deltas -> tool_result
-> tool_finished) and collecting results back into call order.
"""

import asyncio
import json
import time

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.runtime.models import LLMEvent, Message, ToolCall
from coding_agent.runtime.runner import AgentRunner, partition_waves
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult
from coding_agent.tools.registry import ToolRegistry


def test_partition_waves_batches_parallel_safe_and_serializes_same_path():
    calls = [
        ToolCall(id="1", name="read_file", arguments={"path": "a"}),  # parallel-safe
        ToolCall(id="2", name="read_file", arguments={"path": "b"}),  # parallel-safe
        ToolCall(id="3", name="write_file", arguments={"path": "x"}),  # mutate
        ToolCall(id="4", name="write_file", arguments={"path": "x"}),  # same path
        ToolCall(id="5", name="read_file", arguments={"path": "c"}),  # parallel-safe
    ]
    waves = partition_waves(calls)
    assert waves[0] == [calls[0], calls[1]]  # both reads batch
    assert calls[3] in waves[1] or calls[3] in waves[2]  # ordered after calls[2]
    # flatten preserves call order
    assert [c.id for w in waves for c in w] == [c.id for c in calls]


def test_partition_waves_batches_different_path_mutations():
    calls = [
        ToolCall(id="1", name="write_file", arguments={"path": "a.txt"}),
        ToolCall(id="2", name="write_file", arguments={"path": "b.txt"}),
    ]
    waves = partition_waves(calls)
    assert waves == [[calls[0], calls[1]]]


def test_partition_waves_read_joins_mutation_wave_but_not_vice_versa():
    # A read may overlap a distinct-path write once a write wave is open.
    calls = [
        ToolCall(id="1", name="write_file", arguments={"path": "a"}),
        ToolCall(id="2", name="read_file", arguments={"path": "b"}),
    ]
    assert partition_waves(calls) == [[calls[0], calls[1]]]
    # A write never folds into a reads-only wave.
    reversed_calls = [
        ToolCall(id="1", name="read_file", arguments={"path": "a"}),
        ToolCall(id="2", name="write_file", arguments={"path": "b"}),
    ]
    waves = partition_waves(reversed_calls)
    assert len(waves) == 2
    assert all(len(w) == 1 for w in waves)


def test_partition_waves_serializes_non_parallel_safe_non_path_tools():
    calls = [
        ToolCall(id="1", name="run_command", arguments={"command": "ls"}),
        ToolCall(id="2", name="run_command", arguments={"command": "git status"}),
        ToolCall(id="3", name="mystery_tool", arguments={}),
    ]
    waves = partition_waves(calls)
    assert len(waves) == 3
    assert all(len(w) == 1 for w in waves)


def test_partition_waves_resolves_relative_and_absolute_same_path(tmp_path):
    calls = [
        ToolCall(id="1", name="write_file", arguments={"path": "x"}),
        ToolCall(id="2", name="write_file", arguments={"path": str(tmp_path / "x")}),
    ]
    waves = partition_waves(calls, workspace=tmp_path)
    # Same resolved file -> serialized into separate waves.
    assert len(waves) == 2
    assert all(len(w) == 1 for w in waves)


class _ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    async def stream(self, messages, tools, *, model, signal):
        for event in self.responses.pop(0):
            yield event


class _SleepingExecutor:
    """Fake executor that sleeps per call and records start/end timestamps."""

    def __init__(self, delay):
        self.delay = delay
        self.starts = {}
        self.ends = {}
        self.executed = []

    async def execute(self, call, **kwargs):
        self.starts[call.id] = time.monotonic()
        self.executed.append(call)
        await asyncio.sleep(self.delay)
        self.ends[call.id] = time.monotonic()
        return ToolResult(
            tool_call_id=call.id, tool_name=call.name, ok=True, content="ok"
        )


def _tool_calls_response(calls):
    events = []
    for call_id, name, arguments in calls:
        events.append(
            LLMEvent(type="tool_call_start", tool_call_id=call_id, tool_name=name)
        )
        events.append(
            LLMEvent(
                type="tool_call_delta",
                tool_call_id=call_id,
                arguments_delta=json.dumps(arguments),
            )
        )
    events.append(LLMEvent(type="tool_call_end", finish_reason="tool_calls"))
    events.append(LLMEvent(type="response_end", finish_reason="tool_calls"))
    return events


def _make_runner(tmp_path, provider, executor):
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="test",
        context_window=1000,
    )
    store.append_new("turn_start", {"turn_id": "t1"}, run_id="r1", turn_id="t1")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="write files")},
        run_id="r1",
        turn_id="t1",
    )
    events = []

    async def sink(event):
        events.append(event)

    runner = AgentRunner(
        provider=provider,
        registry=ToolRegistry(),
        executor=executor,
        context_policy=TruncatePolicy(1000),
        store=store,
        event_sink=sink,
        system_prompt=Message(role="system", content="system"),
        model="test",
        context_window=1000,
        permission_mode="full",
    )
    return runner, events


async def test_distinct_path_slow_tools_run_in_parallel_before_same_path_third(
    tmp_path,
):
    delay = 0.2
    provider = _ScriptedProvider(
        [
            _tool_calls_response(
                [
                    ("c1", "write_file", {"path": "a.txt", "content": "A"}),
                    ("c2", "write_file", {"path": "b.txt", "content": "B"}),
                    ("c3", "write_file", {"path": "a.txt", "content": "A2"}),
                ]
            ),
            [
                LLMEvent(type="text_delta", text="done"),
                LLMEvent(type="response_end", finish_reason="stop"),
            ],
        ]
    )
    executor = _SleepingExecutor(delay)
    runner, _ = _make_runner(tmp_path, provider, executor)

    started = time.monotonic()
    outcome = await runner.run_turn(
        "write files", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    elapsed = time.monotonic() - started

    assert outcome.reason == "completed"
    assert len(executor.executed) == 3
    # Every slow tool slept its full delay.
    for call_id in ("c1", "c2", "c3"):
        assert executor.ends[call_id] - executor.starts[call_id] >= delay - 0.01
    # The two distinct-path tools overlapped (ran concurrently).
    assert executor.starts["c2"] < executor.ends["c1"]
    assert executor.starts["c1"] < executor.ends["c2"]
    # The same-path third did not overlap the first write to a.txt.
    assert executor.starts["c3"] >= executor.ends["c1"]
    # Two waves of one delay each, not three serialized delays.
    assert elapsed < 2.5 * delay


async def test_parallel_execution_preserves_per_call_event_order(tmp_path):
    provider = _ScriptedProvider(
        [
            _tool_calls_response(
                [
                    ("c1", "read_file", {"path": "a.txt"}),
                    ("c2", "read_file", {"path": "b.txt"}),
                ]
            ),
            [
                LLMEvent(type="text_delta", text="done"),
                LLMEvent(type="response_end", finish_reason="stop"),
            ],
        ]
    )
    executor = _SleepingExecutor(0.01)
    runner, events = _make_runner(tmp_path, provider, executor)

    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    for call_id in ("c1", "c2"):
        started = next(
            i
            for i, e in enumerate(events)
            if e.type == "tool_started" and e.payload["tool_call_id"] == call_id
        )
        finished = next(
            i
            for i, e in enumerate(events)
            if e.type == "tool_finished" and e.payload["tool_call_id"] == call_id
        )
        assert started < finished
