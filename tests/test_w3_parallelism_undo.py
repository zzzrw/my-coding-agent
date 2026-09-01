"""W3 parallelism tests: wave partitioning + parallel tool execution + undo.

Task 1 coverage: ``partition_waves`` batches parallel-safe calls and
distinct-path mutations while serializing same-path mutations; ``AgentRunner``
executes each wave concurrently via ``asyncio.gather``, preserving per-call
event order (tool_call record -> tool_started -> output deltas -> tool_result
-> tool_finished) and collecting results back into call order.

Task 3 coverage: ``MutationJournal`` snapshots file state before successful
``mutate_file`` tools; ``AgentRuntime.undo()`` restores the most recent snapshot
(or unlinks a created file) and emits a local notice; ``/undo`` parses and
dispatches through the TUI.
"""

import asyncio
import json
import time
from pathlib import Path

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.runtime.models import LLMEvent, Message, ToolCall
from coding_agent.runtime.runner import AgentRunner, partition_waves
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tui.commands import SUPPORTED_COMMANDS


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


def test_reducer_renders_plan_banner():
    from coding_agent.runtime.events import RuntimeEvent
    from coding_agent.tui.reducer import reduce
    from coding_agent.tui.state import initial_state

    state = initial_state(".", "test")
    state = reduce(
        state,
        RuntimeEvent(
            type="plan_preview",
            run_id="r",
            turn_id="t",
            payload={
                "tool_calls": [
                    {"name": "write_file", "arguments": {"path": "a"}},
                    {"name": "run_command", "arguments": {"command": "ls"}},
                ]
            },
        ),
    )
    systems = [r for r in state.transcript if r.kind == "system"]
    assert systems and "write_file(a)" in systems[0].text
    assert "run_command(ls)" in systems[0].text
    assert systems[0].text.startswith("→ 2 tool calls:")


async def test_runner_emits_plan_preview_for_multi_call_turns(tmp_path):
    provider = _ScriptedProvider(
        [
            _tool_calls_response(
                [
                    ("c1", "write_file", {"path": "a.txt", "content": "A"}),
                    ("c2", "run_command", {"command": "ls"}),
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
        "do it", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    previews = [e for e in events if e.type == "plan_preview"]
    assert len(previews) == 1
    calls = previews[0].payload["tool_calls"]
    assert [c["name"] for c in calls] == ["write_file", "run_command"]
    assert calls[0]["arguments"] == {"path": "a.txt", "content": "A"}


async def test_runner_does_not_emit_plan_preview_for_single_call(tmp_path):
    provider = _ScriptedProvider(
        [
            _tool_calls_response([("c1", "read_file", {"path": "a.txt"})]),
            [
                LLMEvent(type="text_delta", text="done"),
                LLMEvent(type="response_end", finish_reason="stop"),
            ],
        ]
    )
    executor = _SleepingExecutor(0.01)
    runner, events = _make_runner(tmp_path, provider, executor)

    outcome = await runner.run_turn(
        "read", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    assert all(e.type != "plan_preview" for e in events)


# ---------------------------------------------------------------------------
# Task 3: MutationJournal + /undo
# ---------------------------------------------------------------------------


def test_journal_push_pop_is_lifo():
    from coding_agent.tools.executor import MutationJournal

    journal = MutationJournal()
    assert journal.pop() is None
    journal.push(Path("/a.txt"), "one")
    journal.push(Path("/b.txt"), None)
    assert journal.pop() == (Path("/b.txt"), None)
    assert journal.pop() == (Path("/a.txt"), "one")
    assert journal.pop() is None


class _StubBroker:
    async def request(self, request):
        return "deny"

    def cancel_all(self):
        pass


def _real_executor(tmp_path, *, journal=None):
    from coding_agent.policy.approval import DefaultApprovalPolicy
    from coding_agent.tools.executor import ToolExecutor
    from coding_agent.tools.filesystem import (
        make_edit_file_tool,
        make_read_file_tool,
        make_write_file_tool,
    )

    registry = ToolRegistry()
    registry.register(make_read_file_tool())
    registry.register(make_write_file_tool())
    registry.register(make_edit_file_tool())
    return ToolExecutor(
        registry, DefaultApprovalPolicy(), _StubBroker(), journal=journal
    )


async def test_executor_snapshots_before_and_pushes_after_mutate_file(tmp_path):
    from coding_agent.tools.executor import MutationJournal

    (tmp_path / "a.txt").write_text("old", encoding="utf-8")
    journal = MutationJournal()
    executor = _real_executor(tmp_path, journal=journal)
    result = await executor.execute(
        ToolCall(
            id="c1",
            name="write_file",
            arguments={"path": "a.txt", "content": "new"},
        ),
        run_id="r1",
        workspace=tmp_path,
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert result.ok
    path, original = journal.pop()
    assert path == (tmp_path / "a.txt").resolve()
    assert original == "old"


async def test_executor_records_created_file_as_none_and_skips_failed_mutation(
    tmp_path,
):
    from coding_agent.tools.executor import MutationJournal

    journal = MutationJournal()
    executor = _real_executor(tmp_path, journal=journal)
    created = await executor.execute(
        ToolCall(
            id="c1",
            name="write_file",
            arguments={"path": "new.txt", "content": "hi"},
        ),
        run_id="r1",
        workspace=tmp_path,
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert created.ok
    path, original = journal.pop()
    assert path == (tmp_path / "new.txt").resolve()
    assert original is None
    failed = await executor.execute(
        ToolCall(
            id="c2",
            name="edit_file",
            arguments={"path": "a.txt", "old_text": "nope", "new_text": "x"},
        ),
        run_id="r1",
        workspace=tmp_path,
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert not failed.ok
    assert journal.pop() is None


async def test_executor_does_not_journal_read_tools(tmp_path):
    from coding_agent.tools.executor import MutationJournal

    (tmp_path / "a.txt").write_text("text", encoding="utf-8")
    journal = MutationJournal()
    executor = _real_executor(tmp_path, journal=journal)
    result = await executor.execute(
        ToolCall(id="c1", name="read_file", arguments={"path": "a.txt"}),
        run_id="r1",
        workspace=tmp_path,
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert result.ok
    assert journal.pop() is None


def _tool_response(call_id, name, arguments):
    return [
        LLMEvent(type="tool_call_start", tool_call_id=call_id, tool_name=name),
        LLMEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=json.dumps(arguments),
        ),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]


class _SequencedProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    async def stream(self, messages, tools, *, model, signal):
        for event in self.responses.pop(0):
            if signal.is_set():
                return
            yield event


def _undo_app(tmp_path, responses):
    from coding_agent.app import create_app

    return create_app(
        workspace=tmp_path,
        model="fake",
        session_dir=tmp_path / "sessions",
        context_window=2000,
        provider=_SequencedProvider(responses),
        permission_mode="full",
    )


async def test_undo_restores_overwritten_file(tmp_path):
    application = _undo_app(
        tmp_path,
        [
            _tool_response("c1", "write_file", {"path": "a.txt", "content": "first"}),
            _tool_response("c2", "write_file", {"path": "a.txt", "content": "second"}),
            [
                LLMEvent(type="text_delta", text="done"),
                LLMEvent(type="response_end", finish_reason="stop"),
            ],
        ],
    )
    runtime = application.runtime
    events = []
    finished = asyncio.Event()

    async def sink(event):
        events.append(event)
        if event.type == "run_finished":
            finished.set()

    runtime.subscribe(sink)
    await runtime.submit("write a.txt twice")
    await asyncio.wait_for(finished.wait(), timeout=5)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "second"

    await runtime.undo()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "first"
    notices = [
        event
        for event in events
        if event.type == "notice"
        and event.payload.get("command", "").startswith("undo ")
    ]
    assert notices
    assert notices[-1].payload["message"] == (
        f"undid write to {(tmp_path / 'a.txt').resolve()}"
    )


async def test_undo_removes_created_file(tmp_path):
    application = _undo_app(
        tmp_path,
        [
            _tool_response("c1", "write_file", {"path": "new.txt", "content": "hi"}),
            [
                LLMEvent(type="text_delta", text="done"),
                LLMEvent(type="response_end", finish_reason="stop"),
            ],
        ],
    )
    runtime = application.runtime
    finished = asyncio.Event()

    async def sink(event):
        if event.type == "run_finished":
            finished.set()

    runtime.subscribe(sink)
    await runtime.submit("create new.txt")
    await asyncio.wait_for(finished.wait(), timeout=5)
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hi"

    await runtime.undo()
    assert not (tmp_path / "new.txt").exists()


async def test_undo_with_empty_journal_is_a_noop_that_emits_a_notice(tmp_path):
    application = _undo_app(tmp_path, [])
    runtime = application.runtime
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.undo()
    notices = [event for event in events if event.type == "notice"]
    assert any(event.payload.get("message") == "nothing to undo" for event in notices)


def test_undo_command_parses():
    from coding_agent.tui.commands import Command, parse_command

    assert parse_command("/undo") == Command(name="undo", args=[])
    assert "undo" in SUPPORTED_COMMANDS


async def test_undo_command_dispatches_to_runtime(tmp_path):
    from coding_agent.tui.app import CodingAgentApp
    from coding_agent.tui.state import initial_state

    class _UndoRuntime:
        def __init__(self):
            self.undo_calls = 0

        def subscribe(self, sink):
            return lambda: None

        async def undo(self):
            self.undo_calls += 1

    runtime = _UndoRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state(str(tmp_path), "fake", context_window=1000),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._dispatch_command("undo", [])
        for _ in range(10):
            await pilot.pause()
            if runtime.undo_calls == 1:
                break
    assert runtime.undo_calls == 1
