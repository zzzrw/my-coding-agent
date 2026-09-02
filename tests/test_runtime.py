import asyncio

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import Message, ToolCall, TurnOutcome
from coding_agent.runtime.runtime import AgentRuntime, _ApprovalBroker
from coding_agent.session.models import ApprovalRequest
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult


class BlockingRunner:
    def __init__(self, gate=None):
        self.gate = gate or asyncio.Event()

    async def run_turn(self, prompt, *, run_id, turn_id, signal, usage=None):
        await self.gate.wait()
        return TurnOutcome(reason="completed", final_text=prompt, steps=1)


class EventRunner:
    def __init__(self):
        self.event_sink = None
        self.permission_mode = "default"
        self.last_usage = None

    async def run_turn(self, prompt, *, run_id, turn_id, signal, usage=None):
        self.last_usage = usage
        await self.event_sink(
            RuntimeEvent(
                type="context_updated",
                run_id=run_id,
                turn_id=turn_id,
                payload={"used_tokens": 7, "context_window": 100, "estimated": True},
            )
        )
        return TurnOutcome(reason="completed", final_text=prompt, steps=1)


def make_runtime(tmp_path, runner=None):
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="fake",
        context_window=1000,
    )
    runner = runner or BlockingRunner(asyncio.Event())
    return AgentRuntime(
        store=store,
        runner_factory=lambda *_: runner,
        context_policy_factory=lambda: TruncatePolicy(1000),
        approval_policy=DefaultApprovalPolicy(),
        system_prompt=Message(role="system", content="system"),
        model="fake",
    ), runner


@pytest.mark.asyncio
async def test_concurrent_submit_reserves_only_one_active_run(tmp_path):
    runtime, runner = make_runtime(tmp_path)
    first_started = asyncio.Event()
    release_first_started = asyncio.Event()

    async def blocking_sink(event):
        if event.type == "run_started" and not first_started.is_set():
            first_started.set()
            await release_first_started.wait()

    runtime.subscribe(blocking_sink)
    first = asyncio.create_task(runtime.submit("first"))
    await first_started.wait()
    second = asyncio.create_task(runtime.submit("second"))
    await asyncio.sleep(0)
    release_first_started.set()

    results = await asyncio.gather(first, second, return_exceptions=True)
    successes = [result for result in results if isinstance(result, str)]
    failures = [result for result in results if isinstance(result, RuntimeError)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert str(failures[0]) == "active run"

    runner.gate.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["new_session", "resume", "compact", "set_permission"]
)
async def test_session_mutations_reject_submit_while_setup_is_reserved(
    tmp_path, operation
):
    runtime, runner = make_runtime(tmp_path)
    submit_started = asyncio.Event()
    release_submit = asyncio.Event()

    async def block_submit_setup(event):
        if event.type == "run_started":
            submit_started.set()
            await release_submit.wait()

    runtime.subscribe(block_submit_setup)
    submit = asyncio.create_task(runtime.submit("first"))
    await submit_started.wait()

    with pytest.raises(RuntimeError, match="active run"):
        if operation == "new_session":
            await runtime.new_session()
        elif operation == "resume":
            await runtime.resume(runtime.session_id)
        elif operation == "compact":
            await runtime.compact()
        else:
            await runtime.set_permission("full")

    release_submit.set()
    await submit
    runner.gate.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_cancelled_submit_closes_its_turn_and_clears_active_state(tmp_path):
    runtime, runner = make_runtime(tmp_path)
    submit_started = asyncio.Event()

    async def block_first_start(event):
        if event.type == "run_started" and not submit_started.is_set():
            submit_started.set()
            await asyncio.Event().wait()

    runtime.subscribe(block_first_start)
    submit = asyncio.create_task(runtime.submit("first"))
    await submit_started.wait()
    submit.cancel()

    with pytest.raises(asyncio.CancelledError):
        await submit

    assert runtime.status.status == "idle"
    assert runtime.status.run_id is None
    assert runtime.status.turn_id is None
    assert [record.type for record in runtime.store.records()] == [
        "turn_start",
        "user_message",
        "turn_end",
    ]
    assert runtime.store.records()[-1].payload["reason"] == "aborted"
    assert not runtime.store.has_interrupted_turn()

    await runtime.submit("second")
    runner.gate.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_forced_abort_persists_one_terminal_event_and_blocks_late_completion(
    tmp_path, monkeypatch
):
    release = asyncio.Event()

    class CancellationResistantRunner:
        event_sink = None
        permission_mode = "default"

        async def run_turn(self, prompt, *, run_id, turn_id, signal, usage=None):
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return TurnOutcome(reason="completed", final_text="late", steps=1)

    async def timeout_immediately(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(
        "coding_agent.runtime.runtime.asyncio.wait_for", timeout_immediately
    )
    runtime, _ = make_runtime(tmp_path, CancellationResistantRunner())
    events = []

    async def record(event):
        events.append(event)

    runtime.subscribe(record)
    run_id = await runtime.submit("first")
    await asyncio.sleep(0)

    await runtime.abort(run_id)

    assert [event.type for event in events if event.type == "run_finished"] == [
        "run_finished"
    ]
    finished = next(event for event in events if event.type == "run_finished")
    assert finished.payload["outcome"]["reason"] == "aborted"
    assert runtime.store.records()[-1].type == "turn_end"
    assert runtime.store.records()[-1].payload["reason"] == "aborted"

    release.set()
    await asyncio.sleep(0)
    assert [record.type for record in runtime.store.records()].count("turn_end") == 1
    assert [event.type for event in events if event.type == "run_finished"].count(
        "run_finished"
    ) == 1


@pytest.mark.asyncio
async def test_forced_abort_releases_runtime_for_a_follow_up_run(tmp_path, monkeypatch):
    release = asyncio.Event()

    class CancellationResistantRunner:
        event_sink = None
        permission_mode = "default"

        async def run_turn(self, prompt, *, run_id, turn_id, signal, usage=None):
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return TurnOutcome(reason="completed", final_text="late", steps=1)

    async def timeout_immediately(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(
        "coding_agent.runtime.runtime.asyncio.wait_for", timeout_immediately
    )
    runtime, _ = make_runtime(tmp_path, CancellationResistantRunner())
    run_id = await runtime.submit("first")
    await asyncio.sleep(0)

    await runtime.abort(run_id)

    assert runtime.status.status == "aborted"
    assert runtime._task is None
    follow_up = await runtime.submit("follow up")
    await runtime.abort(follow_up)


@pytest.mark.asyncio
async def test_submit_rejected_while_session_mutation_holds_reservation(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    loaded = asyncio.Event()
    release = asyncio.Event()

    async def block_session_loaded(event):
        if event.type == "session_loaded":
            loaded.set()
            await release.wait()

    runtime.subscribe(block_session_loaded)
    mutation = asyncio.create_task(runtime.new_session())
    await loaded.wait()

    with pytest.raises(RuntimeError, match="active run"):
        await runtime.submit("racing prompt")

    release.set()
    await mutation


@pytest.mark.asyncio
async def test_submit_is_nonblocking_and_rejects_busy_run(tmp_path):
    runtime, runner = make_runtime(tmp_path)
    run_id = await runtime.submit("first")
    assert run_id and runtime.status.status == "running"
    with pytest.raises(RuntimeError, match="active run"):
        await runtime.submit("second")
    runner.gate.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert runtime.status.status == "idle"


@pytest.mark.asyncio
async def test_resume_history_marks_failed_and_cancelled_tools(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    session_id = runtime.session_id
    assistant = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="failed", name="read_file"),
            ToolCall(id="cancelled", name="run_command"),
        ],
    )
    runtime.store.append_new("turn_start", {"turn_id": "t1"}, turn_id="t1")
    runtime.store.append_new(
        "user_message",
        {"message": Message(role="user", content="inspect")},
        turn_id="t1",
    )
    runtime.store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    runtime.store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="failed",
                tool_name="read_file",
                ok=False,
                content="",
                error="denied",
            )
        },
        turn_id="t1",
    )
    runtime.store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="cancelled",
                tool_name="run_command",
                ok=False,
                content="",
                error="cancelled",
            )
        },
        turn_id="t1",
    )
    runtime.store.append_new("turn_end", {"reason": "completed"}, turn_id="t1")

    events = []
    runtime.subscribe(events.append)
    await runtime.new_session()
    await runtime.resume(session_id)

    loaded = [item for item in events if item.type == "session_loaded"][-1]
    statuses = {
        item["message"]["tool_call_id"]: item["tool_status"]
        for item in loaded.payload["history"]
        if item["message"]["role"] == "tool"
    }
    assert statuses == {"failed": "error", "cancelled": "cancelled"}


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_resume_history_carries_command_and_metadata_for_tools(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    session_id = runtime.session_id
    call = ToolCall(id="c1", name="run_command", arguments={"command": "ls -la"})
    assistant = Message(role="assistant", tool_calls=[call])
    runtime.store.append_new("turn_start", {"turn_id": "t1"}, turn_id="t1")
    runtime.store.append_new(
        "user_message",
        {"message": Message(role="user", content="go")},
        turn_id="t1",
    )
    runtime.store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    runtime.store.append_new(
        "tool_call",
        {"tool_call": call, "source_assistant_record_id": "src"},
        turn_id="t1",
    )
    runtime.store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="c1",
                tool_name="run_command",
                ok=True,
                content="out\n",
                metadata={"exit_code": 0, "elapsed_seconds": 0.5, "truncated": False},
            )
        },
        turn_id="t1",
    )
    runtime.store.append_new("turn_end", {"reason": "completed"}, turn_id="t1")

    events = []
    runtime.subscribe(events.append)
    await runtime.new_session()
    await runtime.resume(session_id)

    loaded = [item for item in events if item.type == "session_loaded"][-1]
    tool_items = [
        item for item in loaded.payload["history"] if item["message"]["role"] == "tool"
    ]
    assert len(tool_items) == 1
    item = tool_items[0]
    assert item["command"] == "ls -la"
    assert item["metadata"]["exit_code"] == 0
    assert item["metadata"]["elapsed_seconds"] == 0.5
    assert item["tool_status"] == "success"


@pytest.mark.asyncio
async def test_resume_command_label_excludes_write_payload(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    session_id = runtime.session_id
    body = "SECRETBODY-" + "y" * 5000
    call = ToolCall(
        id="c1", name="write_file", arguments={"path": "a.txt", "content": body}
    )
    assistant = Message(role="assistant", tool_calls=[call])
    runtime.store.append_new("turn_start", {"turn_id": "t1"}, turn_id="t1")
    runtime.store.append_new(
        "user_message",
        {"message": Message(role="user", content="go")},
        turn_id="t1",
    )
    runtime.store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    runtime.store.append_new(
        "tool_call",
        {"tool_call": call, "source_assistant_record_id": "src"},
        turn_id="t1",
    )
    runtime.store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="c1",
                tool_name="write_file",
                ok=True,
                content="saved",
            )
        },
        turn_id="t1",
    )
    runtime.store.append_new("turn_end", {"reason": "completed"}, turn_id="t1")

    events = []
    runtime.subscribe(events.append)
    await runtime.new_session()
    await runtime.resume(session_id)

    loaded = [item for item in events if item.type == "session_loaded"][-1]
    tool_items = [
        item for item in loaded.payload["history"] if item["message"]["role"] == "tool"
    ]
    assert len(tool_items) == 1
    item = tool_items[0]
    assert item["command"] == "path=a.txt"
    assert "SECRETBODY" not in item["command"]
    assert "content" not in item["command"]


async def test_submit_publishes_user_message_and_turn_id(tmp_path):
    runtime, runner = make_runtime(tmp_path)
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    run_id = await runtime.submit("hello")
    runner.gate.set()
    await asyncio.sleep(0.05)

    started = next(event for event in events if event.type == "run_started")
    message = next(event for event in events if event.type == "user_message")
    assert started.run_id == run_id
    assert started.turn_id
    assert message.run_id == run_id
    assert message.turn_id == started.turn_id
    assert message.payload["message_id"]
    assert message.payload["text"] == "hello"


@pytest.mark.asyncio
async def test_subscribers_receive_events_and_unsubscribe(tmp_path):
    runtime, runner = make_runtime(tmp_path)
    events = []
    unsubscribe = runtime.subscribe(lambda event: events.append(event))
    # A synchronous callback is intentionally tolerated by the test adapter.
    unsubscribe()

    async def record(event):
        events.append(event)

    runtime.subscribe(record)
    await runtime.submit("hello")
    runner.gate.set()
    await asyncio.sleep(0.05)
    assert any(event.type == "run_started" for event in events)


@pytest.mark.asyncio
async def test_resume_session_loaded_projects_completed_history_and_excludes_open_turn(
    tmp_path,
):
    runtime, _ = make_runtime(tmp_path)
    session_id = runtime.session_id
    completed_turn = "completed"
    interrupted_turn = "interrupted"
    tool_call = ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"})
    runtime.store.append_new(
        "turn_start", {"turn_id": completed_turn}, turn_id=completed_turn
    )
    runtime.store.append_new(
        "user_message",
        {"message": Message(role="user", content="inspect a.py")},
        turn_id=completed_turn,
    )
    runtime.store.append_new(
        "assistant_message",
        {
            "message": Message(
                role="assistant", content="I will read it.", tool_calls=[tool_call]
            ),
            "complete": True,
        },
        turn_id=completed_turn,
    )
    runtime.store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="call-1", tool_name="read_file", ok=True, content="text"
            )
        },
        turn_id=completed_turn,
    )
    runtime.store.append_new(
        "assistant_message",
        {"message": Message(role="assistant", content="It is text."), "complete": True},
        turn_id=completed_turn,
    )
    runtime.store.append_new(
        "turn_end", {"reason": "completed"}, turn_id=completed_turn
    )
    runtime.store.append_new(
        "turn_start", {"turn_id": interrupted_turn}, turn_id=interrupted_turn
    )
    runtime.store.append_new(
        "user_message",
        {"message": Message(role="user", content="do not replay")},
        turn_id=interrupted_turn,
    )

    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.new_session()
    await runtime.resume(session_id)

    interrupted_records = [
        record
        for record in runtime.store.records()
        if record.type == "turn_end" and record.payload.get("reason") == "interrupted"
    ]
    assert len(interrupted_records) == 1

    loaded = [event for event in events if event.type == "session_loaded"][-1]
    assert loaded.payload["session_id"] == session_id
    assert loaded.payload["workspace"] == str(tmp_path)
    assert loaded.payload["model"] == "fake"
    assert loaded.payload["context_window"] == 1000
    assert all(isinstance(item["record_id"], str) for item in loaded.payload["history"])
    assert loaded.payload["history"] == [
        {
            "record_id": loaded.payload["history"][0]["record_id"],
            "turn_id": completed_turn,
            "seq": 1,
            "message": {
                "role": "user",
                "content": "inspect a.py",
                "tool_calls": [],
                "tool_call_id": None,
                "name": None,
            },
        },
        {
            "record_id": loaded.payload["history"][1]["record_id"],
            "turn_id": completed_turn,
            "seq": 2,
            "message": {
                "role": "assistant",
                "content": "I will read it.",
                "tool_calls": [
                    {"id": "call-1", "name": "read_file", "arguments": {"path": "a.py"}}
                ],
                "tool_call_id": None,
                "name": None,
            },
        },
        {
            "record_id": loaded.payload["history"][2]["record_id"],
            "turn_id": completed_turn,
            "seq": 3,
            "message": {
                "role": "tool",
                "content": "text",
                "tool_calls": [],
                "tool_call_id": "call-1",
                "name": "read_file",
            },
            "tool_status": "success",
        },
        {
            "record_id": loaded.payload["history"][3]["record_id"],
            "turn_id": completed_turn,
            "seq": 4,
            "message": {
                "role": "assistant",
                "content": "It is text.",
                "tool_calls": [],
                "tool_call_id": None,
                "name": None,
            },
        },
    ]


@pytest.mark.asyncio
async def test_new_session_loaded_has_empty_history_and_context_baseline(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.new_session()

    loaded = next(event for event in events if event.type == "session_loaded")
    assert loaded.payload["history"] == []
    assert loaded.payload["context_window"] == 1000


@pytest.mark.asyncio
async def test_new_and_resume_reset_permission(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    old_id = runtime.session_id
    await runtime.set_permission("full")
    new_id = await runtime.new_session()
    assert new_id != old_id and runtime.permission_mode == "workspace"
    await runtime.set_permission("full")
    await runtime.resume(old_id)
    assert runtime.permission_mode == "workspace" and runtime.session_id == old_id


@pytest.mark.asyncio
async def test_approval_cannot_be_resolved_twice():
    events = []

    async def publish(event):
        events.append(event)

    statuses = []
    broker = _ApprovalBroker(publish, statuses.append)
    request = ApprovalRequest(
        request_id="a1",
        run_id="r1",
        tool_call_id="c1",
        tool_name="write_file",
        risk_level="mutate_file",
    )
    pending = asyncio.create_task(broker.request(request))
    await asyncio.sleep(0)
    await broker.resolve("a1", "approve")
    with pytest.raises(RuntimeError, match="not pending"):
        await broker.resolve("a1", "deny")
    assert await pending == "approve"
    assert statuses == ["waiting_approval", "running"]
    assert [event.type for event in events].count("approval_resolved") == 1


@pytest.mark.asyncio
async def test_set_permission_publishes_previous_policy(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.set_permission("full")

    changed = next(event for event in events if event.type == "policy_changed")
    assert changed.payload == {"policy": "full", "previous_policy": "workspace"}


@pytest.mark.asyncio
async def test_compact_records_metadata(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    runtime.store.append_new(
        "user_message", {"message": Message(role="user", content="old")}, turn_id="old"
    )
    runtime.store.append_new(
        "assistant_message",
        {"message": Message(role="assistant", content="answer"), "complete": True},
        turn_id="old",
    )
    runtime.store.append_new(
        "user_message", {"message": Message(role="user", content="new")}, turn_id="new"
    )
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.compact()
    record = runtime.store.records()[-1]
    assert record.type == "compaction"
    assert record.payload["forced"] is True
    assert {
        "strategy",
        "removed_turn_ids",
        "retained_turn_ids",
        "tokens_before",
        "tokens_after",
    } <= record.payload.keys()
    assert any(event.type == "context_updated" for event in events)


@pytest.mark.asyncio
async def test_runner_events_are_bridged_and_status_updates(tmp_path):
    runner = EventRunner()
    runtime, _ = make_runtime(tmp_path, runner)
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.submit("hello")
    await asyncio.sleep(0.05)
    assert any(event.type == "context_updated" for event in events)
    assert runtime.status.context_used == 7
    assert runtime.status.context_window == 100
    assert runtime.status.context_estimated is True


@pytest.mark.asyncio
async def test_permission_updates_runner_and_persists_record(tmp_path):
    runtime, runner = make_runtime(tmp_path)
    await runtime.set_permission("full")
    assert runner.permission_mode == "full"
    assert runtime.store.records()[-1].type == "policy_changed"
