import asyncio

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import Message, TurnOutcome
from coding_agent.runtime.runtime import AgentRuntime, _ApprovalBroker
from coding_agent.session.models import ApprovalRequest
from coding_agent.session.store import SessionStore


class BlockingRunner:
    def __init__(self, gate=None):
        self.gate = gate or asyncio.Event()

    async def run_turn(self, prompt, *, run_id, turn_id, signal):
        await self.gate.wait()
        return TurnOutcome(reason="completed", final_text=prompt, steps=1)


class EventRunner:
    def __init__(self):
        self.event_sink = None
        self.permission_mode = "default"

    async def run_turn(self, prompt, *, run_id, turn_id, signal):
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
async def test_session_loaded_publishes_restored_workspace_and_model(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    session_id = runtime.session_id
    await runtime.new_session()
    await runtime.resume(session_id)

    loaded = [event for event in events if event.type == "session_loaded"][-1]
    assert loaded.payload["session_id"] == session_id
    assert loaded.payload["workspace"] == str(tmp_path)
    assert loaded.payload["model"] == "fake"


@pytest.mark.asyncio
async def test_new_and_resume_reset_permission(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    old_id = runtime.session_id
    await runtime.set_permission("full")
    new_id = await runtime.new_session()
    assert new_id != old_id and runtime.permission_mode == "default"
    await runtime.set_permission("full")
    await runtime.resume(old_id)
    assert runtime.permission_mode == "default" and runtime.session_id == old_id


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
    assert changed.payload == {"policy": "full", "previous_policy": "default"}


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
