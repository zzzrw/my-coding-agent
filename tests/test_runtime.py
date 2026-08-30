import asyncio

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import Message, TurnOutcome
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore


class BlockingRunner:
    def __init__(self, gate=None):
        self.gate = gate or asyncio.Event()

    async def run_turn(self, prompt, *, run_id, turn_id, signal):
        await self.gate.wait()
        return TurnOutcome(reason="completed", final_text=prompt, steps=1)


def make_runtime(tmp_path, runner=None):
    store = SessionStore.create(tmp_path / "sessions", workspace=str(tmp_path), model="fake", context_window=1000)
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
async def test_new_and_resume_reset_permission(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    old_id = runtime.session_id
    await runtime.set_permission("full")
    new_id = await runtime.new_session()
    assert new_id != old_id and runtime.permission_mode == "default"
    await runtime.set_permission("full")
    await runtime.resume(old_id)
    assert runtime.permission_mode == "default" and runtime.session_id == old_id
