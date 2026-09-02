"""Context/token meter calibration tests.

The reported ``used`` must track real provider usage (the last full request's
input+output+cache total) rather than a small chars/4 estimate, and the UI must
show only ``used/window``. These tests drive the meter through the runner and
runtime with deterministic fake provider usage.
"""

import asyncio

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import LLMEvent, Message, TurnOutcome, Usage
from coding_agent.runtime.runner import AgentRunner
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult
from coding_agent.tools.registry import ToolRegistry

_SYSTEM = Message(role="system", content="You are a coding agent.")


def _store(tmp_path, *, context_window: int = 1000) -> SessionStore:
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="fake",
        context_window=context_window,
    )
    store.append_new("turn_start", {"turn_id": "t1"}, run_id="r1", turn_id="t1")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="inspect")},
        run_id="r1",
        turn_id="t1",
    )
    return store


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, call, **kwargs):
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=True,
            content="ok",
        )


def _runner(tmp_path, provider, *, context_window: int = 1000):
    store = _store(tmp_path, context_window=context_window)
    events = []

    async def sink(event):
        events.append(event)

    runner = AgentRunner(
        provider=provider,
        registry=ToolRegistry(),
        executor=_RecordingExecutor(),
        context_policy=TruncatePolicy(),
        store=store,
        event_sink=sink,
        system_prompt=_SYSTEM,
        model="fake",
        context_window=context_window,
        permission_mode="full",
    )
    return runner, store, events


def _runtime(tmp_path, runner_factory):
    store = _store(tmp_path)
    runtime = AgentRuntime(
        store=store,
        runner_factory=runner_factory,
        context_policy_factory=lambda: TruncatePolicy(),
        approval_policy=DefaultApprovalPolicy(),
        system_prompt=_SYSTEM,
        model="fake",
    )
    return runtime


async def _wait_idle(runtime, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while runtime.status.status not in {"idle", "aborted", "error"}:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError("run did not settle")
        await asyncio.sleep(0.005)


class _TextUsageProvider:
    """Yields one plain text response carrying a deterministic usage total."""

    def __init__(self, total: int):
        self.total = total
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append(messages)
        yield LLMEvent(type="text_delta", text="done")
        yield LLMEvent(
            type="response_end",
            finish_reason="stop",
            usage=Usage(input_tokens=60, output_tokens=40, total_tokens=self.total),
        )


@pytest.mark.asyncio
async def test_one_response_turn_reemits_provider_total_without_second_request(
    tmp_path,
):
    """A single-response turn must update the meter to the provider total.

    The runner re-emits ``context_updated`` after ``assistant_finished``, so the
    meter reflects real usage without needing a second request to observe it.
    """
    provider = _TextUsageProvider(total=512)
    runner, _, events = _runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.usage is not None and outcome.usage.total_tokens == 512
    ctx_updates = [event for event in events if event.type == "context_updated"]
    assert ctx_updates, "expected at least one context_updated event"
    last = ctx_updates[-1]
    assert last.payload["used_tokens"] == 512
    assert last.payload["estimated"] is False
    assert len(provider.requests) == 1


class _OutcomeRunner:
    """Fake runner that returns a fixed outcome usage and emits nothing."""

    def __init__(self, usage: Usage | None):
        self.event_sink = None
        self.permission_mode = "default"
        self.usage = usage

    async def run_turn(self, prompt, *, run_id, turn_id, signal, usage=None):
        return TurnOutcome(
            reason="completed", final_text="done", steps=1, usage=self.usage
        )


@pytest.mark.asyncio
async def test_runtime_status_derives_used_from_outcome_usage(tmp_path):
    """The runtime must derive context_used from the outcome's usage.

    Previously the status preserved whatever the runner last reported, so a
    turn that produced usage but never surfaced a matching context_updated left
    the meter stale.
    """
    runtime = _runtime(
        tmp_path,
        lambda *_: _OutcomeRunner(
            Usage(input_tokens=600, output_tokens=40, total_tokens=640)
        ),
    )
    await runtime.submit("hello")
    await _wait_idle(runtime)
    assert runtime.status.context_used == 640
    assert runtime.status.context_estimated is False
    assert runtime.status.usage is not None
    assert runtime.status.usage.total_tokens == 640


class _SeedRecorderRunner:
    """Records the usage kwarg each run_turn receives and feeds usage forward."""

    def __init__(self):
        self.event_sink = None
        self.permission_mode = "default"
        self.seen: list[Usage | None] = []

    async def run_turn(self, prompt, *, run_id, turn_id, signal, usage=None):
        self.seen.append(usage)
        total = (usage.total_tokens if usage is not None else 0) + 7
        return TurnOutcome(
            reason="completed",
            final_text="ok",
            steps=1,
            usage=Usage(total_tokens=total),
        )


@pytest.mark.asyncio
async def test_last_provider_usage_seeds_the_next_turn(tmp_path):
    """Turn 2's first step must start from turn 1's full-request total.

    The runtime persists the last usage between turns and passes it into
    ``run_turn``, so a fresh turn does not drop back to a pure estimate before
    its first provider response.
    """
    runner = _SeedRecorderRunner()
    runtime = _runtime(tmp_path, lambda *_: runner)
    await runtime.submit("one")
    await _wait_idle(runtime)
    assert runner.seen[0] is None  # nothing measured yet on the first turn
    await runtime.submit("two")
    await _wait_idle(runtime)
    assert runner.seen[1] is not None
    assert runner.seen[1].total_tokens == 7
