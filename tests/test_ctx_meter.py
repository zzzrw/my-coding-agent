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


class _TwoStepUsageProvider:
    """A tool-call step followed by a final text step, each with usage."""

    def __init__(self):
        self.requests = []
        self.step1_total = 60
        self.step2_total = 5000
        self.responses = [
            [
                LLMEvent(
                    type="tool_call_start", tool_call_id="c1", tool_name="read_file"
                ),
                LLMEvent(
                    type="tool_call_delta",
                    tool_call_id="c1",
                    arguments_delta='{"path": "main.py"}',
                ),
                LLMEvent(
                    type="tool_call_end",
                    finish_reason="tool_calls",
                    usage=Usage(
                        input_tokens=10, output_tokens=50, total_tokens=self.step1_total
                    ),
                ),
            ],
            [
                LLMEvent(type="text_delta", text="done"),
                LLMEvent(
                    type="response_end",
                    finish_reason="stop",
                    usage=Usage(
                        input_tokens=4000,
                        output_tokens=1000,
                        total_tokens=self.step2_total,
                    ),
                ),
            ],
        ]

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append(messages)
        for event in self.responses.pop(0):
            yield event


@pytest.mark.asyncio
async def test_used_grows_monotonically_across_a_tool_call_turn(tmp_path):
    """The meter grows after every step, including the appended tool result.

    Between the step-1 response (tool call) and the step-2 request, the tool
    result is appended. The step-2 opening estimate must sit above the step-1
    provider total (the appended result is counted), and the sequence of
    ``context_updated`` used values must never decrease.
    """
    provider = _TwoStepUsageProvider()
    runner, _, events = _runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert outcome.usage is not None and outcome.usage.total_tokens == 5000

    used_values = [
        event.payload["used_tokens"]
        for event in events
        if event.type == "context_updated"
    ]
    assert used_values == sorted(used_values), "meter must never decrease"
    # Step-1 assistant_finished reported the provider total (60); the step-2
    # opening emission adds the appended tool result, so it exceeds 60 without
    # waiting for the step-2 request to complete.
    assert used_values[-2] > provider.step1_total
    assert used_values[-1] == provider.step2_total


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


class _ZeroTotalUsageProvider:
    """Streaming endpoints may omit ``total_tokens`` yet report prompt+completion."""

    def __init__(self):
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append(messages)
        yield LLMEvent(type="text_delta", text="done")
        yield LLMEvent(
            type="response_end",
            finish_reason="stop",
            usage=Usage(input_tokens=1000, output_tokens=200, total_tokens=0),
        )


@pytest.mark.asyncio
async def test_reemit_uses_measured_total_when_endpoint_omits_total_tokens(
    tmp_path,
):
    """A zero ``total_tokens`` must not drop the meter to zero after a response.

    Some OpenAI-compatible endpoints omit ``total_tokens`` in streaming usage;
    the measured total is then input+output, and the post-response re-emit must
    report it rather than emit ``used=0`` (which the runtime would then copy
    into its final status).
    """
    provider = _ZeroTotalUsageProvider()
    runner, _, events = _runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.usage is not None and outcome.usage.total_tokens == 0
    ctx_updates = [event for event in events if event.type == "context_updated"]
    last = ctx_updates[-1]
    assert last.payload["used_tokens"] == 1200
    assert last.payload["used_tokens"] != 0
    assert last.payload["estimated"] is False
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_runtime_status_uses_sum_when_outcome_total_is_zero(tmp_path):
    """The final status must derive ``context_used`` from input+output when the
    outcome's ``total_tokens`` is zero (omitted by the endpoint)."""
    runtime = _runtime(
        tmp_path,
        lambda *_: _OutcomeRunner(
            Usage(input_tokens=600, output_tokens=40, total_tokens=0)
        ),
    )
    await runtime.submit("hello")
    await _wait_idle(runtime)
    assert runtime.status.context_used == 640
    assert runtime.status.context_used != 0
    assert runtime.status.context_estimated is False


class _TurnGapUsageProvider:
    """Turn 1 ends with a text-heavy response; turn 2 measures a fresh total.

    The turn-2 opening estimate must count only the new turn's prompt, never
    re-estimate turn 1's output (already inside turn 1's total), so the meter
    never overshoots above turn 2's measured total and then drops.
    """

    def __init__(self):
        self.responses = [
            [
                LLMEvent(type="text_delta", text="z" * 800),
                LLMEvent(
                    type="response_end",
                    finish_reason="stop",
                    usage=Usage(input_tokens=5, output_tokens=395, total_tokens=400),
                ),
            ],
            [
                LLMEvent(type="text_delta", text="done"),
                LLMEvent(
                    type="response_end",
                    finish_reason="stop",
                    usage=Usage(input_tokens=300, output_tokens=200, total_tokens=500),
                ),
            ],
        ]

    async def stream(self, messages, tools, *, model, signal):
        for event in self.responses.pop(0):
            yield event


@pytest.mark.asyncio
async def test_turn_boundary_meter_never_recounts_previous_response_output(
    tmp_path,
):
    """The meter must not rise above the next measured total and then drop.

    Across two turns the ``context_updated`` sequence must be monotonic: turn
    2's opening value sits between turn 1's final total and turn 2's measured
    total (turn 1's large output is already inside turn 1's total, so it is not
    re-estimated on turn 2's first step).
    """
    provider = _TurnGapUsageProvider()

    def factory(store, policy, broker):
        return AgentRunner(
            provider=provider,
            registry=ToolRegistry(),
            executor=_RecordingExecutor(),
            context_policy=policy,
            store=store,
            event_sink=broker,
            system_prompt=_SYSTEM,
            model="fake",
            context_window=1000,
            permission_mode="full",
        )

    runtime = _runtime(tmp_path, factory)
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.submit("one")
    await _wait_idle(runtime)
    await runtime.submit("two")
    await _wait_idle(runtime)

    used = [
        event.payload["used_tokens"]
        for event in events
        if event.type == "context_updated"
    ]
    assert used == sorted(used), "meter must never decrease across turns"
    assert 400 in used  # turn 1's final re-emit reported its provider total
    assert used[-1] == 500  # turn 2's measured total is the last word
    # Turn 2's opening adds only the appended prompt to turn 1's total, so it
    # must not overshoot the total turn 2's own request will measure.
    assert 400 < used[-2] < 500
