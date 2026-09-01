"""W4 — agent robustness: retry, idle timeout, loop detection, summary compression.

Task 1 covers the tool retry loop with an eligibility predicate.
Task 2 covers the provider idle watchdog/heartbeat and progress-loop detection.
"""

import asyncio
import json

import pytest
from fakes import BlockingFakeProvider
from pydantic import BaseModel, ConfigDict

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import LLMEvent, Message, ToolCall
from coding_agent.runtime.runner import AgentRunner
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore
from coding_agent.tools.executor import ToolExecutor, _retryable
from coding_agent.tools.models import ToolResult, ToolSchema
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import TranscriptItem, initial_state
from coding_agent.tui.widgets import _tool_footer


class _NoopBroker:
    """Always approves; never asked when permission_mode="full"."""

    async def request(self, request):
        return "approve"

    def cancel_all(self):
        return None


class _FlakyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class _FlakyTool:
    """Fails the first ``failures`` executions with ``error``, then succeeds."""

    def __init__(self, failures: int, error: str = "tool timed out"):
        self.schema = ToolSchema(
            name="flaky",
            description="d",
            parameters={"type": "object"},
            risk_level="read",
        )
        self.args_model = _FlakyArgs
        self.failures = failures
        self.error = error
        self.calls = 0

    async def execute(self, arguments, *, context, signal):
        self.calls += 1
        if self.calls <= self.failures:
            return ToolResult(
                tool_call_id="c",
                tool_name="flaky",
                ok=False,
                content="",
                error=self.error,
            )
        return ToolResult(tool_call_id="c", tool_name="flaky", ok=True, content="ok")


def _make_executor(tool, *, max_retries=2, retry_backoff_seconds=0.01) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(
        registry,
        DefaultApprovalPolicy(),
        _NoopBroker(),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def test_retryable_predicate():
    assert _retryable("tool timed out")
    assert _retryable("connection reset")
    assert not _retryable("approval denied")
    assert not _retryable("approval cancelled")
    assert not _retryable("invalid tool arguments")
    assert not _retryable("old_text must match exactly once")


@pytest.mark.asyncio
async def test_executor_retries_transient_error_then_succeeds(tmp_path):
    tool = _FlakyTool(failures=2)
    executor = _make_executor(tool)
    result = await executor.execute(
        ToolCall(id="c", name="flaky", arguments={}),
        run_id="r",
        workspace=tmp_path,
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert result.ok
    assert tool.calls == 3
    assert result.metadata.get("retries") == 2


@pytest.mark.asyncio
async def test_executor_respects_max_retries_on_persistent_failure(tmp_path):
    tool = _FlakyTool(failures=99)
    executor = _make_executor(tool, max_retries=2)
    result = await executor.execute(
        ToolCall(id="c", name="flaky", arguments={}),
        run_id="r",
        workspace=tmp_path,
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert not result.ok
    assert tool.calls == 3  # initial attempt + 2 retries
    assert result.metadata.get("retries") == 2


@pytest.mark.asyncio
async def test_executor_does_not_retry_non_retryable_error(tmp_path):
    tool = _FlakyTool(failures=99, error="approval denied")
    executor = _make_executor(tool, max_retries=2)
    result = await executor.execute(
        ToolCall(id="c", name="flaky", arguments={}),
        run_id="r",
        workspace=tmp_path,
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert not result.ok
    assert tool.calls == 1  # never retried
    assert "retries" not in result.metadata


# --------------------------------------------------------------------------
# Task 2: provider idle watchdog + heartbeat + progress-loop detection
# --------------------------------------------------------------------------


class _OkExecutor:
    """Duck-typed executor that reports success for every call."""

    def __init__(self):
        self.calls = []

    async def execute(self, call, **kwargs):
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id, tool_name=call.name, ok=True, content="ok"
        )


class _RepeatingProvider:
    """Re-emits the SAME tool call (name + arguments) each step with a fresh
    call id, so the session store accepts consecutive tool results."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.calls = 0

    async def stream(self, messages, tools, *, model, signal):
        self.calls += 1
        call_id = f"call-{self.calls}"
        yield LLMEvent(
            type="tool_call_start", tool_call_id=call_id, tool_name=self.tool_name
        )
        yield LLMEvent(
            type="tool_call_delta", tool_call_id=call_id, arguments_delta="{}"
        )
        yield LLMEvent(type="response_end", finish_reason="tool_calls")


class _AlternatingProvider:
    """Emits a write_file call that alternates its ``path`` argument each step."""

    def __init__(self):
        self.n = 0

    async def stream(self, messages, tools, *, model, signal):
        self.n += 1
        call_id = f"call-{self.n}"
        path = "a.txt" if self.n % 2 == 0 else "b.txt"
        yield LLMEvent(
            type="tool_call_start", tool_call_id=call_id, tool_name="write_file"
        )
        yield LLMEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=json.dumps({"path": path}),
        )
        yield LLMEvent(type="response_end", finish_reason="tool_calls")


def _make_runner(
    tmp_path,
    provider,
    executor,
    *,
    provider_idle_timeout_seconds=90.0,
    max_steps=20,
):
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="fake",
        context_window=1000,
    )
    store.append_new("turn_start", {"turn_id": "t"}, run_id="r", turn_id="t")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="go")},
        run_id="r",
        turn_id="t",
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
        model="fake",
        context_window=1000,
        permission_mode="full",
        max_steps=max_steps,
        provider_idle_timeout_seconds=provider_idle_timeout_seconds,
    )
    return runner, events


@pytest.mark.asyncio
async def test_provider_idle_timeout(tmp_path):
    # A provider that never yields should trip the idle watchdog and return
    # provider_timeout, while heartbeats flowed during the stall.
    provider = BlockingFakeProvider()
    runner, events = _make_runner(
        tmp_path, provider, _OkExecutor(), provider_idle_timeout_seconds=0.05
    )
    outcome = await runner.run_turn(
        "hi", run_id="r", turn_id="t", signal=asyncio.Event()
    )
    assert outcome.reason == "provider_timeout"
    assert any(event.type == "heartbeat" for event in events)
    heartbeat = next(event for event in events if event.type == "heartbeat")
    assert isinstance(heartbeat.payload.get("elapsed_seconds"), float)


@pytest.mark.asyncio
async def test_loop_detection(tmp_path):
    # A provider that re-emits the SAME tool call 3 times -> progress_loop.
    runner, events = _make_runner(
        tmp_path, _RepeatingProvider("write_file"), _OkExecutor()
    )
    outcome = await runner.run_turn(
        "go", run_id="r", turn_id="t", signal=asyncio.Event()
    )
    assert outcome.reason == "progress_loop"
    assert any(
        event.type == "notice"
        and "repeated tool call" in (event.payload.get("message") or "")
        for event in events
    )


@pytest.mark.asyncio
async def test_loop_detection_resets_on_differing_arguments(tmp_path):
    # Alternating arguments across 3 waves must NOT trip the loop detector;
    # the turn runs to max_steps instead.
    provider = _AlternatingProvider()
    runner, _ = _make_runner(tmp_path, provider, _OkExecutor(), max_steps=3)
    outcome = await runner.run_turn(
        "go", run_id="r", turn_id="t", signal=asyncio.Event()
    )
    assert outcome.reason == "max_steps"


# --------------------------------------------------------------------------
# Task 3: summary compression
# --------------------------------------------------------------------------


class _StubRunner:
    """Minimal runner stub; ``provider`` is None unless a provider is given."""

    def __init__(self, provider=None):
        self.provider = provider


class _SummarizingProvider:
    """Emits a single text_delta summary when asked to summarize."""

    def __init__(self, text="the generated summary"):
        self.text = text
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append((list(messages), list(tools), model))
        yield LLMEvent(type="text_delta", text=self.text)


def _make_runtime(tmp_path, *, summarizer=None, runner=None):
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="fake",
        context_window=1000,
    )
    store.append_new(
        "user_message", {"message": Message(role="user", content="old")}, turn_id="old"
    )
    store.append_new(
        "assistant_message",
        {"message": Message(role="assistant", content="old answer"), "complete": True},
        turn_id="old",
    )
    store.append_new(
        "user_message", {"message": Message(role="user", content="new")}, turn_id="new"
    )
    stub = runner if runner is not None else _StubRunner()
    runtime = AgentRuntime(
        store=store,
        runner_factory=lambda *_: stub,
        context_policy_factory=lambda: TruncatePolicy(1000),
        approval_policy=DefaultApprovalPolicy(),
        system_prompt=Message(role="system", content="system"),
        model="fake",
        summarizer=summarizer,
    )
    return runtime, store


@pytest.mark.asyncio
async def test_compact_stores_summary_and_project_prepends(tmp_path):
    async def summarizer(messages):
        return "the summary"

    runtime, store = _make_runtime(tmp_path, summarizer=summarizer)
    await runtime.compact()
    records = store.records()
    compaction = [r for r in records if r.type == "compaction"]
    assert compaction and compaction[-1].payload.get("summary") == "the summary"
    messages = store.project_messages(include_open_turn=False)
    assert messages and messages[0].message.role == "system"
    assert "the summary" in (messages[0].message.content or "")
    assert messages[0].turn_id is None
    assert messages[0].record_id.startswith("summary-")


@pytest.mark.asyncio
async def test_compact_falls_back_without_summarizer(tmp_path):
    # No injected summarizer and no provider on the runner -> silent truncation.
    runtime, store = _make_runtime(tmp_path)
    await runtime.compact()
    compaction = [r for r in store.records() if r.type == "compaction"]
    assert compaction and "summary" not in compaction[-1].payload
    messages = store.project_messages()
    assert not messages or messages[0].message.role != "system"


@pytest.mark.asyncio
async def test_compact_default_summarizer_uses_provider(tmp_path):
    # No injected summarizer, but the runner exposes a provider: the default
    # summarizer streams the dropped messages with an empty tool list.
    provider = _SummarizingProvider("the generated summary")
    runtime, store = _make_runtime(tmp_path, runner=_StubRunner(provider=provider))
    await runtime.compact()
    compaction = [r for r in store.records() if r.type == "compaction"]
    assert compaction and (
        compaction[-1].payload.get("summary") == "the generated summary"
    )
    assert provider.requests and provider.requests[0][1] == []


@pytest.mark.asyncio
async def test_compact_omits_summary_when_summarizer_fails(tmp_path):
    async def failing_summarizer(messages):
        raise RuntimeError("summarization failed")

    runtime, store = _make_runtime(tmp_path, summarizer=failing_summarizer)
    await runtime.compact()
    compaction = [r for r in store.records() if r.type == "compaction"]
    assert compaction and "summary" not in compaction[-1].payload


# --------------------------------------------------------------------------
# Task 4: TUI surfacing (retry footer + notices)
# --------------------------------------------------------------------------


def test_tool_footer_shows_retry_count():
    item = TranscriptItem(
        kind="tool",
        item_id="c1",
        tool_call_id="c1",
        tool_name="flaky",
        text="ok",
        tool_status="success",
        retries=2,
    )
    footer = _tool_footer(item)
    assert footer is not None
    assert "· retried 2×" in footer

    # No retries -> no retried marker in the footer.
    plain = item.model_copy(update={"retries": None})
    assert "retried" not in (_tool_footer(plain) or "")


def test_tool_footer_omits_retry_marker_when_absent():
    item = TranscriptItem(
        kind="tool",
        item_id="c1",
        tool_call_id="c1",
        tool_name="flaky",
        text="ok",
        tool_status="success",
    )
    footer = _tool_footer(item)
    assert footer is None


def test_heartbeat_is_noop_in_reducer():
    state = initial_state(".", "test")
    before = state
    state = reduce(
        state,
        RuntimeEvent(type="heartbeat", payload={"elapsed_seconds": 5.0}),
    )
    assert state == before


def test_reducer_copies_retries_onto_tool_finished():
    state = initial_state(".", "test")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_finished",
            payload={
                "tool_call_id": "c1",
                "tool_name": "flaky",
                "ok": False,
                "error": "tool timed out",
                "content": "",
                "metadata": {"retries": 2, "exit_code": 1},
            },
        ),
    )
    row = next(r for r in state.transcript if r.tool_call_id == "c1")
    assert row.retries == 2


def test_reducer_retries_defaults_to_none_when_metadata_missing():
    state = initial_state(".", "test")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_finished",
            payload={"tool_call_id": "c1", "ok": True, "content": "ok"},
        ),
    )
    row = next(r for r in state.transcript if r.tool_call_id == "c1")
    assert row.retries is None
