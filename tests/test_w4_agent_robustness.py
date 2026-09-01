"""W4 — agent robustness: retry, idle timeout, loop detection, summary compression.

Task 1 covers the tool retry loop with an eligibility predicate.
"""

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import ToolCall
from coding_agent.tools.executor import ToolExecutor, _retryable
from coding_agent.tools.models import ToolResult, ToolSchema
from coding_agent.tools.registry import ToolRegistry


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
