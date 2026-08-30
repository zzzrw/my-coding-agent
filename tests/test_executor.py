import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import ToolCall
from coding_agent.tools.executor import MAX_TOOL_OUTPUT_CHARS, ToolExecutor
from coding_agent.tools.models import ToolResult, ToolSchema
from coding_agent.tools.registry import ToolRegistry


class Args(BaseModel):
    path: str | None = None
    content: str | None = None


class RecordingTool:
    def __init__(self, name="read_file", *, content="ok", delay=None, fail=False):
        self.schema = ToolSchema(
            name=name,
            description=name,
            parameters=Args.model_json_schema(),
            risk_level="mutate_file" if name == "write_file" else "read",
        )
        self.args_model = Args
        self.content = content
        self.delay = delay
        self.fail = fail
        self.calls = []

    async def execute(self, arguments, *, context, signal):
        self.calls.append((arguments, context))
        if self.fail:
            raise RuntimeError("boom")
        if self.delay:
            await asyncio.sleep(self.delay)
        return ToolResult(
            tool_call_id="",
            tool_name=self.schema.name,
            ok=True,
            content=self.content,
        )


class Broker:
    def __init__(self, answer="approve"):
        self.answer = answer
        self.requests = []

    async def request(self, request):
        self.requests.append(request)
        return self.answer

    def cancel_all(self):
        return None


def make_executor(tool, answer="approve", timeout=120):
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(
        registry,
        DefaultApprovalPolicy(),
        Broker(answer),
        default_timeout_seconds=timeout,
    )


@pytest.mark.asyncio
async def test_tool_exception_becomes_error_result():
    result = await make_executor(RecordingTool(fail=True)).execute(
        ToolCall(id="c1", name="read_file"),
        run_id="r1",
        workspace=Path("."),
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert result.ok is False and "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_denied_approval_does_not_call_tool():
    tool = RecordingTool("write_file")
    result = await make_executor(tool, "deny").execute(
        ToolCall(id="c1", name="write_file", arguments={"path": "a", "content": "x"}),
        run_id="r1",
        workspace=Path("."),
        permission_mode="default",
        signal=asyncio.Event(),
    )
    assert result.ok is False and tool.calls == []


@pytest.mark.asyncio
async def test_approved_outside_path_is_one_call_only(tmp_path):
    tool = RecordingTool("write_file")
    executor = make_executor(tool)
    result = await executor.execute(
        ToolCall(
            id="c1", name="write_file", arguments={"path": "../outside", "content": "x"}
        ),
        run_id="r1",
        workspace=tmp_path,
        permission_mode="workspace",
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert tool.calls[0][1].allow_outside_once is True


@pytest.mark.asyncio
async def test_output_is_bounded():
    result = await make_executor(
        RecordingTool(content="x" * (MAX_TOOL_OUTPUT_CHARS + 1))
    ).execute(
        ToolCall(id="c1", name="read_file"),
        run_id="r1",
        workspace=Path("."),
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert len(result.content) == MAX_TOOL_OUTPUT_CHARS
    assert result.metadata["truncated"] is True


@pytest.mark.asyncio
async def test_abort_cancels_a_slow_tool():
    signal = asyncio.Event()
    task = asyncio.create_task(
        make_executor(RecordingTool(delay=10)).execute(
            ToolCall(id="c1", name="read_file"),
            run_id="r1",
            workspace=Path("."),
            permission_mode="full",
            signal=signal,
        )
    )
    await asyncio.sleep(0)
    signal.set()
    result = await task
    assert result.metadata["cancelled"] is True


@pytest.mark.asyncio
async def test_timeout_and_unknown_tool_are_structured():
    executor = make_executor(RecordingTool(delay=1), timeout=0.01)
    timed_out = await executor.execute(
        ToolCall(id="c1", name="read_file"),
        run_id="r1",
        workspace=Path("."),
        permission_mode="full",
        signal=asyncio.Event(),
    )
    unknown = await executor.execute(
        ToolCall(id="c2", name="missing"),
        run_id="r1",
        workspace=Path("."),
        permission_mode="full",
        signal=asyncio.Event(),
    )
    assert timed_out.metadata["timed_out"] is True
    assert unknown.ok is False and "unknown" in (unknown.error or "")
