"""W1 real-time feedback tests: streamed tool output + statusline spinner/elapsed.

Task 1 coverage lives here: the tool output sink threaded through
``ToolContext`` -> ``_ShellTool`` -> ``ToolExecutor``.
"""

import asyncio

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import ToolCall
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.registry import ToolContext, ToolRegistry
from coding_agent.tools.shell import make_run_command_tool


class _NoopBroker:
    async def request(self, request):
        return "approve"

    def cancel_all(self):
        pass


async def test_tool_context_carries_on_output():
    collected: list[str] = []

    async def sink(text: str) -> None:
        collected.append(text)

    context = ToolContext(workspace=".", permission_mode="full", on_output=sink)
    assert context.on_output is not None
    await context.on_output("hello")
    assert collected == ["hello"]


async def test_shell_tool_streams_output_chunks(tmp_path):
    collected: list[str] = []

    async def sink(text: str) -> None:
        collected.append(text)

    tool = make_run_command_tool()
    context = ToolContext(
        workspace=str(tmp_path), permission_mode="full", on_output=sink
    )
    signal = asyncio.Event()
    result = await tool.execute(
        {"command": "printf 'one\\ntwo\\n'"}, context=context, signal=signal
    )
    assert result.ok
    assert result.content == "one\ntwo\n"
    assert collected and "".join(collected) == "one\ntwo\n"


async def test_executor_forwards_output_sink(tmp_path):
    collected: list[str] = []

    async def sink(text: str) -> None:
        collected.append(text)

    registry = ToolRegistry()
    registry.register(make_run_command_tool())
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), _NoopBroker())
    call = ToolCall(id="c1", name="run_command", arguments={"command": "printf hi"})
    signal = asyncio.Event()
    result = await executor.execute(
        call,
        run_id="r",
        workspace=tmp_path,
        permission_mode="full",
        signal=signal,
        output_sink=sink,
    )
    assert result.ok
    assert "".join(collected) == "hi"
