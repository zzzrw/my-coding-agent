import asyncio

import pytest

from coding_agent.tools.registry import ToolContext
from coding_agent.tools.shell import make_run_command_tool


@pytest.mark.asyncio
async def test_run_command_uses_workspace_cwd_and_exit_metadata(tmp_path):
    result = await make_run_command_tool().execute(
        {"command": "pwd", "timeout_seconds": 5},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert str(tmp_path) in result.content
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_run_command_timeout_returns_bounded_error(tmp_path):
    result = await make_run_command_tool().execute(
        {"command": "sleep 10", "timeout_seconds": 0.1},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_run_command_cancellation_kills_process_group(tmp_path):
    cancel = asyncio.Event()
    task = asyncio.create_task(
        make_run_command_tool().execute(
            {"command": "sleep 30 & wait", "timeout_seconds": 30},
            context=ToolContext(workspace=tmp_path, permission_mode="full"),
            signal=cancel,
        )
    )
    await asyncio.sleep(0.1)
    cancel.set()
    result = await asyncio.wait_for(task, timeout=3)
    assert result.ok is False
    assert result.error == "cancelled"
