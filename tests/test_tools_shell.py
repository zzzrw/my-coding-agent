import asyncio

import pytest

from coding_agent.tools.registry import ToolContext
from coding_agent.tools.shell import MAX_COMMAND_OUTPUT_BYTES, make_run_command_tool


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
async def test_run_command_default_timeout_is_120_seconds(tmp_path):
    assert (
        make_run_command_tool().args_model.model_fields["timeout_seconds"].default
        == 120
    )


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


@pytest.mark.asyncio
async def test_background_descendant_survives_successful_completion(tmp_path):
    # An intentionally backgrounded process (e.g. a dev server) must outlive
    # the tool returning, so later commands can still reach it.
    marker = tmp_path / "late-marker"
    result = await make_run_command_tool().execute(
        {"command": "{ sleep 0.3; touch late-marker; } >/dev/null 2>&1 &"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    await asyncio.sleep(0.5)
    assert marker.exists()


@pytest.mark.asyncio
async def test_timeout_still_kills_background_group(tmp_path):
    # Timeout/cancel remain destructive: they tear down the whole process group
    # so a misbehaving command cannot leave a backgrounded worker behind.
    marker = tmp_path / "leak-marker"
    result = await make_run_command_tool().execute(
        {
            "command": "{ sleep 30; touch leak-marker; } >/dev/null 2>&1 & wait",
            "timeout_seconds": 0.2,
        },
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    await asyncio.sleep(0.3)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_shell_output_is_bounded(tmp_path):
    result = await make_run_command_tool().execute(
        {"command": "yes x | head -c 4000000"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert len(result.content) <= MAX_COMMAND_OUTPUT_BYTES
    assert result.metadata["truncated"] is True
