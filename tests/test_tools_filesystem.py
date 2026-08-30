import asyncio

import pytest

from coding_agent.tools.filesystem import (
    make_edit_file_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from coding_agent.tools.registry import ToolContext


@pytest.mark.asyncio
async def test_read_file_is_line_bounded(tmp_path):
    (tmp_path / "main.py").write_text("a\nb\nc\n", encoding="utf-8")
    result = await make_read_file_tool().execute(
        {"path": "main.py", "start_line": 2, "end_line": 2},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert result.content == "b"


@pytest.mark.asyncio
async def test_write_file_creates_parent_and_replaces_atomically(tmp_path):
    result = await make_write_file_tool().execute(
        {"path": "src/new.txt", "content": "hello"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert (tmp_path / "src/new.txt").read_text() == "hello"
    assert list((tmp_path / "src").glob(".*.new.txt.*")) == []


@pytest.mark.asyncio
async def test_edit_requires_exactly_one_match(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("x\nx\n", encoding="utf-8")
    result = await make_edit_file_tool().execute(
        {"path": "a.txt", "old_text": "x", "new_text": "y"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "exactly once" in (result.error or "")


@pytest.mark.asyncio
async def test_edit_replaces_single_match(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("before\n", encoding="utf-8")
    result = await make_edit_file_tool().execute(
        {"path": "a.txt", "old_text": "before", "new_text": "after"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert path.read_text() == "after\n"


@pytest.mark.asyncio
async def test_path_escape_requires_override_and_is_one_call(
    tmp_path, tmp_path_factory
):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    restricted = ToolContext(workspace=tmp_path, permission_mode="workspace")
    denied = await make_read_file_tool().execute(
        {"path": "../outside.txt"}, context=restricted, signal=asyncio.Event()
    )
    assert denied.ok is False
    assert "workspace" in (denied.error or "")
    allowed = await make_read_file_tool().execute(
        {"path": "../outside.txt"},
        context=restricted.model_copy(update={"allow_outside_once": True}),
        signal=asyncio.Event(),
    )
    assert allowed.ok is True
    assert allowed.content == "secret"
