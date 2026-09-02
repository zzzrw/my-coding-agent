import asyncio

import pytest

from coding_agent.tools.filesystem import (
    MAX_READ_CHARS,
    make_clear_directory_tool,
    make_edit_file_tool,
    make_read_file_tool,
    make_remove_file_tool,
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
async def test_read_file_default_output_is_bounded(tmp_path):
    (tmp_path / "large.txt").write_text("x" * (MAX_READ_CHARS + 100), encoding="utf-8")
    result = await make_read_file_tool().execute(
        {"path": "large.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert len(result.content) == MAX_READ_CHARS
    assert result.metadata["truncated"] is True


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


@pytest.mark.asyncio
async def test_remove_file_deletes_a_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("bye", encoding="utf-8")
    result = await make_remove_file_tool().execute(
        {"path": "a.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert "removed" in result.content
    assert not target.exists()


@pytest.mark.asyncio
async def test_remove_file_deletes_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = await make_remove_file_tool().execute(
        {"path": "empty"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert not empty.exists()


@pytest.mark.asyncio
async def test_remove_file_refuses_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside-remove.txt"
    outside.write_text("secret", encoding="utf-8")
    result = await make_remove_file_tool().execute(
        {"path": "../outside-remove.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="workspace"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "workspace" in (result.error or "")
    assert outside.exists()


@pytest.mark.asyncio
async def test_remove_file_leaves_non_empty_directory_untouched(tmp_path):
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "x.txt").write_text("x", encoding="utf-8")
    result = await make_remove_file_tool().execute(
        {"path": "keep"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert keep.exists()
    assert (keep / "x.txt").exists()


@pytest.mark.asyncio
async def test_clear_directory_removes_contents_and_keeps_directory(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.txt").write_text("a", encoding="utf-8")
    nested = cache / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")
    result = await make_clear_directory_tool().execute(
        {"path": "cache"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert "cleared" in result.content
    assert cache.is_dir()
    assert list(cache.iterdir()) == []


@pytest.mark.asyncio
async def test_clear_directory_refuses_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside-cache"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    result = await make_clear_directory_tool().execute(
        {"path": "../outside-cache"},
        context=ToolContext(workspace=tmp_path, permission_mode="workspace"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "workspace" in (result.error or "")
    assert (outside / "keep.txt").exists()


@pytest.mark.asyncio
async def test_clear_directory_rejects_a_file_path(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    result = await make_clear_directory_tool().execute(
        {"path": "a.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "not a directory" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "x"


def test_delete_tools_registered_with_app_registry():
    from coding_agent.app import _make_registry

    schemas = {schema.name: schema for schema in _make_registry().schemas()}
    for name in ("remove_file", "clear_directory"):
        assert name in schemas
        assert schemas[name].risk_level == "mutate_file"
