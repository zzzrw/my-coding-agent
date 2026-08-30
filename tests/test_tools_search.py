import asyncio
from pathlib import Path

import pytest

from coding_agent.tools.registry import ToolContext
from coding_agent.tools.search import make_grep_files_tool, make_list_files_tool


@pytest.mark.asyncio
async def test_list_and_grep_skip_build_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("needle\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00needle\xff")
    context = ToolContext(workspace=tmp_path, permission_mode="full")
    listing = await make_list_files_tool().execute(
        {}, context=context, signal=asyncio.Event()
    )
    matches = await make_grep_files_tool().execute(
        {"pattern": "needle"}, context=context, signal=asyncio.Event()
    )
    assert listing.ok and "node_modules" not in listing.content
    assert "src/main.py" in matches.content
    assert "ignored.js" not in matches.content
    assert "binary.bin" not in matches.content


@pytest.mark.asyncio
async def test_search_results_are_bounded_and_sorted(tmp_path):
    for name in ("z.txt", "a.txt"):
        (tmp_path / name).write_text("needle\n", encoding="utf-8")
    context = ToolContext(workspace=tmp_path, permission_mode="full")
    result = await make_grep_files_tool().execute(
        {"pattern": "needle", "max_results": 1},
        context=context,
        signal=asyncio.Event(),
    )
    assert result.ok and result.metadata["truncated"] is True
    assert result.content.startswith("a.txt:1:")


@pytest.mark.asyncio
async def test_exact_search_and_listing_limits_are_not_marked_truncated(tmp_path):
    (tmp_path / "only.txt").write_text("needle\n", encoding="utf-8")
    context = ToolContext(workspace=tmp_path, permission_mode="full")
    listing = await make_list_files_tool().execute(
        {"max_entries": 1}, context=context, signal=asyncio.Event()
    )
    matches = await make_grep_files_tool().execute(
        {"pattern": "needle", "max_results": 1},
        context=context,
        signal=asyncio.Event(),
    )
    assert listing.metadata["truncated"] is False
    assert matches.metadata["truncated"] is False


@pytest.mark.asyncio
async def test_recursive_listing_counts_files_not_directories(tmp_path):
    for index in range(205):
        (tmp_path / f"dir-{index:03}").mkdir()
    (tmp_path / "z.txt").write_text("x", encoding="utf-8")
    result = await make_list_files_tool().execute(
        {"recursive": True, "max_entries": 1},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.content == "z.txt"
    assert result.metadata["truncated"] is False


@pytest.mark.asyncio
async def test_list_uses_resolved_workspace_for_relative_display(tmp_path, monkeypatch):
    (tmp_path / "only.txt").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    result = await make_list_files_tool().execute(
        {},
        context=ToolContext(workspace=Path(tmp_path.name), permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert result.content == "only.txt"


@pytest.mark.asyncio
async def test_search_supports_non_recursive_and_include_filter(tmp_path):
    (tmp_path / "top.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "top.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.py").write_text("needle\n", encoding="utf-8")
    context = ToolContext(workspace=tmp_path, permission_mode="full")
    listing = await make_list_files_tool().execute(
        {"recursive": False}, context=context, signal=asyncio.Event()
    )
    result = await make_grep_files_tool().execute(
        {"pattern": "needle", "include": "*.py"},
        context=context,
        signal=asyncio.Event(),
    )
    assert "top.py" in listing.content and "deep.py" not in listing.content
    assert "top.py" in result.content and "top.txt" not in result.content


@pytest.mark.asyncio
async def test_grep_approved_external_path_returns_absolute_match(tmp_path):
    outside = tmp_path.parent / "coding-agent-search-outside.txt"
    outside.write_text("needle\n", encoding="utf-8")
    result = await make_grep_files_tool().execute(
        {"pattern": "needle", "path": str(outside.parent)},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert str(outside) in result.content
