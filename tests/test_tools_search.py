import asyncio

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
