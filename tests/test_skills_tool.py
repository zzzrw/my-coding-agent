import asyncio
from pathlib import Path

import pytest
from coding_agent.skills.tool import make_load_skill_tool

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.tools.registry import ToolContext

BODY = "Do the conventional-commits thing.\n\nUse type, scope, subject."


def _write(root: Path, name: str, text: str) -> None:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workspace_skills(tmp_path: Path) -> Path:
    """The workspace discovery root for a ``tmp_path`` workspace."""
    return tmp_path / ".coding-agent" / "skills"


@pytest.mark.asyncio
async def test_load_skill_returns_body_without_frontmatter(tmp_path):
    _write(
        _workspace_skills(tmp_path), "demo", "---\ndescription: Do it.\n---\n\n" + BODY
    )
    tool = make_load_skill_tool()
    context = ToolContext(workspace=tmp_path, permission_mode="full")

    result = await tool.execute(
        {"skill": "demo"}, context=context, signal=asyncio.Event()
    )

    assert result.ok is True
    assert result.content == BODY
    assert result.metadata["name"] == "demo"
    assert result.metadata["path"].endswith("demo/SKILL.md")


@pytest.mark.asyncio
async def test_load_skill_user_root_skill_is_found(tmp_path):
    user_root = tmp_path / "user"
    _write(user_root, "demo", "---\ndescription: Do it.\n---\n\nUSER-" + BODY)
    tool = make_load_skill_tool(user_root=user_root)
    context = ToolContext(workspace=tmp_path, permission_mode="full")

    result = await tool.execute(
        {"skill": "demo"}, context=context, signal=asyncio.Event()
    )

    assert result.ok is True
    assert result.content == "USER-" + BODY


@pytest.mark.asyncio
async def test_load_skill_prefers_workspace_over_user_root(tmp_path):
    user_root = tmp_path / "user"
    _write(
        _workspace_skills(tmp_path),
        "dupe",
        "---\ndescription: Do it.\n---\n\nWORKSPACE-BODY",
    )
    _write(user_root, "dupe", "---\ndescription: Do it.\n---\n\nUSER-BODY")
    tool = make_load_skill_tool(user_root=user_root)
    context = ToolContext(workspace=tmp_path, permission_mode="full")

    result = await tool.execute(
        {"skill": "dupe"}, context=context, signal=asyncio.Event()
    )

    assert result.ok is True
    assert result.content == "WORKSPACE-BODY"


@pytest.mark.asyncio
async def test_load_skill_frontmatter_name_is_the_effective_name(tmp_path):
    _write(
        _workspace_skills(tmp_path),
        "dir",
        "---\nname: shiny\ndescription: Do it.\n---\n\nSHINY-BODY",
    )
    tool = make_load_skill_tool(user_root=tmp_path / "user")
    context = ToolContext(workspace=tmp_path, permission_mode="full")

    found = await tool.execute(
        {"skill": "shiny"}, context=context, signal=asyncio.Event()
    )
    not_found = await tool.execute(
        {"skill": "dir"}, context=context, signal=asyncio.Event()
    )

    assert found.ok is True
    assert found.metadata["name"] == "shiny"
    assert not_found.ok is False


@pytest.mark.asyncio
async def test_load_skill_unknown_name_is_ok_false(tmp_path):
    tool = make_load_skill_tool(user_root=tmp_path / "user")
    context = ToolContext(workspace=tmp_path, permission_mode="full")

    result = await tool.execute(
        {"skill": "missing"}, context=context, signal=asyncio.Event()
    )

    assert result.ok is False
    assert result.error == "unknown skill: missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe", ["..", ".", "../secret", "/etc", "a/b"])
async def test_load_skill_unsafe_names_are_unknown(tmp_path, unsafe):
    _write(_workspace_skills(tmp_path), "ok", "---\ndescription: Do it.\n---\n\nBODY")
    tool = make_load_skill_tool(user_root=tmp_path / "user")
    context = ToolContext(workspace=tmp_path, permission_mode="full")

    result = await tool.execute(
        {"skill": unsafe}, context=context, signal=asyncio.Event()
    )

    assert result.ok is False
    assert "unknown skill" in (result.error or "")


@pytest.mark.asyncio
async def test_load_skill_truncates_huge_body_with_fixed_note(tmp_path):
    _write(
        _workspace_skills(tmp_path),
        "huge",
        "---\ndescription: Big.\n---\n\n" + "x" * 20_000,
    )
    tool = make_load_skill_tool(user_root=tmp_path / "user")
    context = ToolContext(workspace=tmp_path, permission_mode="full")

    result = await tool.execute(
        {"skill": "huge"}, context=context, signal=asyncio.Event()
    )

    assert result.ok is True
    note = "[skill body truncated at 16000 characters]"
    assert result.content.startswith("x" * 16_000)
    assert note in result.content
    assert result.metadata["truncated"] is True
    assert len(result.content) <= 16_000 + len("\n\n" + note)


def test_load_skill_schema_is_read_and_parallel_safe():
    schema = make_load_skill_tool().schema

    assert schema.name == "load_skill"
    assert schema.risk_level == "read"
    assert schema.is_parallel_safe is True
    assert "skill" in schema.parameters["required"]


def test_load_skill_read_policy_allows_in_all_modes(tmp_path):
    tool = make_load_skill_tool()
    schema = tool.schema
    for mode in ("default", "workspace", "full"):
        decision = DefaultApprovalPolicy().decide(
            schema, {"skill": "demo"}, workspace=tmp_path, mode=mode
        )
        assert decision.kind == "allow"


def test_load_skill_registered_with_app_registry():
    from coding_agent.app import _make_registry

    schemas = {s.name: s for s in _make_registry().schemas()}
    assert schemas["load_skill"].risk_level == "read"
