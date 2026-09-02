import os
from pathlib import Path

import pytest

from coding_agent.skills.discovery import (
    discover_skills,
    resolve_skill,
)
from coding_agent.skills.models import Skill

_SKILL = "---\nname: {name}\ndescription: {desc}\nwhen_to_use: {when}\n---\n\n{body}"


def _write_skill(root: Path, dir_name: str, text: str) -> None:
    path = root / dir_name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workspace_root(tmp_path: Path) -> Path:
    return tmp_path / ".coding-agent" / "skills"


def test_discover_finds_workspace_and_user_skills(tmp_path):
    ws_root = _workspace_root(tmp_path)
    user_root = tmp_path / "user"
    _write_skill(ws_root, "alpha", _SKILL.format(name="alpha", desc="a", when="", body="A"))
    _write_skill(ws_root, "beta", _SKILL.format(name="beta", desc="b", when="", body="B"))
    _write_skill(user_root, "gamma", _SKILL.format(name="gamma", desc="g", when="", body="G"))

    skills = discover_skills(tmp_path, user_root=user_root)

    assert [skill.name for skill in skills] == ["alpha", "beta", "gamma"]
    assert {skill.root for skill in skills} == {"workspace", "user"}
    by_name = {skill.name: skill for skill in skills}
    assert by_name["alpha"].root == "workspace"
    assert by_name["gamma"].root == "user"


def test_workspace_skill_shadows_same_name_in_user_root(tmp_path):
    ws_root = _workspace_root(tmp_path)
    user_root = tmp_path / "user"
    _write_skill(ws_root, "alpha", _SKILL.format(name="alpha", desc="workspace-a", when="", body="W"))
    _write_skill(user_root, "alpha", _SKILL.format(name="alpha", desc="user-a", when="", body="U"))

    skills = discover_skills(tmp_path, user_root=user_root)

    assert [skill.name for skill in skills] == ["alpha"]
    assert skills[0].description == "workspace-a"
    assert skills[0].root == "workspace"


def test_discover_skips_invalid_candidates(tmp_path):
    ws_root = _workspace_root(tmp_path)
    (ws_root / "no-md").mkdir(parents=True)
    (ws_root / "no-frontmatter").mkdir()
    (ws_root / "no-frontmatter" / "SKILL.md").write_text("plain body\n", encoding="utf-8")
    (ws_root / "no-desc").mkdir()
    (ws_root / "no-desc" / "SKILL.md").write_text(
        "---\nname: no-desc\nwhen_to_use: x\n---\nbody", encoding="utf-8"
    )
    (ws_root / "bad-bytes").mkdir()
    (ws_root / "bad-bytes" / "SKILL.md").write_bytes(b"\xff\xfe\x00")
    _write_skill(ws_root, "ok", _SKILL.format(name="ok", desc="valid", when="", body="O"))

    skills = discover_skills(tmp_path, user_root=tmp_path / "user")

    assert [skill.name for skill in skills] == ["ok"]


def test_empty_roots_yield_empty_list(tmp_path):
    assert discover_skills(tmp_path, user_root=tmp_path / "empty-user") == []


def test_frontmatter_name_overrides_directory_name(tmp_path):
    ws_root = _workspace_root(tmp_path)
    _write_skill(ws_root, "zap", _SKILL.format(name="alpha", desc="renamed", when="", body="Z"))

    skills = discover_skills(tmp_path, user_root=tmp_path / "user")

    assert [skill.name for skill in skills] == ["alpha"]
    assert skills[0].path.name == "SKILL.md"


def test_unsafe_load_names_never_resolve(tmp_path):
    ws_root = _workspace_root(tmp_path)
    user_root = tmp_path / "user"
    _write_skill(ws_root, "ok", _SKILL.format(name="ok", desc="fine", when="", body="O"))
    for unsafe in ("..", ".", "a/b", "/etc", "\x00"):
        assert resolve_skill(unsafe, tmp_path, user_root=user_root) is None


def test_symlink_escape_skill_is_skipped(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("os.symlink unavailable")
    ws_root = _workspace_root(tmp_path)
    user_root = tmp_path / "user"
    user_root.mkdir()
    outside = tmp_path / "outside"
    _write_skill(outside, "real", _SKILL.format(name="real", desc="real", when="", body="R"))
    (user_root / "escape").symlink_to(outside, target_is_directory=True)

    skills = discover_skills(tmp_path, user_root=user_root)

    assert [skill.name for skill in skills] == []
    assert resolve_skill("escape", tmp_path, user_root=user_root) is None


def test_resolve_skill_prefers_workspace(tmp_path):
    ws_root = _workspace_root(tmp_path)
    user_root = tmp_path / "user"
    _write_skill(ws_root, "alpha", _SKILL.format(name="alpha", desc="workspace-a", when="", body="W"))
    _write_skill(user_root, "alpha", _SKILL.format(name="alpha", desc="user-a", when="", body="U"))

    skill = resolve_skill("alpha", tmp_path, user_root=user_root)

    assert skill is not None
    assert skill.description == "workspace-a"
    assert str(skill.path).endswith("alpha/SKILL.md")
    assert skill.root == "workspace"


def test_discover_is_deterministic_across_runs(tmp_path):
    ws_root = _workspace_root(tmp_path)
    user_root = tmp_path / "user"
    _write_skill(ws_root, "beta", _SKILL.format(name="beta", desc="b", when="", body="B"))
    _write_skill(ws_root, "alpha", _SKILL.format(name="alpha", desc="a", when="", body="A"))
    _write_skill(user_root, "gamma", _SKILL.format(name="gamma", desc="g", when="", body="G"))

    first = discover_skills(tmp_path, user_root=user_root)
    second = discover_skills(tmp_path, user_root=user_root)

    assert first == second
    assert [skill.name for skill in first] == ["alpha", "beta", "gamma"]


def test_skill_record_fields_are_populated(tmp_path):
    ws_root = _workspace_root(tmp_path)
    _write_skill(
        ws_root,
        "demo",
        "---\ndescription: Do the demo thing\nwhen_to_use: when demoing\n---\n\nDo it.\n",
    )

    skill = discover_skills(tmp_path, user_root=tmp_path / "user")[0]

    assert isinstance(skill, Skill)
    assert skill.name == "demo"
    assert skill.description == "Do the demo thing"
    assert skill.when_to_use == "when demoing"
    assert skill.path.suffix == ".md"
