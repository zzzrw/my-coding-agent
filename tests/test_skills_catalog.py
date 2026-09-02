from pathlib import Path

from coding_agent.skills.catalog import (
    catalog_lines,
    format_catalog,
    overlay_lines,
)
from coding_agent.skills.models import Skill


def _skill(name, description, when_to_use=None):
    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        path=Path(f"/x/{name}/SKILL.md"),
        root="workspace",
    )


def test_format_catalog_renders_sorted_lines():
    skills = [_skill("zulu", "last"), _skill("alpha", "first")]
    assert format_catalog(skills) == "## Available skills\n- alpha: first\n- zulu: last"


def test_format_catalog_empty_is_empty_string():
    assert format_catalog([]) == ""


def test_format_catalog_collapses_multiline_descriptions():
    skill = _skill("demo", "line one\nline two")
    assert format_catalog([skill]) == "## Available skills\n- demo: line one line two"


def test_format_catalog_never_includes_skill_body():
    skill = _skill("demo", "short", when_to_use="use it")
    assert format_catalog([skill]) == "## Available skills\n- demo: short"
    assert not hasattr(skill, "body")
    assert "when to use" not in format_catalog([skill])


def test_catalog_lines_match_format_catalog_rows():
    skills = [_skill("zulu", "last"), _skill("alpha", "first")]
    assert format_catalog(skills) == "## Available skills\n" + "\n".join(
        catalog_lines(skills)
    )
    assert catalog_lines([]) == []


def test_overlay_lines_include_when_to_use_when_present():
    rows = overlay_lines([_skill("demo", "Do it", "when running")])
    assert "  when to use: when running" in rows
    rows_plain = overlay_lines([_skill("plain", "Do it")])
    assert rows_plain == ["- plain: Do it"]


def test_catalog_output_is_deterministic():
    skills = [_skill("zulu", "last"), _skill("alpha", "first")]
    first = format_catalog(skills)
    second = format_catalog(list(reversed(skills)))
    assert first == second
    assert first == format_catalog(skills)
