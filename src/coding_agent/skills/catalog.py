"""Deterministic text renderers for the discovered skill catalog."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Skill

_CATALOG_HEADING = "## Available skills"


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _sorted(skills: Sequence[Skill]) -> list[Skill]:
    return sorted(skills, key=lambda s: s.name)


def catalog_lines(skills: Sequence[Skill]) -> list[str]:
    """One ``- name: description`` line per skill, sorted by name."""
    return [f"- {s.name}: {_one_line(s.description)}" for s in _sorted(skills)]


def format_catalog(skills: Sequence[Skill]) -> str:
    """The ``## Available skills`` system-prompt section; "" when empty."""
    rows = catalog_lines(skills)
    if not rows:
        return ""
    return _CATALOG_HEADING + "\n" + "\n".join(rows)


def overlay_lines(skills: Sequence[Skill]) -> list[str]:
    """TUI overlay rows: ``- name: description`` plus ``when_to_use`` line."""
    rows: list[str] = []
    for skill in _sorted(skills):
        rows.append(f"- {skill.name}: {_one_line(skill.description)}")
        if skill.when_to_use:
            rows.append(f"  when to use: {_one_line(skill.when_to_use)}")
    return rows
