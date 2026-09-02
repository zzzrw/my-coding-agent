"""Skill records for the skills feature.

Skills are user-authored instruction packages: a directory holding ``SKILL.md``
(YAML frontmatter plus a Markdown body) under one of two discovery roots. This
module defines the lightweight :class:`Skill` record and the fixed body cap the
``load_skill`` tool enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_SKILL_CONTENT_CHARS = 16_000

SkillRoot = Literal["workspace", "user"]


@dataclass(frozen=True, slots=True)
class Skill:
    """An installed skill discovered under one of the two roots."""

    name: str
    description: str
    when_to_use: str | None
    path: Path
    root: SkillRoot
