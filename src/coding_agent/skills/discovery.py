"""Two-root discovery of installed skills.

Skills live under ``<workspace>/.coding-agent/skills/<name>/SKILL.md``
(workspace root) or ``<config_dir>/skills/<name>/SKILL.md`` (user-global root,
the same config home that holds approvals). Discovery scans the workspace root
first; the first effective skill name wins and the result is sorted by name so
it is deterministic.

Only files under the two roots are ever read. Skill names are single path
segments; a name containing a separator or NUL byte, or equal to ``.``/``..``,
is rejected both when scanning and when a caller asks to load a skill. A
candidate is skipped silently whenever its ``SKILL.md`` is missing, unreadable,
or has no usable ``description`` — discovery and ``load_skill`` never raise for
a bad skill.
"""

from __future__ import annotations

from pathlib import Path

from coding_agent.config.config import config_dir

from .models import Skill, SkillRoot


def user_skills_root() -> Path:
    """The user-global skills root under the same config home as approvals."""
    return config_dir() / "skills"


def _roots(workspace, user_root):
    ws = Path(workspace) / ".coding-agent" / "skills"
    user = Path(user_root) if user_root is not None else user_skills_root()
    return ((ws, "workspace"), (user, "user"))


def _is_safe_name(name: str) -> bool:
    return (
        bool(name)
        and name not in {".", ".."}
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
    )


def _parse_skill_markdown(text: str) -> tuple[dict[str, str], str]:
    """Return ``({name?, description?, when_to_use?}, body)`` from a SKILL.md.

    Frontmatter is the region after a leading ``---`` line up to the next
    ``---`` line (or end of file). Unknown keys are ignored; a file that does
    not start with ``---`` yields ``({}, "")``. Body is empty when the closing
    delimiter is absent. Never raises.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ""
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    fields: dict[str, str] = {}
    for line in lines[1:] if end is None else lines[1:end]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in {"name", "description", "when_to_use"}:
            fields[key] = value.strip()
    body = "" if end is None else "\n".join(lines[end + 1 :])
    # The conventional blank separator after the closing ``---`` belongs to the
    # delimiter, not the body; drop leading blank lines so the body is exact.
    return fields, body.lstrip("\n")


def _skill_at(skill_dir: Path, root: Path, label: SkillRoot) -> Skill | None:
    if not _is_safe_name(skill_dir.name):
        return None
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None
    resolved_root = root.resolve()
    try:
        resolved = skill_md.resolve()
    except OSError:
        return None
    if not (resolved == resolved_root or resolved_root in resolved.parents):
        return None  # symlink escape / traversal
    try:
        text = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    fields, _ = _parse_skill_markdown(text)
    description = fields.get("description", "")
    if not description:
        return None
    declared = fields.get("name", "")
    name = declared if (declared and _is_safe_name(declared)) else skill_dir.name
    when = fields.get("when_to_use") or None
    return Skill(
        name=name, description=description, when_to_use=when, path=resolved, root=label
    )


def discover_skills(workspace, *, user_root=None) -> list[Skill]:
    """Discover installed skills, workspace root first, sorted by name.

    The workspace root shadows user-global skills with the same effective name.
    Invalid candidates and nonexistent roots are skipped silently; the result is
    deterministic for a given fixture tree. ``user_root`` overrides the config
    home so tests can point the user root at a scratch directory.
    """
    found: dict[str, Skill] = {}
    for root_path, label in _roots(workspace, user_root):
        if not root_path.is_dir():
            continue
        try:
            children = sorted(root_path.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            skill = _skill_at(child, root_path, label)
            if skill is not None and skill.name not in found:
                found[skill.name] = skill
    return [found[name] for name in sorted(found)]


def resolve_skill(name, workspace, *, user_root=None) -> Skill | None:
    """Resolve ``name`` to a Skill across the two roots (workspace first).

    Resolution matches the effective skill name (the frontmatter ``name`` when
    present, else the directory name) exactly as ``discover_skills`` reports it,
    so the catalog and ``load_skill`` always agree on what is loadable. Returns
    ``None`` for an unsafe name or when no matching skill exists, so the caller
    can report it as unknown.
    """
    if not _is_safe_name(name):
        return None
    for skill in discover_skills(workspace, user_root=user_root):
        if skill.name == name:
            return skill
    return None


def skill_body(skill: Skill) -> str:
    """The Markdown body (frontmatter excluded) of a resolved SKILL.md."""
    return _parse_skill_markdown(skill.path.read_text(encoding="utf-8"))[1]
