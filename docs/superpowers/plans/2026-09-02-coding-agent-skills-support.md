# Skills Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let the user author reusable instruction packages ("skills", a
directory per skill holding `SKILL.md`) that the agent enumerates cheaply at the
start of every session and loads on demand via a `load_skill` tool. Implement
the approved design in
`docs/superpowers/specs/2026-09-02-coding-agent-skills-support.md` (already
committed, `3e9d63a`) with small TDD tasks, each independently committable and
the full suite green at every boundary.

**Authoritative contract** is the AGREED DESIGN CONTRACT pinned by the human
owner and mirrored in the spec; where this plan omits detail, the spec's §1-§12
control. Progressive disclosure: a cheap deterministic catalog is always in the
run system prompt; the full `SKILL.md` body is pulled only through `load_skill`.

**Architecture:**
- New `src/coding_agent/skills/` package: `models.py` (`Skill` record +
  `MAX_SKILL_CONTENT_CHARS = 16_000`), `discovery.py` (two-root discovery,
  frontmatter parse, name-safety + path-containment, `discover_skills`,
  `resolve_skill`, `skill_body`), `catalog.py` (`format_catalog` and the TUI
  row builders), `tool.py` (`make_load_skill_tool(...)` factory).
- Discovery roots: workspace `<workspace>/.coding-agent/skills/<name>/SKILL.md`
  first, then user-global `config_dir()/skills/<name>/SKILL.md` (reusing
  `config.config.config_dir`), first-wins on the effective name, results sorted
  by name. `discover_skills(workspace, *, user_root=None)` and
  `resolve_skill(name, workspace, *, user_root=None)` accept a `user_root`
  override so tests point the user root at `tmp_path`; when `None` the root is
  `user_skills_root()` (`config_dir() / "skills"`).
- The system prompt `Message` stays built by `build_system_prompt(...)` in
  `src/coding_agent/app.py`; it gains a `skills` keyword argument and appends
  `format_catalog(skills)` (only when non-empty). `create_app` discovers the
  catalog once (`discover_skills(resolved_workspace)`) and passes it in. Empty
  skill set ⇒ byte-identical system prompt to today.
- `load_skill` is registered in `_make_registry` exactly like the existing
  tools (schema surfaced through `registry.schemas()`, executed/bounded by the
  ordinary `ToolExecutor`), `risk_level="read"`, `is_parallel_safe=True`, so it
  is allowed in every permission mode without an approval prompt.
- TUI: a `/skills` command (`CommandSuggestion` in `tui/commands.py`) and a
  "Skills" section in the help overlay, plus a `SkillsScreen` modal, all driven
  by the same discovery function as the system prompt. No free-form
  `/<skill-name>` expansion in v1.

**Tech Stack:** Python 3.11+, pydantic, pytest (asyncio auto mode), ruff,
textual.

**Spec:** `docs/superpowers/specs/2026-09-02-coding-agent-skills-support.md`
§1-§12.

## Global Constraints

- Every commit ends with the trailer:
  `Co-Authored-By: Claude <noreply@anthropic.com>`
  (second `-m` in the commit commands below).
- Baseline before this round: `632 passed, 1 skipped` (`uv run pytest -q`,
  41s). Each task keeps that suite green; only tests listed as "reconcile"
  change, all new behavior is additive.
- Tests are fast and deterministic with no network. Fixtures are small
  `SKILL.md` trees under `tmp_path` for both the workspace root
  (`tmp/.coding-agent/skills/<name>/SKILL.md`) and, via the `user_root`
  override, the user root.
- Skill files are only ever resolved under the two roots. Names are single path
  segments; a name containing `/` or `\`, a NUL byte, or equal to `.`/`..` is
  rejected both when scanning and when loading. Resolution joins the safe name
  to a root, `.resolve()`s, and requires the final path to live under the
  `.resolve()`d root (this also rejects symlink escapes). Nothing outside the
  two roots is ever read.
- Discovery never raises: an unreadable/undecodable `SKILL.md`, a missing
  `SKILL.md`, or a missing usable `description` ⇒ candidate skipped silently;
  nonexistent roots ⇒ empty result. A bad skill never crashes discovery, the
  app, or the tool.
- No SKILL.md body is ever auto-injected into the system prompt; only the
  catalog section is.
- `load_skill` enforces its own ~16 000-char body cap (the executor's 20 000
  downstream bound is not relied on); truncation appends a fixed note.
- v1 out of scope: resource folders/scripts execution, model authoring of
  skills, forked sub-agents, watchers/hot reload, remote/org/plugin/bundled/
  per-session skills, token budgets, conditional activation, frontmatter beyond
  `name`/`description`/`when_to_use`, and free-form `/<skill-name>` commands.
- Executor commands below assume an activated dev environment (`uv run pytest`
  / `uv run ruff`). Do NOT push; keep all work inside the feature worktree.
- Determinism on developer machines: once `create_app`/`build_system_prompt`
  start discovering the real user root (Task 3), a developer who has skills
  installed in `~/.config/coding-agent/skills` will see a catalog section in
  every real run. The test suite asserts membership, never whole-prompt
  equality, so it stays green; to be certain, run the final gate with an empty
  user skills dir or a scratch `XDG_CONFIG_HOME` (e.g.
  `XDG_CONFIG_HOME=$(mktemp -d) uv run pytest -q`).

## File Map

- `src/coding_agent/skills/__init__.py` (new)
- `src/coding_agent/skills/models.py` (new): `Skill`, `MAX_SKILL_CONTENT_CHARS`
- `src/coding_agent/skills/discovery.py` (new): roots, parse, `discover_skills`,
  `resolve_skill`, `skill_body`, name/path safety
- `src/coding_agent/skills/catalog.py` (new): `format_catalog` + TUI row
  builders
- `src/coding_agent/skills/tool.py` (new): `make_load_skill_tool`
- `src/coding_agent/app.py` (modify): register `load_skill`; discover and pass
  the catalog into `build_system_prompt`
- `src/coding_agent/tui/commands.py` (modify): `skills` `CommandSuggestion`
- `src/coding_agent/tui/widgets.py` (modify): `HelpScreen(skills)`,
  `help_overlay_text(skills)` Skills section, `SkillsScreen` +
  `skills_overlay_text`
- `src/coding_agent/tui/app.py` (modify): `skills_user_root` injection,
  `/skills` dispatch, live catalog into help, `#skills` CSS
- Tests: `tests/test_skills_discovery.py`,
  `tests/test_skills_catalog.py`, `tests/test_skills_system_prompt.py`,
  `tests/test_skills_tool.py`, `tests/test_skills_tui.py` (all new);
  `tests/test_commands.py`, `tests/test_integration_flow.py` (reconcile)
- `README.md` (modify): skills usage note

---

### Task 1: Skill model and two-root discovery with name/path safety

**Files:** New `src/coding_agent/skills/__init__.py`,
`src/coding_agent/skills/models.py`, `src/coding_agent/skills/discovery.py`;
new `tests/test_skills_discovery.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skills_discovery.py`. Fixture helper at top:

```python
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
```

Key tests (names + key assertions):

1. `test_discover_finds_workspace_and_user_skills(tmp_path)` — workspace has
   `alpha` and `beta`, user root has `gamma`; with
   `discover_skills(ws_root_root, user_root=user_root)` the returned names are
   `["alpha", "beta", "gamma"]`, sorted, and each carries its root label
   (`"workspace"`/`"user"`).

2. `test_workspace_skill_shadows_same_name_in_user_root(tmp_path)` — the same
   skill name `alpha` exists in both roots with different descriptions; the
   result list has exactly one `alpha`, and its `description` is the workspace
   one.

3. `test_discover_skips_invalid_candidates(tmp_path)` — a workspace containing:
   a dir with no `SKILL.md`, a `SKILL.md` with no frontmatter, one whose
   `description` is missing/empty, and one whose `SKILL.md` is undecodable
   bytes (`b"\xff\xfe\x00"`). Discovery returns only the valid skill and never
   raises.

4. `test_empty_roots_yield_empty_list(tmp_path)` — empty workspace + empty user
   root ⇒ `discover_skills(...) == []`.

5. `test_frontmatter_name_overrides_directory_name(tmp_path)` — directory
   `zap` whose `SKILL.md` frontmatter has `name: alpha` ⇒ effective name
   `alpha`, not `zap`.

6. `test_unsafe_load_names_never_resolve(tmp_path)` — for each of
   `".."`, `"."`, `"a/b"`, `"/etc"`, and `"\x00"`,
   `resolve_skill(name, ws, user_root=user) is None` even when a sibling skill
   exists.

7. `test_symlink_escape_skill_is_skipped(tmp_path)` — guard `if not
   hasattr(os, "symlink")`. Put a real skill outside the user root, then
   `(user_root / "escape").symlink_to(outside_dir, target_is_directory=True)`.
   `discover_skills` skips `escape`, and
   `resolve_skill("escape", ws, user_root=user_root) is None`.

8. `test_resolve_skill_prefers_workspace(tmp_path)` — `alpha` in both roots;
   `resolve_skill("alpha", ...)` returns the workspace skill; the path
   endswith `alpha/SKILL.md` under the workspace root.

9. `test_discover_is_deterministic_across_runs(tmp_path)` — calling
   `discover_skills` twice on the same fixture roots returns identical lists.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_skills_discovery.py -q`
Expected: collection FAILS — `coding_agent.skills.discovery` does not exist.

- [ ] **Step 3: Implement the model and discovery module**

`src/coding_agent/skills/__init__.py`: empty module docstring only.

`src/coding_agent/skills/models.py`:

```python
"""Skill records for the skills feature."""

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
```

`src/coding_agent/skills/discovery.py`: reuse `config_dir` from
`coding_agent.config.config`. Core pieces:

```python
def user_skills_root() -> Path:
    """The user-global skills root under the same config home as approvals."""
    return config_dir() / "skills"


def _roots(workspace, user_root):
    ws = Path(workspace) / ".coding-agent" / "skills"
    user = Path(user_root) if user_root is not None else user_skills_root()
    return ((ws, "workspace"), (user, "user"))


def _is_safe_name(name: str) -> bool:
    return bool(name) and name not in {".", ".."} and "/" not in name \
        and "\\" not in name and "\x00" not in name


def _parse_skill_markdown(text: str) -> tuple[dict[str, str], str]:
    """Return ({name?, description?, when_to_use?}, body) from a SKILL.md.

    Frontmatter is the region after a leading ``---`` line up to the next
    ``---`` line (or end of file). Unknown keys are ignored; a file that does
    not start with ``---`` yields ({}, ""). Body is empty when the closing
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
    return fields, body


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
    return Skill(name=name, description=description, when_to_use=when,
                 path=resolved, root=label)
```

`discover_skills` scans workspace then user; per root it walks child
directories in sorted order, keeps the first effective name (workspace wins),
then returns the kept skills sorted by name:

```python
def discover_skills(workspace, *, user_root=None) -> list[Skill]:
    found: dict[str, Skill] = {}
    for root_path, label in _roots(workspace, user_root):
        if not root_path.is_dir():
            continue
        for child in sorted(root_path.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            skill = _skill_at(child, root_path, label)
            if skill is not None and skill.name not in found:
                found[skill.name] = skill
    return [found[name] for name in sorted(found)]


def resolve_skill(name, workspace, *, user_root=None) -> Skill | None:
    if not _is_safe_name(name):
        return None
    for root_path, label in _roots(workspace, user_root):
        skill = _skill_at(root_path / name, root_path, label)
        if skill is not None:
            return skill
    return None


def skill_body(skill: Skill) -> str:
    """The Markdown body (frontmatter excluded) of a resolved SKILL.md."""
    return _parse_skill_markdown(skill.path.read_text(encoding="utf-8"))[1]
```

`sorted(root_path.iterdir(), ...)` may itself raise on an unreadable directory,
so wrap the scan body in `try/except OSError: continue`.

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_skills_discovery.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/skills/__init__.py src/coding_agent/skills/models.py \
  src/coding_agent/skills/discovery.py tests/test_skills_discovery.py
git commit -m "Add skill discovery over workspace and user roots" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Deterministic catalog renderer

**Files:** New `src/coding_agent/skills/catalog.py`; new
`tests/test_skills_catalog.py`.

Depends on Task 1 (`Skill` in `skills/models.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skills_catalog.py`:

```python
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
```

1. `test_format_catalog_renders_sorted_lines()` — for
   `[_skill("zulu", "last"), _skill("alpha", "first")]`,
   `format_catalog(skills)` equals exactly:

   ```text
   ## Available skills
   - alpha: first
   - zulu: last
   ```

2. `test_format_catalog_empty_is_empty_string()` —
   `format_catalog([]) == ""`.

3. `test_format_catalog_collapses_multiline_descriptions()` — a description
   `"line one\nline two"` renders as `- demo: line one line two`.

4. `test_format_catalog_never_includes_skill_body()` — a skill whose
   description is short but whose body marker `"BODY-SECRET"` would not be
   present (format_catalog takes `Skill` only, so construct skills with
   `when_to_use`/description and assert no body field exists on the record).

5. `test_catalog_lines_match_format_catalog_rows()` —
   `format_catalog(skills)` is `"## Available skills\n" +
   "\n".join(catalog_lines(skills))`; `catalog_lines([]) == []`.

6. `test_overlay_lines_include_when_to_use_when_present()` —
   `overlay_lines([_skill("demo", "Do it", "when running")])` contains
   `"  when to use: when running"`; a skill without `when_to_use` emits only
   its `- demo: Do it` row.

7. `test_catalog_output_is_deterministic()` — identical inputs (unsorted) give
   identical output across repeated calls.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_skills_catalog.py -q`
Expected: collection FAILS — `coding_agent.skills.catalog` does not exist.

- [ ] **Step 3: Implement the renderer**

`src/coding_agent/skills/catalog.py`:

```python
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
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_skills_catalog.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/skills/catalog.py tests/test_skills_catalog.py
git commit -m "Add deterministic skills catalog renderer" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: System-prompt catalog seam

**Files:** Modify `src/coding_agent/app.py`; new `tests/test_skills_system_prompt.py`.

Hazard: this task edits the same `build_system_prompt`/`create_app` region the
safe-ops round touched. It is the only consumer seam; ordering it before the
tool registration keeps each task green. Existing system-prompt tests assert
membership (not equality), so appending a catalog section for non-empty skill
sets cannot break them; empty sets append nothing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skills_system_prompt.py`:

```python
from pathlib import Path

from coding_agent.app import build_system_prompt, create_app
from coding_agent.skills.discovery import discover_skills


def _workspace_root(tmp_path: Path) -> Path:
    return tmp_path / ".coding-agent" / "skills"


def _write_skill(root: Path, text: str) -> None:
    path = root / "demo" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
```

1. `test_system_prompt_with_fixture_skills_includes_catalog(tmp_path)` — write
   a workspace skill, `skills = discover_skills(tmp_path, user_root=empty_tmp)`,
   `content = build_system_prompt(tmp_path, "default", skills=skills).content`.
   Assert `"## Available skills"` in content and
   `"- demo: Do the demo thing"` in content.

2. `test_system_prompt_empty_skill_set_omits_catalog_and_keeps_content(tmp_path)`
   — `content = build_system_prompt(tmp_path, "full", skills=()).content`; assert
   `"## Available skills" not in content`, `"Permission boundaries" in content`,
   `str(tmp_path) in content`, and `"You are coding-agent" in content`.

3. `test_system_prompt_excludes_skill_bodies(tmp_path)` — the workspace skill
   body contains `"BODY-ONLY-SECRET"`; assert it is not in the prompted content
   even when the catalog line is.

4. `test_create_app_wires_the_discovered_catalog(tmp_path)` — write the fixture
   skill in the workspace; use a tiny recording provider (returning only a
   `response_end` LLMEvent, mirroring `SequencedFakeProvider` in
   `tests/test_integration_flow.py`); `create_app(workspace=tmp_path,
   model="fake-model", session_dir=tmp_path/"sessions", context_window=2_000,
   provider=provider, permission_mode="full")`; `await runtime.submit("hi")`;
   assert the system message the provider received
   (`provider.requests[0][0][0]`) has `role == "system"` and its content
   contains `"- demo: Do the demo thing"`.

   Note: `create_app` discovers the real user root too; assert presence of the
   fixture line only (robust even if the machine has extra user skills).

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_skills_system_prompt.py -q`
Expected: test 1-3 FAIL (catalog absent: `build_system_prompt` has no `skills`
keyword and never emits a catalog); test 4 FAILS (no catalog in the wired
prompt). Existing `tests/test_app_system_prompt.py` still passes.

- [ ] **Step 3: Implement the seam**

In `src/coding_agent/app.py`:

Edit A — imports:

```python
from coding_agent.skills.catalog import format_catalog
from coding_agent.skills.discovery import discover_skills
from coding_agent.skills.models import Skill
```

Edit B — extend the builder signature and append the section when present:

```python
def build_system_prompt(
    workspace: Path,
    permission_mode: PermissionMode,
    *,
    skills: Sequence[Skill] = (),
) -> Message:
    """Return the system prompt for a run.

    ``skills`` is the already-discovered catalog; when empty no catalog section
    is emitted so a skill-less run stays byte-identical to prior builds.
    """
    section = format_catalog(skills)
    return Message(
        role="system",
        content=(
            "... unchanged current text ending with f\"- Workspace root: {workspace}\""
            + (("\n\n" + section) if section else "")
        ),
    )
```

(Mechanically: keep the existing `Message(role="system", content=(...))`
literal exactly as-is, then append the catalog only when `section` is
non-empty.)

Edit C — in `create_app`, discover once and pass it in, replacing the current
`system_prompt = build_system_prompt(resolved_workspace, permission_mode)`:

```python
    skills = discover_skills(resolved_workspace)
    system_prompt = build_system_prompt(
        resolved_workspace, permission_mode, skills=skills
    )
```

`discover_skills` never raises and returns `[]` for a skill-less workspace, so
`create_app` behavior is unchanged there.

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_skills_system_prompt.py tests/test_app_system_prompt.py -q`
Expected: all PASS.
Run: `uv run pytest tests/test_cli_provider_hardening.py tests/test_integration_flow.py tests/test_w6_configuration.py -q`
Expected: all PASS (`create_app` still wires a system prompt end-to-end).

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/app.py tests/test_skills_system_prompt.py
git commit -m "Embed the skills catalog in the system prompt" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: `load_skill` tool factory, registration, and behavior

**Files:** New `src/coding_agent/skills/tool.py`; new
`tests/test_skills_tool.py`; modify `src/coding_agent/app.py` (register) and
`tests/test_integration_flow.py` (reconcile the exact tool-name set).

Hazard: registration makes the registry's schema list grow by one; the only
exact tool-name snapshot is `test_factory_runs_write_verification_and_completion`
in `tests/test_integration_flow.py` (lines ~79-88) and it must gain
`"load_skill"` in the same commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skills_tool.py` (mirrors `tests/test_tools_search.py`
direct-execution style; `asyncio_mode = "auto"`):

```python
import asyncio
from pathlib import Path

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.skills.tool import make_load_skill_tool
from coding_agent.tools.registry import ToolContext

BODY = "Do the conventional-commits thing.\n\nUse type, scope, subject."


def _write(root: Path, name: str, text: str) -> None:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
```

1. `test_load_skill_returns_body_without_frontmatter(tmp_path)` — workspace
   skill `demo` whose file is
   `"---\ndescription: Do it.\n---\n\n" + BODY`; execute
   `make_load_skill_tool().execute({"skill": "demo"}, context=ToolContext(workspace=tmp_path, permission_mode="full"), signal=asyncio.Event())`.
   Assert `ok is True`, `content == BODY`, `metadata["name"] == "demo"`, and
   `metadata["path"]` endswith `demo/SKILL.md`.

2. `test_load_skill_user_root_skill_is_found(tmp_path)` — skill only under the
   user root; execute with `make_load_skill_tool(user_root=user_root)` and the
   same `context`; assert found with the user body.

3. `test_load_skill_prefers_workspace_over_user_root(tmp_path)` — `dupe` in
   both roots with different bodies; assert content is the workspace body.

4. `test_load_skill_frontmatter_name_is_the_effective_name(tmp_path)` —
   directory `dir` whose frontmatter declares `name: shiny`; executing with
   `{"skill": "shiny"}` succeeds and `metadata["name"] == "shiny"`; executing
   with `{"skill": "dir"}` returns `ok is False`.

5. `test_load_skill_unknown_name_is_ok_false(tmp_path)` — `{"skill":
   "missing"}` ⇒ `ok is False` and `error == "unknown skill: missing"`.

6. `test_load_skill_unsafe_names_are_unknown(tmp_path)` — parametrize over
   `".."`, `"."`, `"../secret"`, `"/etc"`, `"a/b"`; each returns `ok is False`
   with `"unknown skill"` in the error and never touches the filesystem outside
   the roots.

7. `test_load_skill_truncates_huge_body_with_fixed_note(tmp_path)` — body is
   20 000 `x` characters; execute; assert `ok is True`,
   `content.startswith("x" * 16_000)`, the fixed note
   `"[skill body truncated at 16000 characters]"` is in `content`, and
   `len(content) <= 16_000 + len(note)`.

8. `test_load_skill_schema_is_read_and_parallel_safe()` —
   `schema = make_load_skill_tool().schema`; assert `schema.name ==
   "load_skill"`, `schema.risk_level == "read"`,
   `schema.is_parallel_safe is True`, and `"skill"` is a required property of
   `schema.parameters`.

9. `test_load_skill_read_policy_allows_in_all_modes(tmp_path)` — for each of
   `"default"`, `"workspace"`, `"full"`,
   `DefaultApprovalPolicy().decide(schema, {"skill": "demo"}, workspace=tmp_path, mode=mode).kind == "allow"`.

10. `test_load_skill_registered_with_app_registry()` —

    ```python
    from coding_agent.app import _make_registry

    schemas = {s.name: s for s in _make_registry().schemas()}
    assert schemas["load_skill"].risk_level == "read"
    ```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_skills_tool.py -q`
Expected: collection FAILS — `coding_agent.skills.tool` does not exist, so the
module import errors.

- [ ] **Step 3: Implement the tool**

`src/coding_agent/skills/tool.py`, following the
`tools/filesystem.py`/`tools/search.py` factory pattern:

```python
"""The ``load_skill`` model tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from coding_agent.tools.models import ToolResult, ToolSchema

from .discovery import resolve_skill, skill_body
from .models import MAX_SKILL_CONTENT_CHARS

_TRUNCATION_NOTE = "\n\n[skill body truncated at 16000 characters]"


class _LoadSkillArgs(BaseModel):
    skill: str


def _result(name, ok, content="", error=None, **metadata) -> ToolResult:
    return ToolResult(
        tool_call_id="", tool_name=name, ok=ok, content=content,
        error=error, metadata=metadata,
    )


class _LoadSkillTool:
    args_model = _LoadSkillArgs
    schema = ToolSchema(
        name="load_skill",
        description=(
            "Load the body of an installed skill by name. Available skills are "
            "listed in the system prompt under 'Available skills'; call this "
            "before acting when the user names a skill or the task matches a "
            "listed description."
        ),
        parameters=_LoadSkillArgs.model_json_schema(),
        risk_level="read",
        is_parallel_safe=True,
    )

    def __init__(self, *, user_root: Path | None = None) -> None:
        self._user_root = user_root

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            skill = resolve_skill(
                args.skill, context.workspace, user_root=self._user_root
            )
            if skill is None:
                return _result(
                    self.schema.name, False,
                    error=f"unknown skill: {args.skill}",
                )
            if signal.is_set():
                return _result(self.schema.name, False, error="cancelled")
            body = skill_body(skill)
            truncated = len(body) > MAX_SKILL_CONTENT_CHARS
            if truncated:
                body = body[:MAX_SKILL_CONTENT_CHARS] + _TRUNCATION_NOTE
            return _result(
                self.schema.name, True, body,
                name=skill.name, path=str(skill.path), truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


def make_load_skill_tool(user_root: Path | None = None):
    return _LoadSkillTool(user_root=user_root)
```

`skill_body` re-reads the already-contained resolved `SKILL.md` and strips the
frontmatter; the tool performs no free filesystem reads (its only argument is a
skill name).

- [ ] **Step 4: Register the tool**

In `src/coding_agent/app.py`:

Edit A — add the import:

```python
from coding_agent.skills.tool import make_load_skill_tool
```

Edit B — in `_make_registry`, register it alongside the other tools (after the
filesystem/search tools, before `run_command`):

```python
        make_clear_directory_tool(),
        make_load_skill_tool(),
        make_run_command_tool(),
```

Edit C — reconcile the exact tool-name set in `tests/test_integration_flow.py`
(~line 79): add `"load_skill",` to the set literal so it reads
`{"read_file", "list_files", "grep_files", "write_file", "edit_file",
"remove_file", "clear_directory", "load_skill", "run_command"}`.

- [ ] **Step 5: Run and verify pass**

Run: `uv run pytest tests/test_skills_tool.py -q`
Expected: all PASS.
Run: `uv run pytest tests/test_integration_flow.py tests/test_registry.py -q`
Expected: all PASS (registry now surfaces `load_skill`; the integration flow
tool set is reconciled).

- [ ] **Step 6: Commit**

```bash
git add src/coding_agent/skills/tool.py tests/test_skills_tool.py \
  src/coding_agent/app.py tests/test_integration_flow.py
git commit -m "Add and register the load_skill tool" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: `/skills` command and the help overlay Skills section

**Files:** Modify `src/coding_agent/tui/commands.py`,
`src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/app.py`,
`tests/test_commands.py`; new `tests/test_skills_tui.py`.

Hazard: the TUI wiring reads `self.state.workspace`; existing command/palette
tests use nonexistent fake workspaces (`/tmp/project`), so discovery must
tolerate a missing workspace root (it does, per Task 1). Adding a
`CommandSuggestion` changes `SUPPORTED_COMMANDS`; the only exact-set test is
`test_supported_commands_match_the_mvp_registry` in `tests/test_commands.py` and
it must gain `"skills"` in the same commit.

- [ ] **Step 1: Write the failing tests**

Edit `tests/test_commands.py` — add `"skills"` to the expected set in
`test_supported_commands_match_the_mvp_registry`.

Create `tests/test_skills_tui.py` with a minimal `FakeRuntime` (workspace is
supplied through `initial_state`, mirroring `tests/test_tui.py`), fixture
`_write_skill` helpers (workspace root `tmp/.coding-agent/skills`), and a
`CodingAgentApp(runtime=..., initial_state=..., skills_user_root=user_tmp)`.

1. `test_skills_command_registered()` — `"skills" in SUPPORTED_COMMANDS` and
   `command_suggestions("ski")[0].description == "List available skills"`.

2. `test_help_overlay_empty_default_shows_no_skills()` —
   `help_overlay_text().plain` contains `"Skills"` and `"no skills installed"`.

3. `test_help_overlay_with_catalog_lists_skills(tmp_path)` —
   `help_overlay_text(skills=[...Skill("demo", "Do it")]).plain` contains
   `"Skills"` and `"- demo: Do it"`.

4. `test_skills_command_opens_overlay_listing_catalog(tmp_path)` — put
   `demo` (description + `when_to_use`) in the workspace root; app over the
   fixture workspace + empty user root; type `/skills` and press enter (as in
   `test_help_command_typed_in_composer_opens_help`); assert
   `isinstance(app.screen, SkillsScreen)` and `app.screen.body.plain` contains
   `"demo"`, the description, and the `when_to_use` text.

5. `test_skills_overlay_empty_state(tmp_path)` — empty workspace + empty user
   root; after `/skills`, the screen body contains `"no skills installed"`.

6. `test_skills_command_with_args_is_a_usage_notice(tmp_path)` — type
   `/skills extra`; assert a local notice with `"usage: /skills"` appears and
   no `SkillsScreen` is pushed.

7. `test_help_typed_in_composer_lists_installed_skills(tmp_path)` — with a
   fixture skill in the workspace, type `/help`; assert `app.screen` is
   `HelpScreen` and `app.screen.body.plain` contains the skill name under the
   Skills section.

8. `test_arbitrary_skill_slash_is_not_a_command()` — `parse_command("/demo")`
   yields `Command(name="demo", args=[])` and `"demo" not in SUPPORTED_COMMANDS`
   (no free-form expansion in v1).

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_skills_tui.py -q`
Expected: FAILS (no `SkillsScreen`, no `skills_user_root`, no `/skills` branch,
`help_overlay_text()` takes no arguments). The edited
`test_supported_commands_match_the_mvp_registry` also FAILS until the command
is registered.

- [ ] **Step 3: Implement the command, overlay, modal, and dispatch**

`src/coding_agent/tui/commands.py` — add to `_COMMANDS` (alphabetical, between
`session` and `undo`):

```python
    CommandSuggestion("skills", "List available skills", usage="/skills"),
```

`src/coding_agent/tui/widgets.py` — imports gain
`from collections.abc import Sequence` and
`from coding_agent.skills.catalog import catalog_lines, overlay_lines` and
`from coding_agent.skills.models import Skill`.

- `help_overlay_text(skills: Sequence[Skill] = ())`: after the existing
  "Permissions" section append a Skills section:

  ```python
  body.append("Skills", style="bold")
  body.append("\n")
  for line in catalog_lines(skills):
      body.append(f"  {line}\n")
  if not skills:
      body.append("  no skills installed\n")
  ```

- `HelpScreen.__init__(self, skills: Sequence[Skill] = ())` stores
  `self.skills = tuple(skills)` and `self.body = help_overlay_text(self.skills)`
  (no-argument construction stays valid).

- New `SkillsScreen(ModalScreen[None])` mirroring `HistoryScreen` (bordered,
  `$surface`, centered; `escape` closes):

  ```python
  class SkillsScreen(ModalScreen[None]):
      BINDINGS: ClassVar = [("escape", "close_skills", "Close")]

      def __init__(self, skills: Sequence[Skill]) -> None:
          super().__init__()
          self.skills = list(skills)
          self.body = skills_overlay_text(self.skills)

      def compose(self) -> ComposeResult:
          yield Static(self.body, id="skills", markup=False)

      def action_close_skills(self) -> None:
          self.dismiss(None)
  ```

- `skills_overlay_text(skills) -> Text`: heading `"Skills"` (bold), then one
  `overlay_lines` row per skill (`  - name: description` plus the indented
  `when to use:` line when present); empty ⇒ `"  no skills installed"`.

`src/coding_agent/tui/app.py` — imports gain
`from coding_agent.skills.discovery import discover_skills` and
`SkillsScreen` is added to the `coding_agent.tui.widgets` import.

- `CodingAgentApp.__init__` gains keyword-only
  `skills_user_root: Path | None = None` and stores
  `self._skills_user_root = skills_user_root`.
- New helper:

  ```python
  def _skills_catalog(self):
      return discover_skills(
          Path(self.state.workspace), user_root=self._skills_user_root
      )
  ```

- `action_open_help` pushes `HelpScreen(skills=self._skills_catalog())`.
- `_dispatch_command` gains a branch (with the other read-only local commands):

  ```python
  elif name == "skills":
      if args:
          self._show_notice("usage: /skills")
          return
      self.push_screen(SkillsScreen(self._skills_catalog()))
  ```

- `_dismiss_transient_screens` adds `SkillsScreen` to the screen tuple.
- CSS: add `#skills` to the modal selectors that currently style `#help,
  #history` (bordered `$surface`, centered; `max-height: 80%;
  overflow-y: auto;`) so the new overlay matches existing modal styling.

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_skills_tui.py tests/test_commands.py -q`
Expected: all PASS.
Run: `uv run pytest tests/test_tui.py tests/test_w5_interaction_history.py tests/test_tui_visual_refresh.py tests/test_tui_display_regression.py -q`
Expected: all PASS (help/command-palette/`/help` behaviors unchanged; the new
command appears in `SUPPORTED_COMMANDS` and help, which those tests already
iterate over).

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/commands.py src/coding_agent/tui/widgets.py \
  src/coding_agent/tui/app.py tests/test_commands.py tests/test_skills_tui.py
git commit -m "Add skills command and help overlay skills section" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Documentation, full suite, linters, CLI gate

**Files:** Modify `README.md`; verification only otherwise (fix only what a task
above missed).

- [ ] **Step 1: Document the skills mechanism**

In `README.md`, under "Running locally" (or a short new "## Skills" section
before "## Architecture"), add a concise note describing the two discovery
roots (`<workspace>/.coding-agent/skills/<name>/SKILL.md` and
`~/.config/coding-agent/skills/<name>/SKILL.md`), the `SKILL.md` frontmatter
(`description` required; optional `name` and `when_to_use`), that discovered
skills appear in the run system prompt and can be loaded with the `load_skill`
tool, and that `/skills` lists them in the TUI. Keep it short and accurate to
the implementation (no resource-folder or authoring claims).

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q` (with an empty user skills dir or a scratch
`XDG_CONFIG_HOME` if real user skills are installed).
Expected: all PASS. Baseline was 632 passed / 1 skipped; the new files add
discovery/catalog/tool/system-prompt/TUI cases. One timing-sensitive concurrency
test (`test_distinct_path_slow_tools_run_in_parallel_before_same_path_third`)
can flake under a loaded machine — if only it fails, re-run it alone to confirm
it passes; do not change it.

- [ ] **Step 3: Run the linters**

Run: `uv run ruff check src tests`
Run: `uv run ruff format --check src tests`
Expected: both clean.

- [ ] **Step 4: CLI smoke**

Run: `uv run python -m coding_agent.app --help`
Expected: prints usage, exit 0.

- [ ] **Step 5: Commit (only if a fix or the README note is uncommitted)**

```bash
git add README.md
git commit -m "Document the skills mechanism" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

- Storage/format: two roots, workspace-first first-wins; `SKILL.md` YAML
  frontmatter with `description` required; missing/empty description ⇒ skip,
  never crash; frontmatter `name` overrides the directory name. ✅ (Task 1)
- Name safety/containment: `..`/`.`/separators/NUL rejected on scan and load;
  `.resolve()` containment rejects traversal and symlink escapes; nothing
  outside the two roots is read or returned; `load_skill` takes no path. ✅
  (Tasks 1, 4)
- Catalog: deterministic, sorted, `## Available skills` section, empty ⇒ empty
  string (byte-identical system prompt), never includes a body. ✅ (Tasks 2, 3)
- System-prompt seam: `build_system_prompt` gains `skills`; `create_app`
  discovers once and passes it; the context policy already prepends it every
  turn. Empty skill set ⇒ today's exact prompt. ✅ (Task 3)
- `load_skill`: registered like existing tools (schema + registry), read-risk +
  parallel-safe so allowed in all three permission modes; found ⇒ body without
  frontmatter + `name`/`path` metadata; unknown/unsafe ⇒ `ok=False`
  `unknown skill: <name>`; ~16 000-char cap with a fixed truncation note. ✅
  (Task 4)
- TUI: `/skills` command + `SkillsScreen` listing name/description/`when_to_use`
  with a no-skills empty state; help overlay Skills section fed the live
  catalog; no free-form `/<skill-name>` expansion. ✅ (Task 5)
- Governance: TDD task-by-task, one atomic commit per task with the
  `Co-Authored-By` trailer; full suite, `ruff check`, `ruff format --check`,
  and the CLI `--help` gate all green at the end. ✅ (Tasks 1-6)
