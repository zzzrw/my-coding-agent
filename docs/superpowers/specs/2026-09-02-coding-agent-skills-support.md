# Coding Agent Skills Support Design

## Status

Approved on 2026-09-02 by the human owner, distilled from a seven-project
reference study of mainstream agent skill mechanisms. This design is an
increment on the approved MVP design
(`2026-08-30-coding-agent-mvp-design.md`), whose §14 originally listed a skill
loader as a post-MVP item, and it reuses infrastructure introduced by the
feature roadmap (`2026-09-01-coding-agent-feature-roadmap-design.md`): W5's
local slash-command and help-overlay patterns and W6's config-dir helper. It is
implemented as a separate workstream with its own dated plan under
`docs/superpowers/plans/`, following TDD task-by-task.

## 1. Motivation

The reference projects (Pi, `my-pi-agent`, Claude Code, Codex/oh-my-cli, and
opencode-style agents) converge on one pattern for reusable instruction
packages ("skills"): **progressive disclosure**. A cheap, deterministic catalog
is always present so the model knows what is installed; the full body is pulled
in only when a task actually needs it. The alternative — always injecting every
skill body — is explicitly rejected.

Coding-agent mirrors that pattern as a mechanism **separate** from the
always-injected guidance in `build_system_prompt`:

- A small, fixed catalog is rendered into the run system prompt each session, so
  the model sees every skill name and a one-line purpose on every turn.
- A `load_skill` tool returns the full `SKILL.md` body for one named skill on
  demand. The body arrives through the ordinary tool-result path, is recorded in
  the session like any tool result, and is fed back to the model so it can
  follow the skill on the next step.
- Skills are plain files the user authors in one of two roots — no registry, no
  recompilation, and never any auto-injection of full bodies.

## 2. Goal

Let the user author reusable instruction packages (a directory per skill holding
`SKILL.md`) that the agent can enumerate cheaply at the start of every session
and load on demand, from either the current workspace or the user's config home.

## 3. Storage and Format

### 3.1 Discovery roots

Two discovery roots are scanned **workspace-first then user-global**; the first
effective skill name wins:

```text
workspace root:   <workspace>/.coding-agent/skills/<skill-name>/SKILL.md
user-global root: <config_dir>/skills/<skill-name>/SKILL.md
```

- `<workspace>` is the current session workspace, available on the session store
  header (`store.header.workspace`). It is resolved once at launch by
  `_resolve_workspace` in `src/coding_agent/app.py`, stored on every session
  header, and reused by `new_session` / `resume`
  (`src/coding_agent/runtime/runtime.py`), so discovery keyed on the workspace
  is stable across session transitions in one running app.
- `<config_dir>` is the **same config home already resolved by `config_dir()`**
  in `src/coding_agent/config/config.py` — the base that already holds
  `config.toml` (W6) and `approvals.json` (W2). Reuse that helper rather than
  inventing a new path. A `config_dir` override parameter is accepted by the
  discovery functions so tests can point the user root at `tmp_path`.

### 3.2 SKILL.md

A skill is a directory `<skill-name>/` containing `SKILL.md`: YAML frontmatter
delimited by `---` lines, then a free Markdown body.

```text
---
name: conventional-commits   # optional; kebab-case; defaults to the dir name
description: Write Conventional Commits.   # REQUIRED
when_to_use: craft a commit message        # optional
---

Use the Conventional Commits spec. Write a type, scope, and subject...
```

Frontmatter fields:

- `name`: optional, kebab-case. When absent or empty, the directory name is the
  skill's effective name.
- `description`: REQUIRED and must be a usable (non-empty) value. Discovery
  **skips** any skill whose `SKILL.md` lacks a usable description; a bad
  description never crashes discovery or the app.
- `when_to_use`: optional free text, shown in the TUI `/skills` listing.

Body rules: no other frontmatter keys are parsed in v1 — unknown keys are
ignored, never an error. The Markdown body may be empty. The whole file is read
as UTF-8 text; a file that cannot be read or decoded is treated as an invalid
skill (skipped), never an exception.

## 4. Discovery and Precedence

`discover_skills(workspace, *, user_root=None) -> list[Skill]` scans the
workspace root then the user-global root. Within a root it visits skill
directories in sorted order so the result is deterministic. A candidate
directory yields a `Skill` only when it contains a readable `SKILL.md` with a
usable description and passes the §5 containment checks; every other candidate
is skipped silently.

First-wins precedence:

- Workspace candidates shadow user-global candidates with the same effective
  name. A name present only in the user root still appears.
- The returned list is sorted by effective name (deterministic).

`Skill` is a small record carrying `name` (effective), `description`,
`when_to_use` (optional), `path` (the resolved `SKILL.md`), and the root it came
from. `resolve_skill(name, workspace, *, user_root=None) -> Skill | None` is the
single lookup used by both the catalog and `load_skill`, returning the first
matching skill across the two roots (workspace first) so the catalog and the
tool always agree on what is loadable.

## 5. Name Safety and Path Containment

Skill files are only ever resolved under the two roots; nothing else is read.

- Skill names are single path segments. A name containing a path separator
  (`/` or `\`), a NUL byte, or equal to `.` or `..` is unsafe and is rejected —
  both when scanning directories and when a caller asks to load a skill. Unsafe
  names never match a catalog entry and never reach the filesystem as a path.
- Resolution joins the safe name to a root (`root / <name> / "SKILL.md"`), then
  `.resolve()`s the result and requires the final path to live under the
  `.resolve()`d root. An absolute input, a `..` traversal, or a **symlink
  escape** (the skill directory or `SKILL.md` itself being a symlink whose
  target leaves the root) fails containment → the skill is skipped during
  discovery and reported `unknown` by `load_skill`.
- `load_skill` takes no path argument and performs no free filesystem reads; its
  content comes only from a resolved `SKILL.md` under the two roots.

## 6. Deterministic Catalog in the System Prompt

The model always sees the catalog. Discovery is cheap, deterministic, sorted,
and rendered as a fixed section appended to the system prompt used for the run;
full SKILL.md bodies are never auto-injected anywhere.

Assembly point (verified in source): the run system prompt `Message` is built by
`build_system_prompt(workspace, permission_mode)` in `src/coding_agent/app.py`
and is threaded into `AgentRuntime` and every `AgentRunner`
(`src/coding_agent/runtime/runtime.py`, `src/coding_agent/runtime/runner.py`).
The context policy always prepends it as the first message and preserves it on
truncation (`src/coding_agent/context/truncate.py`). Render the catalog into
that same content so every `ContextView` and therefore every model turn carries
it.

Catalog format (deterministic):

```text
## Available skills
- name: description
```

- One `- name: description` line per skill, sorted by name; descriptions are
  collapsed to a single line.
- When no skills are installed the section is omitted entirely (emit nothing),
  so a skill-less run stays byte-identical to today. This is the chosen,
  deterministic empty behavior.
- No SKILL.md body is ever placed in the system prompt.

`build_system_prompt` runs once at app construction and the session workspace is
stable for the app's lifetime, so discovery effectively runs at session start.
Discovery is idempotent and cheap, so an implementation may re-run it on session
load / `new_session` without any behavioral change.

## 7. load_skill Tool

A tool named `load_skill` is registered in the tools registry exactly like the
existing tools, so it is surfaced to the provider through `registry.schemas()`
and executed/validated/bounded by the ordinary `ToolExecutor` path.

Schema:

```text
name:         load_skill
risk_level:   read
is_parallel_safe: true
parameters:   {skill: str}
description:  "Load the body of an installed skill by name. Available skills are
               listed in the system prompt under 'Available skills'; call this
               before acting when the user names a skill or the task matches a
               listed description."
```

The description is kept short on purpose: it points the model at the catalog
(the source of truth) rather than re-listing skills.

Execution semantics:

- Resolve `skill` by name across the two roots (§4, §5), workspace first.
- Not found (or an unsafe name) → `ok=False`,
  `error == "unknown skill: <name>"`.
- Found → `ok=True`; `content` is the SKILL.md **Markdown body** (the YAML
  frontmatter is excluded); `metadata` is `{"name": <skill name>,
  "path": <resolved SKILL.md path>}` where the name is the directory name (the
  frontmatter `name`, when present and valid, overrides it) — the effective name
  used for resolution.
- Bound the body: when `content` exceeds a fixed cap of about 16,000 characters,
  truncate at the cap and append a fixed, deterministic note that the content
  was truncated. (The executor's downstream general output bound of 20,000 chars
  is not relied on; the tool enforces its own tighter cap so a huge skill can
  never blow context.)
- Content comes only from the resolved skill file. The tool takes no path, reads
  nothing else, and never traverses outside the two roots.

Because `load_skill` is a read-risk tool with no `path` argument, the executor's
approval policy allows it in every permission mode without a prompt, and the
mutation journal/memory do not apply. The returned result is recorded in the
session and fed back to the model through the normal tool-result channel, so the
model can follow the skill on its next step.

## 8. TUI: /skills and Help

Add a local `/skills` command following the W5 command and help-overlay
patterns (how `/inbox`, `/help`, `/undo` register; how help content is
authored):

- `src/coding_agent/tui/commands.py`: add a `CommandSuggestion` —
  name `skills`, description `List available skills`, usage `/skills` — to
  `_COMMANDS`. It then appears in the composer command palette and, via
  `command_suggestions`, in the help overlay's Commands section automatically.
- `src/coding_agent/tui/app.py`: dispatch `/skills` to build the catalog for the
  current workspace (`state.workspace` / the runtime store header workspace)
  using the same discovery function as the system prompt, then push a modal that
  lists the discovered catalog: each skill's name, description, and
  `when_to_use` when present. With no skills installed it shows a short
  "no skills installed" overlay, mirroring `HistoryScreen`'s empty state. The
  overlay matches existing modal styling (bordered, `$surface`, centered).
- Help overlay: `help_overlay_text()` (`src/coding_agent/tui/widgets.py`) gains
  a "Skills" section listing the discovered catalog (name + description). The
  pure builder accepts the catalog as an argument and defaults to a
  deterministic "no skills installed" note, so existing no-argument
  constructions stay valid; the app supplies the live catalog when it opens
  help. `/skills` also appears under Commands automatically once registered.
- v1 has **no free-form `/<skill-name>` expansion**: typing an arbitrary skill
  name is not a command. `/skills` and the help overlay are the only skills
  surfaces.

## 9. Scope Exclusions (v1)

Explicitly out of scope; do not build:

- skill resource folders (`scripts/`, `references/`, `assets/`) and executing or
  interpreting them;
- model authoring, editing, deleting, or installing skills (skills are plain
  files authored on disk by the user);
- forked sub-agents that run skills;
- file watchers / hot reload of skill directories;
- remote, org, plugin, or bundled skills; per-session skills;
- token budgets for skill content;
- conditional activation / activation expressions;
- any frontmatter beyond `name`, `description`, `when_to_use`.

## 10. Components

Proposed new `src/coding_agent/skills/` package owning the feature (a tool
factory may instead live under `tools/` and import it; the shared
discovery/safety path is the load-bearing design, not the module home):

| File | Purpose |
|---|---|
| `skills/discovery.py` | discovery roots, SKILL.md frontmatter parse, `discover_skills`, `resolve_skill`, name-safety and path-containment checks |
| `skills/models.py` | `Skill` record (name, description, when_to_use, path, root) |
| `skills/catalog.py` | `format_catalog(skills)`, the `## Available skills` system-prompt section, and the overlay row/help text builders |
| `skills/tool.py` | `make_load_skill_tool(...)` factory following the `tools/filesystem.py` / `tools/search.py` factory pattern |

Changed files:

| File | Change |
|---|---|
| `src/coding_agent/app.py` | register `load_skill` in `_make_registry`; append the catalog section inside `build_system_prompt` |
| `src/coding_agent/config/config.py` | unchanged — reuse `config_dir()` for the user-global skills root |
| `src/coding_agent/tui/commands.py` | `skills` `CommandSuggestion` |
| `src/coding_agent/tui/widgets.py` | "Skills" help section; `/skills` overlay text builder |
| `src/coding_agent/tui/app.py` | `/skills` dispatch; pass the current catalog into the help overlay |
| `tests/*` | new deterministic tests |

The registry mechanism, executor, session store, context policy, and permission
modes are unchanged; one more tool is registered and one more system-prompt
section is emitted.

## 11. Testing and Verification

All tests are fast and deterministic with no network. Fixtures are small
`SKILL.md` trees under `tmp_path` for the workspace root and (via the injected
`config_dir` override) the user-global root.

- Discovery over the two roots: workspace entries and user entries are both
  found; **first-wins precedence** (a workspace skill shadows the same name in
  the user root); invalid candidates (missing `SKILL.md`, no usable description,
  unreadable/undecodable file, malformed frontmatter) are skipped and never
  raise; empty roots yield an empty list.
- Name safety: a name/argument that is `..`, absolute, contains a path
  separator, or escapes the root via a symlink is rejected — nothing outside the
  two roots is ever read or returned.
- Catalog formatting is deterministic and sorted, including the empty case
  (empty string), and never includes a SKILL.md body.
- `load_skill`: returns the Markdown body (frontmatter excluded) for a found
  skill with `metadata["name"]`/`metadata["path"]`; returns `ok=False` with
  `unknown skill: <name>` for an unknown name; content above the ~16,000-char
  cap is truncated with a fixed appended note.
- `/skills` at the TUI-state level: the command is registered in
  `SUPPORTED_COMMANDS`; its data path builds the catalog list from fixture roots
  and renders name/description/`when_to_use`; the empty case renders the
  no-skills state. Mirrors existing command tests; no live model.
- Help overlay mentions skills: `help_overlay_text` given a catalog includes the
  skill names and a Skills heading; `/skills` appears in the Commands list.
- Determinism: identical fixture roots produce byte-identical catalog strings
  and identical `load_skill` outputs across repeated runs.
- Executor path: `load_skill` is allowed in all three permission modes without
  an approval prompt.

Final gates: the full suite (`pytest -q`), `ruff check src tests`,
`ruff format --check src tests`, and `python -m coding_agent.app --help`, plus a
real TUI smoke that loads a sample skill installed in the workspace root.

## 12. Governance and Conventions

Implement under a dated plan in `docs/superpowers/plans/` with small TDD tasks,
each independently committable as one atomic commit (test first → fail →
implement → pass). Commit bodies end with the trailer
`Co-Authored-By: Claude <noreply@anthropic.com>`. Run the full test suite and
ruff before finishing. Never push; keep all work inside the feature worktree.
