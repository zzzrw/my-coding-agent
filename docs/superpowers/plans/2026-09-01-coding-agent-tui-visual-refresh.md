# Coding Agent TUI Visual Refresh Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development
> (or executing-plans) to implement this plan. Steps use checkbox (`- [ ]`)
> syntax. Run the focused test, watch it fail, implement, watch it pass, then
> run the full suite before committing each task.

**Goal:** Deliver the approved visual refresh: row spacing, user message cards,
compact click-expandable tool rows, and lightweight Markdown rendering.

**Architecture constraint:** The transcript stays a per-row Static rebuild.
Tool data (command + metadata) must survive both the live event path and the
session projection path so resumed sessions render identically.

**Commits:** Each task commits only its owned files with the message style used
by previous tasks, ending with the `Co-Authored-By: Claude Opus 4.8 (1M
context) <noreply@anthropic.com>` trailer.

---

## File Map

- `src/coding_agent/tui/widgets.py`: row classes, renderable rows, compact tool
  form + click expand, `markdown_to_text`, spacing helpers.
- `src/coding_agent/tui/app.py`: CSS for `.row`/`.user`/`.tool`, click handler.
- `src/coding_agent/tui/reducer.py`: tool command/metadata merge.
- `src/coding_agent/tui/state.py`: `TranscriptItem` optional fields.
- `src/coding_agent/runtime/runner.py`: `tool_finished` metadata payload.
- `src/coding_agent/session/store.py`: projection attaches command/metadata.
- `tests/test_tui_visual_refresh.py` (new): all new coverage.
- Existing `tests/*`: unchanged behavior; existing prefix assertions preserved.

---

## Task 1: Tool data chain (command + metadata)

**Files:** `runtime/runner.py`, `tui/state.py`, `tui/reducer.py`,
`session/store.py`, `tests/test_tui_visual_refresh.py`

- [ ] **Step 1: Failing tests for the data chain**
  - Runner: a `tool_finished` event emitted after a successful tool run carries
    a `metadata` payload containing `exit_code`, `elapsed_seconds`, `truncated`
    (construct a ToolResult with metadata and assert the emitted event).
  - Reducer: after `tool_started` with `run_command` arguments
    `{"command": "ls -la"}`, the tool row's `command` equals `"ls -la"`; after
    `tool_finished` with `content`, `error`, and metadata, the row keeps
    `command`, and sets `elapsed_seconds`, `truncated`, `exit_code`, and
    `text` to the body. `expanded` stays `False`.
  - Reducer: re-applying `tool_finished` for the same call does not duplicate
    the row (idempotent).
  - Projection: after persisting a tool_call/tool_result pair, resumed
    `_projected_transcript` rows carry `command` and metadata fields.
- [ ] **Step 2: Run focused tests and verify failure**
  - `pytest tests/test_tui_visual_refresh.py -q`
- [ ] **Step 3: Implement**
  - `runner.py:247-255` add `metadata=result.metadata` to the `tool_finished`
    payload.
  - `state.py`: add `command: str | None`, `elapsed_seconds: float | None`,
    `truncated: bool | None`, `exit_code: int | None`,
    `expanded: bool = False` to `TranscriptItem`.
  - `reducer.py`: in `tool_started`, `command = _command_text(arguments,
    tool_name)`; store it. In `tool_finished`, preserve `command`, set `text`
    to body, read `elapsed_seconds`/`truncated`/`exit_code` from
    `payload.metadata`, keep `expanded`.
  - `store.py`: `project_messages` tool projection carries `command` (from the
    matched tool_call arguments) and result metadata; `_projected_transcript`
    in `reducer.py` reads them.
- [ ] **Step 4: Run focused tests green, then full suite**
  - `pytest tests/test_tui_visual_refresh.py -q && pytest -q`
- [ ] **Step 5: Commit**
  - Add only `runner.py`, `state.py`, `reducer.py`, `store.py`,
    `test_tui_visual_refresh.py`.
  - Message: `Carry tool command and result metadata through the TUI data chain`

---

## Task 2: Row spacing and per-kind CSS classes

**Files:** `tui/widgets.py`, `tui/app.py`, `tests/test_tui_visual_refresh.py`

- [ ] **Step 1: Failing tests**
  - `TranscriptRow` built for each kind carries `classes` containing `row` and
    `row-<kind>`.
  - App CSS contains rules for `.row` spacing and `.row.user` full-width card
    (`width: 1fr`).
- [ ] **Step 2: Run and verify failure**
- [ ] **Step 3: Implement**
  - `TranscriptRow.__init__`: `classes=f"row row-{item.kind}"`.
  - `app.py` CSS block: `#transcript .row { margin: 0 0 1 0; }`,
    `.row.user { width: 1fr; background: ...; border: round $primary; padding:
    0 1; margin: 1 0 1 0; }`, muted treatment for `.row.local_command`, and a
    turn-start larger top margin for `.row.user`.
- [ ] **Step 4: Run focused + full suite**
- [ ] **Step 5: Commit**
  - Message: `Add per-kind transcript row styling and user message cards`

---

## Task 3: Compact tool rows with click-to-expand

**Files:** `tui/widgets.py`, `tui/app.py`, `tests/test_tui_visual_refresh.py`

- [ ] **Step 1: Failing tests**
  - `_row_text` for a tool item with `command="ls"`, status `running`,
    `expanded=False` renders a header `● Bash(ls)`.
  - Success: header uses `✓`, error uses `✕`, cancelled uses `⊘`.
  - Preview: first non-empty output line appears under the header prefixed
    `⎿  `; lines beyond the preview are absent.
  - Footer: `⎿  ({elapsed})` appears (e.g. `2s`, `1m`); `· truncated` when
    `truncated`; `· exit 2` when `exit_code=2` and not ok.
  - `expanded=True` renders the full truncated body instead of the preview.
  - Pilot: clicking a tool row toggles its `expanded` flag and re-renders.
- [ ] **Step 2: Run and verify failure**
- [ ] **Step 3: Implement**
  - `widgets.py`: `_tool_header(item)`, `_tool_preview(item)`,
    `_tool_footer(item)`; `_row_text` branches on `item.expanded`.
  - `TranscriptRow.on_click` for tool rows posts a message / calls the app to
    toggle; `CodingAgentApp` handler flips `expanded` in state and refreshes.
- [ ] **Step 4: Run focused + full suite**
- [ ] **Step 5: Commit**
  - Message: `Render compact expandable tool rows in the transcript`

---

## Task 4: Lightweight Markdown rendering for assistant rows

**Files:** `tui/widgets.py` (new `markdown_to_text`), `tests/test_tui_visual_refresh.py`

- [ ] **Step 1: Failing tests**
  - `**bold**` renders bold (style contains `bold`) and no literal `**`.
  - `# Heading` renders a colored bold heading with the `#` stripped.
  - `` `inline` `` renders with a distinct style (reverse or background).
  - A fenced code block renders as an indented dimmed block without fences.
  - `- item` renders with `• ` and no leading `-`.
  - `[text](url)` keeps `text` and includes a dimmed `url`.
  - Malformed input (e.g. unbalanced `**`) does not raise and returns text.
  - `TranscriptRow` for an assistant item renders the styled `Text`
    (no raw markers visible in the row renderable).
- [ ] **Step 2: Run and verify failure**
- [ ] **Step 3: Implement**
  - `markdown_to_text(text) -> rich.text.Text` line-based scanner handling
    fenced blocks, ATX headings, list prefixes, inline bold/italic/code/links.
  - Wire into `_row_text`/`TranscriptRow` for `kind == "assistant"`; update
    `_rendered_text` fallback to a plain-text join.
- [ ] **Step 4: Run focused + full suite**
- [ ] **Step 5: Commit**
  - Message: `Render assistant markdown with lightweight styling`

---

## Task 5: Final verification and real TUI smoke

**Files:** only files required by failed checks.

- [ ] **Step 1: Run the complete gates**
  - `pytest -q`, `ruff check src tests`, `ruff format --check src tests`,
    `python -m coding_agent.app --help`.
- [ ] **Step 2: Real TUI smoke**
  - Launch the app with a real or fake provider; verify visually: row spacing,
    user card, compact tool row with click-expand, styled assistant text.
  - Confirm no `DuplicateIds`, no traceback, statusline intact.
- [ ] **Step 3: Report faithfully**
  - Summarize changed files, gate output, and the smoke result. Do not claim
    the smoke passed without observing it.
