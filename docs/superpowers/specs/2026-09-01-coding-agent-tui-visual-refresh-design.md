# Coding Agent TUI Visual Refresh Design

## Status

Approved by user on 2026-09-01 after subagent exploration of the reference
projects (`ref/oh-my-cli`, `ref/minicode`, `ref/pi`, `ref/deer-flow`,
`ref/codex`) and Textual/Rich capabilities. This design covers a focused visual
refresh of the existing TUI transcript, building on the approved
`2026-08-31-coding-agent-tui-mvp-design.md`.

## 1. Goal and Scope

Improve transcript readability to match mature agent TUIs (Claude Code, Codex,
opencode-style):

- Visible vertical separation between transcript rows and between turns.
- A distinct, inset dark-card treatment for user messages.
- Compact Claude-style tool rows: `● Bash(cmd)` header, a 1-2 line output
  preview, and a duration/status footer, with mouse-click expansion to the full
  (already truncated) body.
- Lightweight Markdown rendering for assistant messages so `**bold**`, `#`
  headings, inline code, lists, and fenced code blocks display with styles
  instead of raw markers.

The implementation stays inside the existing single-transcript three-region
layout (`transcript`, `composer`, `statusline`). It does not add themes,
keyboard navigation of history rows, mouse wheel interactions, session trees,
or any backend capability.

## 2. User Experience

### 2.1 Row spacing

Every transcript row gains a bottom margin so consecutive rows no longer touch.
A logical turn starts at a `user` or `local_command` row; the first row of a
new turn gets a larger top margin (or a faint top border) to visually separate
turns. The exact values are tuned in CSS (`margin: 0 0 1 0` per row, `margin:
1 0 1 0` for turn-start rows).

### 2.2 User message card

User rows render as a full-width inset panel:

- A dark background derived from the current theme (`$surface` or a slight
  `lighten`/`darken` of `$background`), a subtle rounded border, and small
  horizontal padding.
- The existing `> ` text prefix is preserved (tests assert it) and shown in
  the panel.
- `local_command` rows keep their `$ ` prefix but may share a muted background
  treatment to distinguish them from model turns.

### 2.3 Compact tool rows

A tool row renders in a Claude-style multi-line form:

```text
● Bash(git push origin main 2>&1 | tail -2)
  ⎿  To github.com:zzzrw/my-coding-agent.git
  ⎿  (1m)
```

- Header: `{glyph} {DisplayName}({command})` where glyph is `●` (running),
  `✓` (success), `✕` (error), `⊘` (cancelled); `run_command` displays as
  `Bash`, other tools display their capitalized name.
- Preview: the first non-empty output line, capped at ~160 characters.
- Footer: `({elapsed})` where elapsed is formatted compactly (`2s`, `1m`,
  `1m 30s`), plus `· truncated` when the stored result was truncated and
  `· exit N` for non-zero exit codes. On error, the footer notes the error
  reason.
- **Click to expand/collapse**: clicking the row toggles between the compact
  form and the full (already line/char-truncated) body. Expansion is mouse-only
  by design; keyboard row navigation is out of scope because making history
  rows focusable would steal Tab focus from the composer.

### 2.5 Statusline

The bottom statusline gains two focused improvements:

- **Git branch detection**: the current git branch of the workspace is read at
  app start and again whenever the workspace changes (on `session_loaded`).
  The read runs asynchronously with a short timeout; non-git workspaces, a
  missing `git`, or a slow repository render `branch -`.
- **Key/value color distinction**: each `key value` field renders its key
  dimmed and its value in the emphasized text style, so `branch main`,
  `model deepseek-chat`, `perm default`, `session abc123`, and
  `ctx 0/128000/128000 (configured)` are readable at a glance. The runtime
  `status` field is colored by state (`running` cyan, `error` red, `aborted`
  dim, `idle` default). Width truncation behaviour is unchanged: the existing
  field-removal logic still operates on plain text lengths, and styling is
  applied to the surviving fields last.

### 2.4 Markdown rendering

Assistant rows render a lightweight styled subset of Markdown:

- `#`/`##`/`###` headings: colored + bold, marker stripped.
- `**bold**`: bold. `*italic*`/`_italic_`: italic.
- `` `inline code` ``: styled with a subtle background/reverse.
- Fenced code blocks (` ``` `): indented, dimmed block, fence markers hidden.
- Lists (`-`, `*`, `1.`): bullet `•` (or number) prefix.
- Links `[text](url)`: text shown, URL dimmed after it.
- Unsupported constructs fall back to literal text; no parse errors escape.

Full GFM (tables, syntax highlighting, nested formatting) is out of scope now;
the converter is isolated so a swap to `rich.markdown.Markdown` remains a
one-place change later.

## 3. Architecture

### 3.1 Per-kind row classes

`TranscriptRow.__init__` passes `classes=f"row row-{item.kind}"`. CSS in
`src/coding_agent/tui/app.py` styles `.row` (spacing) and `.row.user` /
`.row.tool` / etc. (per-kind background/border/color). `width: 1fr` is required
on styled rows because Textual `Static` defaults to shrink-to-content width.

### 3.2 Tool data chain

The command and timing data currently stop at the runner. The chain is
extended so both live runs and resumed sessions render compactly:

- `AgentRunner` `tool_finished` event gains `metadata=result.metadata`
  (already carries `exit_code`, `elapsed_seconds`, `truncated` from
  `run_command`/filesystem tools).
- `TranscriptItem` gains optional fields: `command`, `elapsed_seconds`,
  `truncated`, `exit_code`, `expanded` (default `False`).
- Reducer: on `tool_started`, set `command` from arguments (for `run_command`,
  `arguments["command"]`; otherwise a compact string). On `tool_finished`,
  preserve `command`, set `text` to the body, copy `elapsed_seconds` /
  `truncated` / `exit_code` from metadata, and never reset `expanded`.
- Session projection (`SessionStore.project_messages` and the matching
  `_projected_transcript`) attaches the command and result metadata so resumed
  transcripts keep the compact form.

### 3.3 Expand/collapse

`expanded` lives on each `TranscriptItem`. `_row_text` chooses compact or
expanded rendering based on it. `TranscriptRow` for tool items defines
`on_click`; the app toggles the row's `expanded` flag and schedules a refresh.
Because `render_state` rebuilds rows from state on every snapshot, the toggle
is a pure state flip plus a refresh.

### 3.4 Markdown converter

A new pure function `markdown_to_text(text) -> rich.text.Text` in
`tui/widgets.py` (or a small `tui/markdown.py` module) converts assistant
message text into a styled `Text` using a line-based scanner. It is called only
for `kind == "assistant"` rows. `TranscriptRow` accepts a renderable, so the
styled `Text` is passed to `Static` directly; `item.text` keeps the plain
source. The `_rendered_text` fallback in `TranscriptView` is updated to a
plain-text join so the non-children render path stays consistent.

## 4. Components

| Component | Change |
|---|---|
| `tui/widgets.py` | `TranscriptRow` classes + renderable support + `on_click` on tool rows; `_row_text` compact/expanded tool form; `markdown_to_text`; `_truncate_tool_output` retained for expanded bodies; `format_statusline` returns a styled `rich.text.Text` with dim keys and emphasized values; `detect_git_branch` helper |
| `tui/app.py` | CSS for `.row`, `.row.user`, `.row.tool`, `.row.local_command`; click-to-expand handler routing to state toggle; async git-branch detection on mount and on `session_loaded` |
| `tui/reducer.py` | `tool_started` stores command; `tool_finished` merges metadata, preserves command/expanded |
| `tui/state.py` | `TranscriptItem` new optional fields |
| `runtime/runner.py` | `tool_finished` carries `metadata` |
| `session/store.py` | projection attaches command + metadata to tool messages |
| `tests/*` | regression + new unit/Pilot tests |

## 5. Data Flow

### Tool row, live run

1. Runner emits `tool_started` (`arguments`) then `tool_finished`
   (`content`, `error`, `metadata`).
2. Reducer builds/updates the tool `TranscriptItem` with `command`,
   `elapsed_seconds`, `truncated`, `exit_code`.
3. Widget renders compact (or expanded) form.
4. Click toggles `expanded`; state refresh re-renders the row.

### Tool row, resumed session

1. `project_messages` returns tool messages carrying `command` and metadata.
2. `_projected_transcript` rebuilds `TranscriptItem` with those fields.
3. Rendering is identical to the live path.

### Assistant row

1. Reducer accumulates `assistant_delta` text into the assistant item.
2. Widget calls `markdown_to_text(text)` and renders the styled `Text`.

## 6. Safety and Behavior Rules

- No UI change bypasses `AgentRuntime` or `ToolExecutor`; tool execution and
  approval semantics are untouched.
- `expanded` is a display flag only; it is not persisted.
- The markdown converter must never raise on malformed input; unknown
  constructs render literally.
- Tests asserting plain-text prefixes (`> `, `$ `, `[status]`) remain valid;
  only their rendering styling changes.
- Secret redaction behavior is unchanged; rendered text never includes
  credentials.

## 7. Testing and Verification

- `TranscriptRow` instances carry `row` + kind classes.
- User row card renders (full-width, background) without changing its text
  prefix.
- Tool compact form: header glyph + command, preview line, duration/status
  footer; collapsed vs expanded `_row_text` outputs differ; click toggles.
- Reducer unit tests: `tool_finished` preserves `command` and sets metadata
  fields; re-applying events stays idempotent.
- Runner test: `tool_finished` payload includes `metadata`.
- Projection test: projected tool rows carry command + metadata.
- `markdown_to_text`: bold, heading, inline code, fenced block, list render
  with the expected styles and no raw markers; malformed input does not raise.
- Statusline: `format_statusline` returns a `Text` whose keys are dim and whose
  values are emphasized; the `status` field is colored per state; existing
  `in`/`len`/width-truncation tests still pass unchanged.
- Branch detection: a git workspace populates `git_branch`; a non-git
  workspace renders `branch -`; the value is refreshed on workspace change.
- Final gates: `pytest -q`, `ruff check src tests`,
  `ruff format --check src tests`, `python -m coding_agent.app --help`, plus a
  real TUI smoke confirming spacing, user card, compact tool row, styled
  assistant text, and a statusline with branch + colored key/value fields.
