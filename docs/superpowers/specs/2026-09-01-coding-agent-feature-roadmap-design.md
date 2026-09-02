# Coding Agent Feature Roadmap Design

## Status

Approved for implementation on 2026-09-01. This design covers the full extension
roadmap confirmed by the user (16 items, groups A + B). It builds on the
approved MVP design (`2026-08-31-coding-agent-tui-mvp-design.md`) and the visual
refresh design (`2026-09-01-coding-agent-tui-visual-refresh-design.md`). The
roadmap is split into six workflow groups (W1–W6); each group is implemented,
tested, and merged independently via its own workflow + worktree. This document
is the authoritative design; each group also gets its own plan under
`docs/superpowers/plans/`.

Scope guardrail (from user): demo-reliable quality, no expansion outside the
confirmed list. Explicitly **not** in scope: headless demo driver (`--task`),
offline/mock provider mode, record/replay determinism.

## 0. Roadmap Summary

| Group | Features |
|---|---|
| W1 Real-time feedback | Streaming tool output; statusline running spinner + elapsed timer |
| W2 Approval experience | Approval diff preview; remember decision (once/turn/session/always); deny feedback to model |
| W3 Parallelism & undo | Parallel tool calls (same-file serialized); `/undo`; multi-call plan preview |
| W4 Agent robustness | Tool failure auto-retry; provider heartbeat/idle timeout; progress-loop detection; summary compression |
| W5 Interaction & history | Help panel overlay; composer history ring; session selector workspace filter; call-history inbox |
| W6 Configuration | TOML config file + first-run setup wizard |

## 1. W1 — Real-time feedback (streaming tool output + statusline spinner/elapsed)

### 1.1 Goal

While `run_command` executes, its output appears in the transcript tool row
line-by-line instead of only after completion. The statusline shows a running
spinner and a live elapsed timer whenever the agent is active.

### 1.2 User experience

- During a `run_command` tool call, the tool row's body fills with the command's
  stdout/stderr as it is produced. The compact form's preview line updates too.
- The statusline `status` field shows an animated spinner frame (``⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏``)
  while `status == "running"` and the time since the current run started
  (`12s`; no clock glyph, which overlapped the time in some terminals).
  `waiting_approval` keeps the elapsed time but pauses the spinner.
- A long-running command shows the live output AND the elapsed time; nothing
  blocks until the command finishes.

### 1.3 Architecture

- **Event**: new `RuntimeEvent` type `tool_output_delta` with payload
  `{tool_call_id, text}` (a decoded output chunk).
- **Tool output sink**: `ToolContext` gains an optional
  `on_output: Callable[[str], Awaitable[None]] | None = None`. The shell tool
  streams each `stdout` chunk through `context.on_output` when set, before it is
  appended to the collected buffer. Other tools ignore the sink (their execution
  is atomic). The `Tool` protocol signature is unchanged.
- **Runner wiring**: `AgentRunner` builds a per-call output sink that publishes
  `tool_output_delta` events scoped to the active run/turn, and passes it into
  `ToolContext` via `ToolExecutor.execute(..., output_sink=...)`.
- **Reducer**: `tool_output_delta` appends `text` to the matching tool row's
  `text` (creating the row from `tool_started` data if not present). On
  `tool_finished`, `text` is replaced by the authoritative result content (as
  today). `tool_status` stays `running` while deltas arrive.
- **Bridge coalescing**: `_RuntimeBridge` treats `tool_output_delta` like
  `assistant_delta` for coalescing/buffering (key = `(generation, tool_call_id)`)
  so a burst of chunks cannot stall control events.
- **Statusline spinner + elapsed**: `StatusLine` renders the spinner frame from
  `self.state`-owned counters updated by an app timer. `CodingAgentApp` runs a
  `set_interval` timer while `status == "running"`/`"waiting_approval"` that (a)
  advances the spinner frame index, (b) recomputes elapsed from a stored
  `run_started_at` timestamp, and (c) calls `render_state`. `TuiState` gains
  `run_started_at: float | None`, `spinner_frame: int`.
  `run_started` sets `run_started_at`; `run_finished`/`run_error` clears it.

### 1.4 Components

| Component | Change |
|---|---|
| `runtime/events.py` | `tool_output_delta` event type |
| `tools/registry.py` | `ToolContext.on_output` optional callback |
| `tools/shell.py` | call `context.on_output` per chunk when set |
| `tools/executor.py` | `execute(..., output_sink=None)` threaded into `ToolContext` |
| `runtime/runner.py` | per-call output sink publishing `tool_output_delta` |
| `tui/state.py` | `run_started_at`, `spinner_frame` on `TuiState` |
| `tui/reducer.py` | handle `tool_output_delta` |
| `tui/app.py` | `_RuntimeBridge` coalescing + app timer for spinner/elapsed |
| `tui/widgets.py` | `StatusLine` spinner frame + elapsed rendering |
| `tests/*` | new coverage (see §1.5) |

### 1.5 Testing

- Shell tool streams chunks through `on_output` (fake async collector asserts
  chunk order) and still returns the full combined content.
- Executor forwards `output_sink` into `ToolContext`.
- Runner emits `tool_output_delta` events between `tool_started` and
  `tool_finished` (fake provider + fake executor).
- Reducer accumulates deltas into the tool row; `tool_finished` replaces text.
- Bridge coalesces `tool_output_delta` without reordering control events.
- Statusline: running state renders a spinner frame + elapsed; timer advances
  the frame; elapsed stops after `run_finished`.
- Full gates unchanged.

## 2. W2 — Approval experience (diff preview + remember decision + deny feedback)

### 2.1 Goal

Approving a `write_file`/`edit_file` shows exactly what changes. The user can
remember an approval/denial across scopes, and a denied tool call tells the
model *why* so the agent can correct course.

### 2.2 User experience

- The approval panel for `write_file`/`edit_file` renders a colorized unified
  diff (`-` old lines red/dim, `+` new lines green) with line numbers, truncated
  to ~40 hunk lines with a `… (N more lines)` note. Read-only tools show their
  argument summary (unchanged).
- The approval panel adds a **Remember** selector: `once` (default) | `turn` |
  `session` | `always`. The choice applies to the current decision.
- An optional short feedback input: when the user denies, the text (if any) is
  passed back to the model as the tool result error (e.g.
  `approval denied: use a relative path`). The denial reason is always included.
- A remembered decision short-circuits the next identical call: `always`
  persists across app restarts (user-level allowlist file).

### 2.3 Architecture

- **Remember scopes**:
  - `once`: current behavior (one-time `allow_outside_once` already exists).
  - `turn`: `DecisionMemory` entries cleared when the turn ends.
  - `session`: cleared on `new_session`/`resume`.
  - `always`: persisted to `<config_dir>/approvals.json` (gitignored, mode
    `0600`), loaded at startup.
- **DecisionMemory**: a new `policy/memory.py` module. API:
  `remember(signature, decision, scope)`, `lookup(signature) -> decision | None`,
  `clear_turn()`, `clear_session()`, `load_always()/persist_always()`. A
  signature is `(tool_name, normalized_arguments)` where normalization sorts
  keys and strips nothing else.
- **Policy integration**: `DefaultApprovalPolicy.decide` is unchanged. Before
  calling `policy.decide`, `ToolExecutor.execute` queries the memory for an
  exact-signature hit; a hit short-circuits to allow/deny. A miss proceeds as
  today, and on resolution the chosen decision+scope is recorded.
- **Approval flow extension**: `_ApprovalBroker.resolve(request_id, decision,
  remember="once", feedback=None)` and `AgentRuntime.resolve_approval` gain the
  same params. On `deny`, the executor's returned error becomes
  `approval denied: <reason>[; <feedback>]` (reason from the request).
- **Diff preview**: `widgets.py` gains `render_approval_diff(request) -> Text`
  using `difflib.unified_diff` between the file's current content and the
  proposed content (`arguments["content"]` for write; apply `old_text→new_text`
  for edit). `ApprovalScreen` renders the diff for `risk_level == "mutate_file"`
  tools in a bordered, scrollable `Static`.
- **Record persistence**: `_ApprovalBroker` persists an `approval` record
  (new `SessionRecord` type) with `{request_id, tool_name, decision, scope,
  feedback, tool_call_id}` so the W5 inbox can render it.

### 2.4 Components

| Component | Change |
|---|---|
| `policy/memory.py` (new) | `DecisionMemory` with scopes + always-file persistence |
| `tools/executor.py` | memory lookup before decide; deny feedback into result error; record decisions |
| `runtime/runtime.py` | broker `resolve` gains `remember`/`feedback`; record approvals |
| `tui/widgets.py` | `render_approval_diff`; `ApprovalScreen` remember selector + feedback input + diff view |
| `tui/app.py` | wire new `resolve_approval` params from screen callback |
| `session/models.py` | `approval` record type; `ApprovalRequest` unchanged |
| `tests/*` | new coverage |

### 2.5 Testing

- Memory: remember/lookup per scope; `turn` cleared on turn end; `always`
  persists to/from a temp file with `0600`; signature normalization.
- Executor: remembered allow short-circuits approval (broker not called);
  remembered deny short-circuits; deny with feedback returns
  `approval denied: reason; feedback`.
- Broker/runtime: `resolve` accepts `remember`/`feedback`; an `approval` record
  is appended.
- Diff: write diff shows `+` lines; edit diff reflects `old→new`; missing file
  shows all-added diff; truncation note appears for large diffs.
- TUI: ApprovalScreen shows remember selector + feedback input; choosing deny
  passes feedback. Pilot test for panel content.

## 3. W3 — Parallelism & undo (parallel tool calls + `/undo` + plan preview)

### 3.1 Goal

A turn's tool calls run concurrently when safe (same-file writes stay ordered),
`/undo` restores the most recent write/edit, and a multi-call turn shows a plan
summary before executing.

### 3.2 User experience

- Multiple `read_file`/`grep` calls (parallel-safe) run at once; a
  `write_file`+`read_file` on different paths can overlap. Two mutations of the
  *same* file execute strictly in order.
- `/undo` reverts the most recent successful `write_file`/`edit_file` (restores
  prior content, or removes the file if it did not exist), emits a local notice
  row, and can be repeated to walk back the mutation stack.
- When the assistant emits 2+ tool calls in one response, the transcript shows a
  plan banner before the tool rows: `→ 4 tool calls: write_file(a), write_file(b),
  run_command(c), read_file(d)`, then rows fill in as they start.

### 3.3 Architecture

- **Parallel execution in `AgentRunner`**: after parsing calls, partition into
  ordered *waves*. A wave is a maximal set of calls that can start together: a
  call joins the current wave iff (a) it is `is_parallel_safe` **or** its
  mutation key is not already used in the wave, where the mutation key is the
  resolved path for `write_file`/`edit_file` and the tool name otherwise; calls
  that are neither parallel-safe nor key-distinct start a new wave. Each wave
  runs with `asyncio.gather`; results are collected back into call order before
  the next wave. `signal`, `output_sink`, and event emission stay per-call.
- **Reducer/state with concurrency**: `active_tool_call_id` remains a single
  value (the last-started call is the "active" one); `tool_finished` clears it
  only when it matches. No transcript keying changes — rows key on
  `tool_call_id`.
- **`/undo` journal**: `tools/executor.py` owns a `MutationJournal`
  (list of `(path, original_content_or_None)`). For `risk_level == "mutate_file"`
  tools, before running, the executor resolves the path and reads current
  content (or `None`); on a successful result it pushes the snapshot. `runtime`
  exposes `async def undo()` that pops the journal and restores via atomic write
  (or unlink when original is `None`), then emits a `notice` event with a
  `command="undo <path>"` payload so the transcript shows a local row. Journal
  is in-memory (not persisted across sessions) — demo-reliable.
- **Plan preview**: `AgentRunner`, before executing any calls, emits a
  `plan_preview` event when `len(parsed_calls) >= 2` with payload
  `{tool_calls: [{name, arguments}...]}`. Reducer inserts a `system`/`plan`
  transcript row (kind `system`, styled dim) summarizing the calls. It is
  display-only; no gating.

### 3.4 Components

| Component | Change |
|---|---|
| `runtime/runner.py` | wave partitioner + `asyncio.gather` per wave; `plan_preview` event |
| `runtime/events.py` | `plan_preview` event type |
| `tools/executor.py` | `MutationJournal`; snapshot before + push after mutate_file |
| `runtime/runtime.py` | `undo()` method; notice emission |
| `tui/commands.py` | `undo` command |
| `tui/app.py` | dispatch `/undo` → `runtime.undo()` |
| `tui/reducer.py` | `plan_preview` row; multi-active-tool handling stays per-call |
| `tests/*` | new coverage |

### 3.5 Testing

- Wave partition: parallel-safe reads batch; same-path mutations serialize;
  different-path mutations batch; order of results preserved.
- Parallel execution: two independent slow tools finish before a third
  serialized one (timing test with controlled `asyncio.sleep`).
- Journal/undo: write records original; undo restores content; undo on a
  created file removes it; undo emits a `notice` with `command`.
- Plan preview: 2+ calls → `plan_preview` event; reducer renders a plan row;
  1 call → no preview.
- `/undo` dispatch + runtime method covered; `pytest -q` green.

## 4. W4 — Agent robustness (retry + heartbeat/timeout + loop detection + summary compression)

*Amendment (2026-09-02): the run loop's step budget and W4's loop detection are
superseded by `2026-09-02-coding-agent-unbounded-max-steps.md`. The step budget
is unbounded by default (optional int `max_steps` config), and the no-progress
detector is a windowed, result-aware repetition detector over executed calls
rather than 3 consecutive identical calls. Retry, heartbeat, idle timeout, and
summary compression below are unchanged.*

### 4.1 Goal

The agent recovers from transient tool failures, surfaces liveness and idle
timeouts, warns on no-progress loops, and compacts by summarizing instead of
silently discarding old turns.

### 4.2 User experience

- A tool that fails transiently (timeout, generic error) is retried up to `N`
  times with short backoff before an error is returned to the model. Approval
  denials, cancellations, argument errors, and `old_text must match exactly once`
  are never retried. The transcript notes `· retried 2×` in the tool footer when
  retries occurred.
- If the provider streams nothing for a long idle window, the TUI shows a
  `provider idle` warning notice and the run ends with `provider_timeout`. A
  heartbeat keeps the statusline elapsed timer honest during slow model output.
- If an *executed* tool call repeats unchanged — same tool, same arguments,
  and the same result content/error — at least 3 times within the last 8
  executed calls, the TUI warns `repeated tool call without progress` and the
  run stops with `reason="progress_loop"`. Repetitions count anywhere inside
  the sliding window, so interleaved repeats (`A,B,A,B,A`) trip it, and the
  result is part of the identity: re-reading the same unchanged file/grep 3+
  times trips it, but re-reading a file whose content changed after a write
  does not. This replaces the earlier "≥3 identical calls in a row" wording;
  see the superseding spec
  `2026-09-02-coding-agent-unbounded-max-steps.md`.
- `/compact` summarizes removed turns (model-generated) and prepends the summary
  as a system message instead of dropping them silently. If summarization is
  unavailable/fails, it falls back to the current silent truncation.

### 4.3 Architecture

- **Retry**: `ToolExecutor` gains `max_retries: int = 2`,
  `retry_backoff_seconds: float = 1.0`. `_retryable(result)` returns False when
  `error` matches approval/cancelled/invalid-arguments/exact-match markers.
  Retry loops re-run `_run_tool` (fresh task) with linear backoff; the final
  result carries `metadata["retries"]`.
- **Idle timeout + heartbeat**: `AgentRunner.run_turn` wraps the provider stream
  in an idle watchdog (no event for `provider_idle_timeout_seconds = 90` →
  break with `provider_timeout`). While inside the stream, the runner publishes
  a `heartbeat` event every `15s` (payload `{elapsed_seconds}`) — the reducer
  ignores it except to force a statusline refresh (cheap no-op). The UI-side
  elapsed timer (W1) provides the visual heartbeat.
- **Loop detection** (amended by
  `2026-09-02-coding-agent-unbounded-max-steps.md`): `AgentRunner` keeps a
  bounded sliding window of the last `W = 8` executed-call signatures. A
  signature is a deterministic hash of `(tool_name, canonical sorted-JSON
  arguments, result fingerprint)`, where the result fingerprint hashes the
  persisted result content + error (input capped near 4096 chars). After each
  executed wave the runner inserts that wave's signatures into the window; if
  any signature occurs ≥3 times within the window it emits a `notice` and
  returns `TurnOutcome(reason="progress_loop")`. Consecutive and interleaved
  repeats both trip it; a signature that differs — different tool, different
  arguments, or a changed result — never does.
- **Summary compression**: `context/policy.py` truncation is unchanged.
  `AgentRuntime.compact()`:
  1. computes the removed-turn message span as today,
  2. if a summarizer is available, calls `_summarize(messages)` via
     `provider.stream([...], [], model=..., signal=...)` collecting text deltas,
  3. stores the result in the `compaction` record as `summary`,
  4. `SessionStore.project_messages` prepends a `system` message
     `"Summary of earlier conversation: <summary>"` from the latest compaction
     record when present.
  The runner gets a `summarize_available` flag so tests inject a fake summarizer
  or disable it.
- **Provider protocol**: summarization reuses `LLMProvider.stream` with an empty
  tool list; no protocol change.

### 4.4 Components

| Component | Change |
|---|---|
| `tools/executor.py` | retry loop + `_retryable` + `retries` metadata |
| `runtime/runner.py` | idle watchdog, heartbeat, loop detection |
| `runtime/events.py` | `heartbeat` event type; `provider_timeout`/`progress_loop` reasons |
| `runtime/runtime.py` | compact summarization + fallback |
| `session/store.py` | prepend summary system message from compaction record |
| `tui/state.py`/`reducer.py`/`widgets.py` | tool footer `· retried 2×`; notices for idle/loop; heartbeat no-op |
| `tests/*` | new coverage |

### 4.5 Testing

- Retry: transient error retried and succeeds; non-retryable markers not
  retried; `retries` metadata set; retry count respected.
- Idle timeout: a provider that stalls past the window yields
  `provider_timeout`; heartbeat emitted during slow output.
- Loop: same tool + arguments + unchanged result repeated ≥3 times within the
  window → `progress_loop`, including interleaved repeats; broad distinct
  pure-read exploration and a re-read after a content-changing write never trip
  it. Detailed expectations live in
  `2026-09-02-coding-agent-unbounded-max-steps.md`.
- Summary compression: fake provider returns a summary; compaction record stores
  it; `project_messages` prepends it; summarizer failure falls back to silent
  truncation (no summary, no error).

## 5. W5 — Interaction & history (help overlay + composer history + workspace filter + inbox)

### 5.1 Goal

Better in-TUI discoverability: a real help overlay, composer prompt history with
draft preservation, a workspace-filtered session picker, and a call-history
inbox.

### 5.2 User experience

- `/help` (and a `?` binding) opens a modal **HelpScreen** overlay listing every
  command with its description and usage, plus keybindings and a permissions
  legend. It matches the styling of the approval/session screens.
- In the composer, `↑` recalls the previous submitted prompt (saving any current
  draft first), `↓` moves forward through history; typing edits the recalled
  text. History is bounded (last 50).
- `/session` / bare `/resume` picker lists sessions for the **current workspace**
  by default; a footer toggle (`browse all`) switches to every session. The
  toggle is both a button and a key binding.
- `/inbox` opens a modal listing recent tool calls and approvals (tool, args
  summary, status/decision, elapsed/timestamp), newest first, from the session
  records.

### 5.3 Architecture

- **HelpScreen**: a new modal `Screen` (styled like `ApprovalScreen`) built from
  an expanded `CommandSuggestion` (`description`, `usage`). `_COMMANDS` entries
  gain `usage`. Keybindings section is a small constant list.
- **Composer history**: `CodingAgentApp` keeps `prompt_history: list[str]`
  (capped 50, appended on submit) and `_history_index`. `SubmitTextArea` gains
  an `on_key`/`Up`/`Down` binding that posts a message to the app; the app
  computes draft/recall. When history is empty, arrows do nothing. The current
  draft is stashed when the user navigates away.
- **Workspace filter**: `_open_session_selector` passes the current workspace to
  `SessionSelector`, which renders a footer toggle. Toggling re-filters the
  cached session list (`summary.workspace == workspace`) without re-listing.
- **Inbox**: `_inbox_data()` in the app reads `runtime.store.records()` for
  `tool_call`/`tool_result`/`approval` records (newest first, last 20), and
  builds rows. A `HistoryScreen` modal renders them. `/inbox` command +
  `parse_command` support.

### 5.4 Components

| Component | Change |
|---|---|
| `tui/commands.py` | `help` usage metadata; `inbox`; `undo` (W3); richer `CommandSuggestion` |
| `tui/widgets.py` | `HelpScreen`, `HistoryScreen`; `SessionSelector` workspace filter + toggle |
| `tui/app.py` | `/help`, `/inbox` dispatch; composer history state + up/down handler; workspace filter wiring |
| `tui/state.py` | none required beyond W1 (history lives in the app) |
| `tests/*` | new coverage |

### 5.5 Testing

- Help screen lists every `SUPPORTED_COMMANDS` entry with usage; opening via
  `/help` shows it; `?` binding present.
- Composer history: submit → recall via up; draft preserved; down returns;
  history capped.
- Session selector: default filters to current workspace; toggle shows all.
- Inbox: rows reflect tool/approval records newest-first; `/inbox` dispatch.
- Pilot tests for the new screens.

### 5.6 Extension: skills surfacing (2026-09-02)

The skills feature adds a `/skills` command and a "Skills" help-overlay section.
It builds directly on this group's infrastructure — the `_COMMANDS` /
`command_suggestions` command registry and `HelpScreen` / `help_overlay_text`
help overlay — but is a separate mechanism designed and implemented under its
own dated spec and plan: `2026-09-02-coding-agent-skills-support.md`. Implement
it there, not inside W5.

## 6. W6 — Configuration (config file + first-run wizard)

### 6.1 Goal

A TOML config file supplies model/key/base_url/context-window; on first run with
no configuration anywhere, an interactive wizard walks the user through setup.

### 6.2 User experience

- Config sources (highest to lowest): CLI flags → env vars → workspace
  `.coding-agent.toml` → user `~/.config/coding-agent/config.toml` → defaults.
- First run with no model, no key, and no config file opens an interactive
  **SetupScreen** (Textual form): model name, base URL (optional), API key
  (masked input), context window. On save it writes
  `~/.config/coding-agent/config.toml` (mode `0600`, key never printed/logged)
  and launches the app.
- A `--config <path>` flag overrides the user config path.

### 6.3 Architecture

- **`config/config.py`**: `Config` pydantic model (`model`, `api_key`,
  `base_url`, `context_window`, `permission_mode`), `load_config(user_path,
  workspace) -> Config` merging sources (parse errors are surfaced as clear
  errors, never crash), `save_config(path, config)` with `0600` perms and
  secret-redaction on any echoed text.
- **Resolution**: `create_app` gains `config: Config | None`. `_resolve_model`,
  `_resolve_api_key`, `_resolve_context_window`, and base_url resolution check
  config in the fallback chain (CLI > env > config).
- **SetupScreen**: a Textual modal (like `ConfigurationScreen` but interactive)
  with labeled `Input` widgets and a `Save` action that writes the config then
  continues. `_run_onboarding` runs it instead of the static screen when
  interactively launchable (TTY); non-TTY still prints the guidance text.
- **Approvals file reuse**: W2's `always` decisions persist to
  `<config_dir>/approvals.json` (same directory helper from `config/config.py`).
- **Skills root reuse**: the user-global skills root is
  `<config_dir>/skills/<skill-name>/SKILL.md`, i.e. the same `config_dir()`
  helper; the skills feature is designed in
  `2026-09-02-coding-agent-skills-support.md`.

### 6.4 Components

| Component | Change |
|---|---|
| `config/config.py` (new) | `Config`, `load_config`, `save_config`, config-dir helper |
| `app.py` | config-aware resolution + `--config` flag + `SetupScreen` onboarding |
| `tests/*` | new coverage |

### 6.5 Testing

- Load merge priority (workspace over user over env); malformed TOML → clear
  error; save writes `0600` and redacts the key in any output.
- `create_app` uses config model/key when env absent; CLI still wins.
- SetupScreen exists; onboarding path reaches it when interactive; non-TTY path
  prints guidance.
- W2 approval always-file uses the same config dir.

## 7. Cross-Cutting Rules

- **No secret in repo**: the two exposed DeepSeek keys must never appear in
  files, tests, docs, or commits. Config/approval files are gitignored (add
  `config.toml`, `approvals.json` patterns if needed).
- **No UI bypass**: TUI changes go through `AgentRuntime`/`ToolExecutor`; no
  tool or approval semantics change outside the designed extensions.
- **Pure reducers**: `tui/reducer.py` stays a pure function; all timers/history
  live in the app or widgets.
- **Textual 8.2.8 CSS**: no `lighten()`/`mix()`; colors are hex.
- **Existing behavior preserved**: `project_messages` validation and the session
  record sequence/parent chain are unchanged; new record types are additive.
- Each W group merges to `main` only after `pytest -q`, `ruff check`,
  `ruff format --check`, and `python -m coding_agent.app --help` pass; the final
  group is followed by a real TUI (tmux) smoke of the whole roadmap.

## 8. Verification Plan (all groups)

1. `uv run pytest -q` (expect all suites green, previous 395 tests still pass).
2. `uv run ruff check src tests` and `uv run ruff format --check src tests`.
3. `python -m coding_agent.app --help` exits 0.
4. Real TUI smoke in tmux with a fake/live provider: streamed command output,
   spinner + elapsed, approval diff + remember + deny feedback, `/undo`,
   parallel calls, `/help` overlay, composer history, workspace-filtered
   `/session`, `/inbox`, config wizard, `/compact` summary.
