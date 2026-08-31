# Coding Agent TUI MVP Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Tasks 13-15 so the coding-agent package exposes a tested Textual TUI, local session commands, and a runnable CLI wired to the existing runtime.

**Architecture:** Keep `AgentRuntime` as the only owner of provider, tools, policy, and session behavior. `CodingAgentApp` owns a bounded runtime-event bridge, immutable `TuiState`, and Textual rendering; command parsing stays local. The top-level CLI constructs all existing dependencies and injects them into the TUI. Work is staged so pure command utilities can be developed independently, while shared `tui/app.py` integration is handled by one agent after the shell contract exists.

**Tech Stack:** Python 3.11+, Pydantic 2, Textual >=0.80, pytest/pytest-asyncio, ruff, existing OpenAI-compatible provider and six local tools.

---

## File Map

- `src/coding_agent/tui/commands.py`: slash-command value model, parser, command names, and local command dispatch helpers; no Textual or runtime internals.
- `src/coding_agent/tui/widgets.py`: transcript rows, compact one-line statusline, approval modal, and session selector widgets; rendering-focused code only.
- `src/coding_agent/tui/app.py`: `CodingAgentApp`, exact three-region layout, runtime queue bridge, reducer application, workers, command handling, approval/selector orchestration.
- `src/coding_agent/app.py`: CLI parser, dependency factory, configuration validation, and Textual launch.
- `tests/test_tui.py`: Pilot tests for shell, event bridge, commands, statusline, modal/selector behavior.
- `tests/test_integration_flow.py`: Fake Provider end-to-end workspace test.
- `README.txt`: assignment-facing concise Chinese run documentation.
- `README.md`: developer-facing setup and architecture notes.
- `docs/superpowers/specs/2026-08-31-coding-agent-tui-mvp-design.md`: approved design contract; do not change unless implementation reveals a genuine contract error.

Existing runtime/session/tool/provider files are consumed, not refactored, unless a focused compatibility fix is required by a failing test.

---

## Chunk 1: Local Command Contract

### Task 1: Implement and test slash-command parsing

**Files:**
- Create: `src/coding_agent/tui/commands.py`
- Modify: `tests/test_tui.py`

- [ ] **Step 1: Write focused failing tests**

Add tests that parse `/permission workspace` into name `permission` and args `["workspace"]`, parse `/resume ab`, treat non-slash text as a prompt, preserve quoted/whitespace-separated arguments according to the chosen simple parser contract, and identify `/unknown` without raising.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_tui.py -q`
Expected: collection or import failure because `commands.py` and the TUI test helper do not yet exist.

- [ ] **Step 3: Implement the minimal local parser**

Define a strict Pydantic or dataclass command value with `name` and `args`, a `parse_command(text)` function that recognizes only leading slash commands, and a supported-command constant covering `/help`, `/new`, `/session`, `/resume`, `/compact`, `/context`, `/permission`, `/clear`, and `/quit`. Parsing must not call runtime methods or persist anything.

- [ ] **Step 4: Run command tests**

Run: `pytest tests/test_tui.py -q`
Expected: command parsing tests pass; unrelated missing-shell tests may remain skipped or fail until later chunks.

- [ ] **Step 5: Commit the isolated command utility**

```bash
git add src/coding_agent/tui/commands.py tests/test_tui.py
git commit -m "Add local TUI command parser" -m "Constraint: Keep slash commands local and out of model history.\nRejected: Runtime-aware parsing and arbitrary command execution.\nConfidence: High; parser follows approved TUI contract.\nScope-risk: Textual dispatch is deferred to the shell task.\nDirective: Preserve exact MVP command names.\nTested: Focused command parser tests.\nNot-tested: Full Textual shell and CLI." 
```

---

## Chunk 2: Textual Shell and Runtime Bridge

### Task 2: Build the fixed three-region shell

**Files:**
- Create: `src/coding_agent/tui/widgets.py`
- Create: `src/coding_agent/tui/app.py`
- Modify: `tests/test_tui.py`

- [ ] **Step 1: Add failing Pilot tests for the shell**

Cover `#transcript`, `#composer`, and `#statusline` existence under `app.run_test()`. Add an injected fake runtime test where Enter calls `runtime.submit("inspect the project")` only while idle. Add Ctrl-C coverage for an active run calling `runtime.abort(run_id)`.

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest tests/test_tui.py -q`
Expected: import/constructor failure because `CodingAgentApp` and widgets do not exist.

- [ ] **Step 3: Implement rendering widgets and exact layout**

Implement a transcript container based on `VerticalScroll`, a `TextArea#composer`, and a fixed `Static#statusline` in that order. Render user/assistant/tool/system `TranscriptItem` rows with stable ids and readable tool status. Keep all session information in the statusline; do not add a sidebar or footer.

- [ ] **Step 4: Implement app state and worker submission**

Make `CodingAgentApp` accept an injected runtime and initial state/fallback metadata. On Enter, trim input, route slash text locally when command handling exists, otherwise schedule `runtime.submit()` in a Textual worker and clear the composer. Reject a second prompt while non-idle with a visible notice. Ensure runtime exceptions become system rows.

- [ ] **Step 5: Implement the bounded runtime event bridge**

Subscribe on mount and unsubscribe on unmount. Feed events into a bounded asyncio queue. Preserve lifecycle, tool, approval, policy, context, and error events; coalesce assistant deltas by message id under saturation. Drain on the Textual event loop, call the pure reducer, refresh transcript/statusline, and auto-scroll only when already at the bottom. Do not call widgets from runtime tasks.

- [ ] **Step 6: Add lifecycle and delta tests**

Test consecutive assistant deltas followed by `run_finished` produce one assistant row containing both fragments and an idle state. Test approval events update `pending_approval` and display the modal shell without executing tools. Test malformed/runtime bridge errors become system rows.

- [ ] **Step 7: Run Task 13 tests**

Run: `pytest tests/test_tui.py -q`
Expected: all shell, bridge, reducer rendering, abort, and approval tests pass.

- [ ] **Step 8: Commit the shell**

```bash
git add src/coding_agent/tui/app.py src/coding_agent/tui/widgets.py tests/test_tui.py
git commit -m "Build Textual TUI shell and runtime bridge" -m "Constraint: TUI communicates only through AgentRuntime and the reducer.\nRejected: Direct widget calls from runtime tasks and persistent sidebar layout.\nConfidence: High; Pilot tests cover the event and lifecycle contract.\nScope-risk: Commands and CLI wiring are separate chunks.\nDirective: Keep transcript, composer, and statusline as the exact three regions.\nTested: Task 13 Textual Pilot tests.\nNot-tested: Real provider credentials and final CLI." 
```

---

## Chunk 3: Commands, Selector, and Statusline Integration

### Task 3: Add local command actions and modals

**Files:**
- Modify: `src/coding_agent/tui/app.py`
- Modify: `src/coding_agent/tui/widgets.py`
- Modify: `src/coding_agent/tui/commands.py`
- Modify: `tests/test_tui.py`

- [ ] **Step 1: Add failing command/action tests**

Test `/permission workspace` calls only `runtime.set_permission("workspace")`, `/new` calls `new_session`, `/compact` calls `compact`, `/context` emits a local notice, `/clear` clears only visible transcript, `/quit` exits, unknown commands emit a notice, and command text never appears in submitted prompts. Test `/session` opens a selector and selecting the second newest session calls `runtime.resume("s2")`. Test ambiguous resume prefixes stay local.

- [ ] **Step 2: Run tests and verify the new behavior fails**

Run: `pytest tests/test_tui.py -q`
Expected: failures for missing action handlers, selector, and command dispatch.

- [ ] **Step 3: Implement command dispatch**

Add async workers for runtime APIs and a local notice helper. Support the exact command list and argument validation. Resolve session prefixes only against `runtime.list_sessions()` results, reject zero/multiple matches, and never turn command strings into model history.

- [ ] **Step 4: Implement the session selector modal**

Use a Textual modal and `OptionList`. Show short id, updated timestamp, bounded title, and workspace; use newest-first summaries. On Enter call only `runtime.resume()` and close on success; convert failures to notices.

- [ ] **Step 5: Implement approval modal actions**

Render tool name, arguments, risk level, and reason. Approve/Deny call only `runtime.resolve_approval(request_id, decision)`. Ctrl-C denies/cancels pending approval without tool execution.

- [ ] **Step 6: Implement compact statusline formatting**

Render workspace directory, git branch or `-`, model, reasoning or `-`, permission, short session id, runtime status, context used/remaining/window, and known usage. Use field-priority truncation/hiding so the row does not wrap on narrow terminals.

- [ ] **Step 7: Run Task 14 tests and the existing suite**

Run: `pytest tests/test_tui.py -q && pytest -q`
Expected: all TUI and existing tests pass.

- [ ] **Step 8: Commit the command integration**

```bash
git add src/coding_agent/tui tests/test_tui.py
git commit -m "Add TUI commands sessions and statusline" -m "Constraint: Local commands and runtime-only session actions remain within MVP boundaries.\nRejected: Command persistence, session trees, and sidebar metadata.\nConfidence: High; command, selector, approval, and statusline tests cover the contract.\nScope-risk: CLI construction remains to be wired.\nDirective: Keep safety decisions in AgentRuntime and ToolExecutor.\nTested: Full pytest suite.\nNot-tested: Configured live provider run." 
```

---

## Chunk 4: CLI and End-to-End Wiring

### Task 4: Build application factory and CLI

**Files:**
- Create: `src/coding_agent/app.py`
- Create: `tests/test_integration_flow.py`
- Create: `README.txt`
- Modify: `README.md`

- [ ] **Step 1: Write the failing Fake Provider integration test**

Use the existing Fake Provider helpers (or add focused test-only helpers) to submit a write-file tool call followed by a run-command call and final assistant text in a temporary workspace. Assert the file contents and `runtime.last_outcome.reason == "completed"`.

- [ ] **Step 2: Run the integration test and verify failure**

Run: `pytest tests/test_integration_flow.py -q`
Expected: import/factory failure until the CLI construction path exists.

- [ ] **Step 3: Implement the application factory**

Resolve workspace, model, base URL, API key environment variable, context window, and session directory. Construct `OpenAICompatibleProvider`, all six built-in tools, `ToolRegistry`, `DefaultApprovalPolicy`, `ToolExecutor`, context policy, `SessionStore`, `AgentRuntime`, and `CodingAgentApp`. Keep provider wire details out of the TUI. Make dependencies injectable for Fake Provider tests.

- [ ] **Step 4: Implement argument parsing and main**

Support `--workspace`, `--model`, `--base-url`, `--session-dir`, and `--context-window`. Make `python -m coding_agent.app --help` work without credentials. On the actual run path, require model and credential with a redacted configuration error; never print secret values. Include the module main guard.

- [ ] **Step 5: Write assignment-facing documentation**

Create `README.txt` under 1,000 Chinese characters with the real public repository URL (`https://github.com/zzzrw/my-coding-agent`), install/run commands, environment-variable credential setup, MVP feature summary, and explicit self-implemented core-agent note. Do not include credentials or identity. Update `README.md` with developer setup and architecture details only.

- [ ] **Step 6: Run integration and CLI checks**

Run:

```bash
pytest tests/test_integration_flow.py -q
python -m coding_agent.app --help
```

Expected: integration PASS and redacted help output with no credential requirement.

- [ ] **Step 7: Commit the CLI wiring**

```bash
git add src/coding_agent/app.py tests/test_integration_flow.py README.txt README.md
git commit -m "Wire runnable coding-agent MVP" -m "Constraint: CLI composes existing boundaries and keeps credentials redacted.\nRejected: Network calls in tests and new framework dependencies.\nConfidence: High; Fake Provider integration and help path are deterministic.\nScope-risk: Real provider behavior remains opt-in manual validation.\nDirective: Start the interactive Textual app only after argument parsing.\nTested: Integration test and CLI help.\nNot-tested: Live API and video acceptance." 
```

---

## Chunk 5: Final Verification and Manual Startup

### Task 5: Verify the complete MVP and launch the TUI

**Files:**
- Modify only files required by failed checks; do not broaden scope.

- [ ] **Step 1: Run the complete automated checks**

```bash
pytest -q
ruff check src tests
ruff format --check src tests
python -m coding_agent.app --help
```

Expected: all tests pass, lint and format checks pass, and help prints without secrets.

- [ ] **Step 2: Check assignment artifacts**

Verify `README.txt` remains below 1,000 Chinese characters, contains the actual public repository URL and no credential/identity data, and that no generated screenshots, keys, or media are staged.

- [ ] **Step 3: Launch the real TUI when configuration is available**

Use a disposable workspace and configured model/key environment. Start `coding-agent` (or `python -m coding_agent.app`) and verify the single transcript, composer, and bottom statusline are visible. Do not persist credentials or include them in output.

- [ ] **Step 4: Report the result faithfully**

Summarize changed files, automated verification output, and whether real TUI startup was completed or blocked by missing external credentials. Do not claim live startup without observing it.
