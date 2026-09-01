# W5 — Interaction & History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** A real help overlay, composer prompt-history recall with draft
preservation, a workspace-filtered session picker, and a call-history inbox.

**Architecture:** `CommandSuggestion` gains `usage`; `/help` and `?` open a modal
`HelpScreen`. `CodingAgentApp` keeps a capped `prompt_history` list and handles
`Up`/`Down` from the composer. `SessionSelector` filters the cached session list
to the current workspace with a browse-all toggle. A `HistoryScreen` renders
recent `tool_call`/`tool_result`/`approval` records (newest first).

**Tech Stack:** Python 3.11+, Textual 8.2.8.

**Spec:** `docs/superpowers/specs/2026-09-01-coding-agent-feature-roadmap-design.md` §5.

## Global Constraints

- New screens match existing modal styling (bordered, `$surface` background, centered).
- `parse_command` stays pure; history lives in the app.
- Existing 395+ tests stay green.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/tui/commands.py`: `usage` on `CommandSuggestion`; `inbox` command.
- `src/coding_agent/tui/widgets.py`: `HelpScreen`, `HistoryScreen`, `SessionSelector` filter.
- `src/coding_agent/tui/app.py`: `/help`, `/inbox` dispatch; composer history; filter wiring.
- `tests/test_w5_interaction_history.py` (new).

---

## Task 1: Help overlay

**Files:** Modify `src/coding_agent/tui/commands.py`, `src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/app.py`; test `tests/test_w5_interaction_history.py`.

**Interfaces:**
- Produces:
  - `CommandSuggestion(name, description, usage="")`.
  - `_COMMANDS` entries gain usage text.
  - `HelpScreen` modal widget with a `RichLog`/`Static` listing commands + usage + keybindings + permission legend.
  - `parse_command("/help")` unchanged; app dispatch pushes `HelpScreen`.

- [ ] **Step 1: Failing tests**

```python
def test_command_suggestions_carry_usage():
    from coding_agent.tui.commands import command_suggestions
    for entry in command_suggestions(""):
        assert hasattr(entry, "usage")
        assert isinstance(entry.usage, str)
    suggestions = command_suggestions("resume")
    assert suggestions and suggestions[0].name == "resume"
    assert suggestions[0].usage


def test_help_screen_composes_commands():
    from coding_agent.tui.widgets import HelpScreen
    screen = HelpScreen()
    # composing yields at least one Static containing each SUPPORTED_COMMANDS name
    statics = [w for w in screen.compose()]
    rendered = "\n".join(str(w) for w in statics)
    for name in {"help", "undo", "session", "permission"}:
        assert name in rendered
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `commands.py`: add `usage` default to the dataclass; fill `usage` for each `_COMMANDS` entry.
  - `widgets.py`: `HelpScreen(App, ...)` modal with a bordered `Static` containing commands/usage, keybindings (`Ctrl+C abort`, `↑/↓ composer history`, `? help`), and permission legend. Add a `?` binding in `CodingAgentApp.BINDINGS` (`Binding("?", "open_help", "Help")`).
  - `app.py`: `action_open_help` / `/help` dispatch pushes `HelpScreen()`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Add a full help overlay with command usage and keybindings`

---

## Task 2: Composer history ring

**Files:** Modify `src/coding_agent/tui/app.py`, `src/coding_agent/tui/widgets.py`; test `tests/test_w5_interaction_history.py`.

**Interfaces:**
- Produces:
  - `CodingAgentApp.prompt_history: list[str]` (capped 50), `_history_index: int | None`.
  - Composer `Up`/`Down` messages handled by the app: `Up` saves current draft, recalls previous; `Down` recalls newer or restores the draft; arrows do nothing when history is empty.
  - History is appended on submit.

- [ ] **Step 1: Failing tests**

```python
def test_history_cap_and_recall():
    app = CodingAgentApp(runtime=fake_runtime)
    for i in range(60):
        app.prompt_history.append(f"p{i}")
    assert len(app.prompt_history) == 50
    assert app.prompt_history[0] == "p10"


def test_up_recalls_previous_prompt():
    # after two submits, pressing Up recalls the last prompt and preserves draft
```

  Drive the handler methods directly (`app._history_recall(-1)`, `+1`) with a
  stubbed composer. Use a fake runtime (see `tests/fakes.py` or a minimal stub).

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `app.py`: `self.prompt_history = []`, `self._history_index = None`,
    `self._history_draft = ""`. In `on_submit_text_area_submitted`, after a
    successful prompt submit, `self.prompt_history.append(prompt)` and trim to 50.
  - `widgets.py` `SubmitTextArea`: add `Up`/`Down` key handling that posts
    `ComposerHistoryRequested(offset)` via a new message or calls an injected
    callback. Simpler: add an `on_key` handler that, for `Up`/`Down`, calls
    `self.app`-provided callables `self.history_navigate(-1/+1)` set by the app at mount.
  - `app.py` history navigation: on `-1`, if `_history_index is None` stash the
    current text as `_history_draft` and set index to `len-1`; else decrement
    (floor 0). On `+1`, increment; when index reaches `len`, restore draft and
    clear index. Set the composer text accordingly.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Recall composer prompt history with draft preservation`

---

## Task 3: Session selector workspace filter

**Files:** Modify `src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/app.py`; test `tests/test_w5_interaction_history.py`.

**Interfaces:**
- Produces:
  - `SessionSelector(sessions, workspace: str | None, ...)`; a footer toggle `[browse: workspace|all]`; toggling re-filters the cached list.
  - `_open_session_selector` passes the current workspace.

- [ ] **Step 1: Failing tests**

```python
def test_session_selector_filters_by_workspace():
    from coding_agent.session.models import SessionSummary
    from coding_agent.tui.widgets import SessionSelector
    sessions = [
        SessionSummary(id="a", workspace="/w1", created_at=..., updated_at=..., title="t1", last_status="idle"),
        SessionSummary(id="b", workspace="/w2", created_at=..., updated_at=..., title="t2", last_status="idle"),
    ]
    selector = SessionSelector(sessions, workspace="/w1")
    assert [s.id for s in selector.visible_sessions()] == ["a"]
    selector.toggle_filter()
    assert [s.id for s in selector.visible_sessions()] == ["a", "b"]
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `SessionSelector.__init__(sessions, workspace=None)`: store `_workspace`,
    `_browse_all = False`; `visible_sessions()` filters `s.workspace == _workspace`
    unless `_browse_all`. Render a footer with a `browse all` / `current workspace`
    toggle button; clicking toggles and re-renders the option list.
  - `app.py`: pass `self.state.workspace` when constructing `SessionSelector`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Default the session picker to the current workspace with a browse-all toggle`

---

## Task 4: Call-history inbox

**Files:** Modify `src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/commands.py`, `src/coding_agent/tui/app.py`; test `tests/test_w5_interaction_history.py`.

**Interfaces:**
- Produces:
  - `HistoryScreen` modal rendering recent `tool_call`/`tool_result`/`approval` records newest-first (last 20).
  - `parse_command("/inbox")` → `Command("inbox", [])`; app dispatch builds rows from `runtime.store.records()` and pushes `HistoryScreen`.

- [ ] **Step 1: Failing tests**

```python
def test_inbox_builds_rows_from_records():
    app = CodingAgentApp(runtime=fake_runtime_with_records)
    rows = app._inbox_rows()
    assert rows  # contains tool/approval summaries newest first
    # approval record from W2 appears as "approve write_file"


def test_parse_command_inbox():
    from coding_agent.tui.commands import parse_command
    assert parse_command("/inbox").name == "inbox"
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `commands.py`: add `CommandSuggestion("inbox", "Show recent tool calls and approvals", usage="/inbox")`.
  - `app.py`: `_inbox_rows()` reads `self.runtime.store.records()`, filters
    `tool_call` (name + compact args), `tool_result` (status), `approval`
    (decision + tool), sorts by `timestamp` desc, caps at 20, formats strings.
  - `widgets.py`: `HistoryScreen` modal with a bordered `Static`/`RichLog`
    rendering the rows; `/inbox` dispatch pushes it.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Add a call-history inbox surfaced by /inbox`

---

## Self-Review

- Spec §5 covered: help overlay (T1), composer history (T2), workspace filter (T3), inbox (T4). ✅
- W2's `approval` records feed the inbox (T4). Types consistent across tasks. ✅
