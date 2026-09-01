# Ctrl+C Exit Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `ctrl+c` never exits on the first press. While idle, the first `ctrl+c` clears any composer draft (when present) and arms an exit confirmation (with a `Press ctrl+c again to exit` notice); a second `ctrl+c` performs the shutdown. Editing the draft or starting a new run disarms. Abort/approval behavior while a run is active is unchanged.

**Architecture:** Add a `_exit_armed` flag on the app. `action_interrupt`'s idle branch clears the draft (when non-empty) then, if armed, shuts down, otherwise arms and shows a notice. The composer's text-changed handler disarms when a non-empty draft is typed; `run_started` events disarm too. The existing `Binding("ctrl+c", "interrupt", "Abort", priority=True)` and the non-idle branches stay untouched.

**Tech Stack:** Python 3.11+, Textual 8.2.8.

**Spec:** `docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md` §11 (Runtime Events and TUI).

## Global Constraints

- The active-run abort and pending-approval dismiss paths of `action_interrupt` are unchanged.
- No reducer/state changes; the flag is app-interaction state only.
- Existing tests stay green; `ruff check` and `ruff format --check` clean.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/tui/app.py` (modify): `_exit_armed`, `action_interrupt` idle branch, disarm on `run_started`.
- `src/coding_agent/tui/widgets.py` (modify): `SubmitTextArea.on_text_area_changed` disarms when a non-empty draft is typed.
- `tests/test_tui.py` (modify): adjust/add interrupt tests.

---

### Task 1: Two-press exit with armed confirmation

**Files:** Modify `src/coding_agent/tui/app.py`; test `tests/test_tui.py`.

**Interfaces:**
- Consumes: `CodingAgentApp`, `FakeRuntime`, `make_state`, `TextArea`, `pytest`.
- Produces: `CodingAgentApp._exit_armed: bool`; idle `ctrl+c` clears the draft and arms, second `ctrl+c` exits; a `Press ctrl+c again to exit` notice on first arm.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tui.py` (next to `test_idle_ctrl_c_clears_text_then_exits_on_empty_composer` at ~line 1297):

```python
@pytest.mark.asyncio
async def test_idle_ctrl_c_on_empty_composer_requires_two_presses() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        # First ctrl+c on an empty composer arms the confirmation; no exit.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running
        assert app._exit_armed is True
        assert app.state.transcript[-1].text == "Press ctrl+c again to exit"

        # Second ctrl+c exits.
        await pilot.press("ctrl+c")
        await pilot.pause()

    assert not app.is_running
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_tui.py::test_idle_ctrl_c_on_empty_composer_requires_two_presses -q`
Expected: FAIL — the first `ctrl+c` on an empty composer currently exits immediately, so the app stops running and the assertion `app.is_running` fails.

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/app.py` `__init__` (near `self._submitted_run_id = None`, ~line 297):

```python
        self._exit_armed: bool = False
```

Replace the idle branch of `action_interrupt` (currently `elif self.state.status == "idle":` …):

```python
        elif self.state.status == "idle":
            composer = self.query_one("#composer-input", SubmitTextArea)
            if composer.text:
                composer.text = ""
                self.state = self.state.model_copy(update={"input_text": ""})
            if self._exit_armed:
                self._request_shutdown()
            else:
                self._exit_armed = True
                self._show_notice("Press ctrl+c again to exit", level="notice")
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_tui.py -k "ctrl_c" -q`
Expected: all `ctrl_c` tests PASS, including the existing `test_idle_ctrl_c_clears_text_then_exits_on_empty_composer` (first press clears + arms, second press exits) and the new two-press test.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/app.py tests/test_tui.py
git commit -m "Require two ctrl+c presses to exit while idle"
```

---

### Task 2: Disarm the exit confirmation on typing or a new run

**Files:** Modify `src/coding_agent/tui/app.py`, `src/coding_agent/tui/widgets.py`; test `tests/test_tui.py`.

**Interfaces:**
- Consumes: `SubmitTextArea`, `CodingAgentApp._exit_armed`, `RuntimeEvent`.
- Produces: `SubmitTextArea.on_text_area_changed` resets `_exit_armed` when a non-empty draft is typed; `_apply_event` resets it on `run_started`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tui.py`:

```python
@pytest.mark.asyncio
async def test_typing_disarms_exit_confirmation() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        composer = pilot.app.query_one("#composer-input", TextArea)
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._exit_armed is True

        # Typing a draft disarms the confirmation.
        composer.text = "x"
        await pilot.pause()
        assert app._exit_armed is False

        # Next ctrl+c clears the draft and re-arms instead of exiting.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert composer.text == ""
        assert app.is_running
        assert app._exit_armed is True

        await pilot.press("ctrl+c")
        await pilot.pause()

    assert not app.is_running


@pytest.mark.asyncio
async def test_run_start_disarms_exit_confirmation() -> None:
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=make_state())

    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._exit_armed is True

        app._apply_event(
            RuntimeEvent(
                type="run_started",
                run_id="r",
                turn_id="t",
                payload={"session_id": "s", "model": "fake", "policy": "default"},
            )
        )
        assert app._exit_armed is False
```

(Import `RuntimeEvent` at the top of `tests/test_tui.py` if not already imported.)

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_tui.py -k "disarm" -q`
Expected: both FAIL — typing does not reset the flag yet, and `run_started` does not reset it.

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/widgets.py`, `SubmitTextArea.on_text_area_changed` (currently ~line 188):

```python
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self:
            self._update_command_palette()
            # A non-empty draft means the user moved on; disarm the exit
            # confirmation. (The programmatic clear in action_interrupt sets
            # text to "", which must NOT disarm — the clear is the first press.)
            if self.text:
                app = self.app
                if getattr(app, "_exit_armed", False):
                    app._exit_armed = False
```

In `src/coding_agent/tui/app.py`, `_apply_event`: reset the flag when a new run
starts. Add near the top of `_apply_event` (after the existing event guards, or
in the `run_started` handling):

```python
        if event.type == "run_started":
            self._exit_armed = False
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_tui.py -k "ctrl_c or disarm" -q`
Expected: all PASS.

- [ ] **Step 5: Run the full suite and linters**

Run: `pytest -q` then `ruff check src tests` then `ruff format --check src tests`
Expected: 544 + new tests pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/coding_agent/tui/app.py src/coding_agent/tui/widgets.py tests/test_tui.py
git commit -m "Disarm the exit confirmation when typing or starting a run"
```

---

## Self-Review

- Spec §11 exit confirmation covered: two-press idle exit (T1), disarm on typing/new run (T2). ✅
- Active-run abort and approval-dismiss paths untouched. ✅
- No reducer/state/model changes. ✅
- Existing `test_idle_ctrl_c_clears_text_then_exits_on_empty_composer` still passes (clear + arm on press 1, exit on press 2). ✅
