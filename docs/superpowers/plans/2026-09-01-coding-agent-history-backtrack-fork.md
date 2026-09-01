# History Backtracking (Fork) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let the user rewind to any earlier user message and continue from there with a fork: Esc-Esc (empty composer, idle) opens a rewind picker; Enter forks a new session truncated at the selected message, refills the composer with that prompt (not auto-submitted), and leaves the original session untouched.

**Architecture:** `AgentRuntime.fork_at(message_id)` creates a new `SessionStore` whose records end at the selected `user_message` record, swaps the runtime onto it (like `resume`), emits `session_loaded`, and returns the prompt text. The TUI drives it: `SubmitTextArea` detects double-Esc (800 ms window, empty text, no palette) and posts `RewindRequested`; the app opens a `RewindPicker` modal (user messages with preview + relative time); on selection it calls `fork_at` and refills the composer. `TranscriptItem` gains a `timestamp` so the picker can show relative time.

**Tech Stack:** Python 3.11+, Textual 8.2.8, pydantic.

**Spec:** `docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md` §11 (Runtime Events and TUI).

## Global Constraints

- The original session file is never modified by a fork; `fork_at` only appends to a NEW store.
- Fork is rejected while a run is active or an approval is pending (idle-only).
- Single-Esc behavior is unchanged (palette close; otherwise falls through); only a double-Esc on an empty composer triggers the picker.
- No changes to event payload protocols; `session_loaded` is reused as-is.
- Existing 547 tests stay green; `ruff check` and `ruff format --check` clean.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/runtime/runtime.py` (modify): `fork_at`, `_find_user_message_record`.
- `src/coding_agent/tui/state.py` (modify): `TranscriptItem.timestamp`.
- `src/coding_agent/tui/reducer.py` (modify): capture `event.timestamp` on user rows.
- `src/coding_agent/tui/widgets.py` (modify): `RewindPicker`, `SubmitTextArea.RewindRequested` + double-Esc, `_relative_time`.
- `src/coding_agent/tui/app.py` (modify): rewind handlers (`_rewind_rows`, `_rewind_selected`, `_fork_from_message`).
- `tests/test_history_backtrack.py` (new): all new tests.

---

### Task 1: `AgentRuntime.fork_at(message_id)`

**Files:** Modify `src/coding_agent/runtime/runtime.py`; test `tests/test_history_backtrack.py` (new).

**Interfaces:**
- Consumes: `self.store` (`SessionStore`), `SessionRecord`, `Message`, `RuntimeStatus`, `RuntimeEvent`.
- Produces: module helper `_find_user_message_record(records, message_id) -> tuple[int, SessionRecord] | None`; `AgentRuntime.fork_at(message_id) -> str` (returns the prompt text).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_history_backtrack.py`:

```python
import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import Message, TurnOutcome
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore


class _NoopRunner:
    async def run_turn(self, prompt, *, run_id, turn_id, signal):
        return TurnOutcome(reason="completed", final_text=prompt, steps=1)


def _store_with_turns(root, *, turns: int = 2) -> SessionStore:
    store = SessionStore.create(
        root, workspace=str(root), model="fake", context_window=1000
    )
    for index in range(turns):
        tid = f"t{index + 1}"
        store.append_new("turn_start", {"turn_id": tid}, run_id=f"r{index + 1}", turn_id=tid)
        store.append_new(
            "user_message",
            {"message": Message(role="user", content=f"prompt {index + 1}")},
            run_id=f"r{index + 1}",
            turn_id=tid,
        )
        store.append_new(
            "turn_end", {"reason": "completed", "turn_id": tid}, run_id=f"r{index + 1}", turn_id=tid
        )
    return store


def _runtime(store: SessionStore) -> AgentRuntime:
    return AgentRuntime(
        store=store,
        runner_factory=lambda *_: _NoopRunner(),
        context_policy_factory=lambda: TruncatePolicy(1000),
        approval_policy=DefaultApprovalPolicy(),
        system_prompt=Message(role="system", content="s"),
        model="fake",
    )


@pytest.mark.asyncio
async def test_fork_at_creates_truncated_new_session_and_returns_prompt(tmp_path):
    store = _store_with_turns(tmp_path)
    original_path = store.path
    original_text = original_path.read_text(encoding="utf-8")
    runtime = _runtime(store)
    original_session_id = runtime.session_id

    prompt = await runtime.fork_at("user-t1")

    assert prompt == "prompt 1"
    assert runtime.session_id != original_session_id
    assert [r.type for r in runtime.store.records()] == [
        "turn_start",
        "user_message",
        "turn_end",
    ]
    # Original session file is untouched and still holds every record.
    assert original_path.read_text(encoding="utf-8") == original_text
    # New session file is truncated at the fork point.
    new_text = runtime.store.path.read_text(encoding="utf-8")
    assert "prompt 1" in new_text
    assert "prompt 2" not in new_text


@pytest.mark.asyncio
async def test_fork_at_unknown_message_raises(tmp_path):
    store = _store_with_turns(tmp_path)
    runtime = _runtime(store)
    with pytest.raises(ValueError):
        await runtime.fork_at("user-nope")


@pytest.mark.asyncio
async def test_fork_at_latest_message_works_with_open_turn(tmp_path):
    store = _store_with_turns(tmp_path, turns=1)
    runtime = _runtime(store)
    prompt = await runtime.fork_at("user-t1")
    assert prompt == "prompt 1"
    assert [r.type for r in runtime.store.records()] == [
        "turn_start",
        "user_message",
        "turn_end",
    ]
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_history_backtrack.py -q`
Expected: all three FAIL (`AgentRuntime.fork_at` does not exist → AttributeError).

- [ ] **Step 3: Implement**

In `src/coding_agent/runtime/runtime.py`, add a module-level helper (near the top-level helpers) and the method.

Helper:

```python
def _find_user_message_record(
    records: list[SessionRecord], message_id: str
) -> tuple[int, SessionRecord] | None:
    """Locate the ``user_message`` record for ``user-<turn_id>``, or None."""
    prefix = "user-"
    if not message_id.startswith(prefix):
        return None
    turn_id = message_id[len(prefix):]
    for index, record in enumerate(records):
        if record.type == "user_message" and record.turn_id == turn_id:
            return index, record
    return None
```

(Add `SessionRecord` to the existing `from coding_agent.session.store import SessionStore` import if `SessionRecord` is imported from `session.models` instead — use whatever import already brings `SessionRecord` into scope; if none, import it from `coding_agent.session.models`.)

Method (place next to `resume`):

```python
    async def fork_at(self, message_id: str) -> str:
        """Fork the current session at a past user message.

        Creates a new session whose persisted records end at that user
        message, swaps the runtime onto it, and returns the prompt text so the
        TUI can refill the composer. The original session is untouched.
        """
        await self._reserve_operation()
        try:
            found = _find_user_message_record(self.store.records(), message_id)
            if found is None:
                raise ValueError("message not found in session history")
            index, record = found
            prompt = Message.model_validate(record.payload["message"]).content or ""
            prefix = self.store.records()[: index + 1]
            new_store = SessionStore.create(
                self.store.path.parent,
                workspace=self.store.header.workspace,
                model=self._model,
                context_window=self.store.header.context_window,
                title=f"Fork of {self.store.session_id[:8]}",
            )
            for item in prefix:
                new_store.append(item)
            self.store = new_store
            self._permission_mode = "default"
            self._last_outcome = None
            self._status = RuntimeStatus(
                context_window=new_store.header.context_window
            )
            self._runner = self._make_runner()
            await self._publish(
                RuntimeEvent(
                    type="session_loaded",
                    payload={
                        "session_id": self.session_id,
                        "workspace": new_store.header.workspace,
                        "model": new_store.header.model,
                        "context_window": new_store.header.context_window,
                        "history": [
                            item.model_dump(mode="json")
                            for item in new_store.project_messages(
                                include_open_turn=False
                            )
                        ],
                    },
                )
            )
            return prompt
        finally:
            self._release_operation()
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_history_backtrack.py tests/test_runtime.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/runtime/runtime.py tests/test_history_backtrack.py
git commit -m "Add runtime fork_at to create a truncated session fork"
```

---

### Task 2: Timestamp user rows in the transcript

**Files:** Modify `src/coding_agent/tui/state.py`, `src/coding_agent/tui/reducer.py`; test `tests/test_history_backtrack.py`.

**Interfaces:**
- Consumes: `TranscriptItem`, `RuntimeEvent`, `reduce`, `initial_state`.
- Produces: `TranscriptItem.timestamp: datetime | None = None`; the `user_message` reducer branch stores `event.timestamp` on the row.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_history_backtrack.py`:

```python
from datetime import UTC, datetime

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import initial_state


def test_user_message_row_carries_timestamp() -> None:
    stamp = datetime.now(UTC)
    event = RuntimeEvent(
        type="user_message",
        run_id="r",
        payload={"message_id": "user-t1", "text": "hi"},
        timestamp=stamp,
    )
    state = reduce(initial_state(workspace="/tmp/project", model="fake"), event)
    row = state.transcript[-1]
    assert row.kind == "user"
    assert row.timestamp == stamp
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_history_backtrack.py -q`
Expected: the new test FAILS — `TranscriptItem` has no `timestamp` field (pydantic ValidationError).

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/state.py`, add the `datetime` import and the field:

```python
from datetime import datetime
```

In `TranscriptItem` (after `started_at: float | None = None`):

```python
    timestamp: datetime | None = None
```

In `src/coding_agent/tui/reducer.py`, the `user_message` branch:

```python
            _append_or_update(
                transcript,
                TranscriptItem(
                    kind="user",
                    item_id=message_id,
                    text=_text(payload.get("text")),
                    timestamp=event.timestamp,
                ),
            )
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_history_backtrack.py tests/test_reducer.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/state.py src/coding_agent/tui/reducer.py tests/test_history_backtrack.py
git commit -m "Stamp user transcript rows with their event timestamp"
```

---

### Task 3: `RewindPicker` modal and relative-time helper

**Files:** Modify `src/coding_agent/tui/widgets.py`; test `tests/test_history_backtrack.py`.

**Interfaces:**
- Consumes: `ModalScreen`, `OptionList`, `Option`, `datetime`, `UTC`, `timedelta`.
- Produces: `RewindPicker(ModalScreen[str | None])` (compose yields a title, an `OptionList` of user messages keyed by message id, and a hint line; `enter`/selection dismisses with the selected id, `escape` dismisses `None`); `_relative_time(value: datetime | None) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_history_backtrack.py`:

```python
from datetime import UTC, datetime, timedelta

from coding_agent.tui.widgets import RewindPicker, _relative_time


def test_relative_time_formats() -> None:
    now = datetime.now(UTC)
    assert _relative_time(None) == "-"
    assert _relative_time(now - timedelta(seconds=30)) == "30s ago"
    assert _relative_time(now - timedelta(minutes=5)) == "5m ago"
    assert _relative_time(now - timedelta(hours=2)) == "2h ago"


@pytest.mark.asyncio
async def test_rewind_picker_selects_message_id() -> None:
    from coding_agent.tui.app import CodingAgentApp

    rows = [("user-1", "first prompt", "2m ago"), ("user-2", "second", "-")]

    class Host(CodingAgentApp):
        def __init__(self) -> None:
            super().__init__(
                runtime=_FakeRuntime(),
                initial_state=initial_state("/tmp/project", "fake"),
                branch_detector=lambda workspace: None,
            )

        def on_mount(self) -> None:
            self.push_screen(RewindPicker(rows), callback=self._picked)

        def _picked(self, value: str | None) -> None:
            self._picked_value = value

    app = Host()
    async with app.run_test():
        picker = app.screen
        assert isinstance(picker, RewindPicker)
        await app.press("down", "enter")
        await app.pause()
        assert app._picked_value == "user-2"
```

(Define `_FakeRuntime` at the top of the test module, before this test; it needs `status`, `subscribe`, and `fork_at`. Its `fork_at` returns a canned prompt.)

Add the `_FakeRuntime` helper (used by Task 3 and Task 4) at the top of the file, after the imports:

```python
class _FakeRuntime:
    def __init__(self) -> None:
        self.status: str | None = None
        self.forked: list[str] = []

    def subscribe(self, callback):
        return lambda: None

    async def fork_at(self, message_id: str) -> str:
        self.forked.append(message_id)
        return f"restored:{message_id}"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_history_backtrack.py -q`
Expected: the relative-time and picker tests FAIL (no `_relative_time`, no `RewindPicker`).

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/widgets.py`, add `timedelta` to the `datetime` import (line 9):

```python
from datetime import UTC, datetime, timedelta
```

Add `_relative_time` near the other row helpers (e.g. near `_short_id`):

```python
def _relative_time(value: datetime | None) -> str:
    """Human relative time for a transcript row, or '-' when unknown."""
    if value is None:
        return "-"
    seconds = max(0, int((datetime.now(UTC) - value).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"
```

Add `RewindPicker` (place near the other `ModalScreen` subclasses, e.g. after `HistoryScreen`):

```python
class RewindPicker(ModalScreen[str | None]):
    """Modal picker of past user messages to fork from.

    ``rows`` is a list of ``(message_id, preview, relative_time)`` tuples. Enter
    or selecting an option dismisses with the chosen message id; Escape
    dismisses with ``None``.
    """

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        super().__init__()
        self.rows = rows

    def compose(self) -> ComposeResult:
        with Container(id="rewind-picker"):
            yield Static("Rewind — pick a message to fork from", id="rewind-picker-title")
            yield OptionList(
                *[
                    Option(f"{preview}  {reltime}", id=message_id)
                    for message_id, preview, reltime in self.rows
                ],
                id="rewind-picker-options",
                markup=False,
            )
            yield Static("↑↓ select · Enter fork · Esc cancel", id="rewind-picker-hint")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        event.stop()
        self.dismiss(event.option.id)
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_history_backtrack.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/widgets.py tests/test_history_backtrack.py
git commit -m "Add a RewindPicker modal for choosing a fork point"
```

---

### Task 4: Wire double-Esc and the fork flow into the app

**Files:** Modify `src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/app.py`; test `tests/test_history_backtrack.py`.

**Interfaces:**
- Consumes: `SubmitTextArea`, `RewindPicker`, `CodingAgentApp`, `_FakeRuntime`.
- Produces: `SubmitTextArea.RewindRequested` message; double-Esc detection in `SubmitTextArea._on_key` (empty text, no palette, 800 ms window); app handlers `_rewind_rows()`, `on_submit_text_area_rewind_requested`, `_rewind_selected`, `_fork_from_message` (refills the composer with the returned prompt).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_history_backtrack.py`:

```python
import time as _time

from coding_agent.tui.widgets import SubmitTextArea, RewindPicker, TranscriptView, TranscriptRow


@pytest.mark.asyncio
async def test_double_escape_opens_rewind_picker() -> None:
    from coding_agent.tui.app import CodingAgentApp
    from coding_agent.tui.state import initial_state

    app = CodingAgentApp(
        runtime=_FakeRuntime(),
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test() as pilot:
        # Seed a user message so the picker has a row.
        app._apply_event(
            RuntimeEvent(
                type="user_message",
                run_id="r",
                payload={"message_id": "user-t1", "text": "hello"},
            )
        )
        await pilot.pause()
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = ""
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RewindPicker)


@pytest.mark.asyncio
async def test_rewind_selected_forks_and_refills_composer() -> None:
    from coding_agent.tui.app import CodingAgentApp
    from coding_agent.tui.state import initial_state

    runtime = _FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        await pilot.pause()
        await app._rewind_selected("user-t1")
        await pilot.pause()
        assert runtime.forked == ["user-t1"]
        assert composer.text == "restored:user-t1"


@pytest.mark.asyncio
async def test_rewind_rejected_while_run_active() -> None:
    from coding_agent.tui.app import CodingAgentApp
    from coding_agent.tui.state import initial_state

    runtime = _FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test() as pilot:
        app._apply_event(
            RuntimeEvent(
                type="run_started",
                run_id="r",
                turn_id="t",
                payload={"session_id": "s", "model": "fake", "policy": "default"},
            )
        )
        await pilot.pause()
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = ""
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, RewindPicker)
        assert runtime.forked == []
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_history_backtrack.py -q`
Expected: the three Task 4 tests FAIL (no `RewindRequested`, no double-Esc detection, no app handlers).

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/widgets.py`, `SubmitTextArea`:

Add the message class and the last-escape anchor in `__init__`:

```python
class SubmitTextArea(TextArea):
    """TextArea that submits plain Enter and preserves modified Enter keys."""

    class Submitted(Message):
        def __init__(self, text_area: SubmitTextArea, text: str) -> None:
            self.text_area = text_area
            self.text = text
            super().__init__()

    class ComposerHistoryRequested(Message):
        def __init__(self, offset: int) -> None:
            self.offset = offset
            super().__init__()

    class RewindRequested(Message):
        """Double-Esc while idle with an empty composer: open the rewind picker."""
```

In `__init__`, add `self._last_escape_at: float = 0.0` (find the existing `__init__` and add the attribute).

In `_on_key`, after the palette-visible escape branch, add:

```python
        if event.key == "escape" and not self._palette_visible:
            now = time.monotonic()
            if self.text == "" and now - self._last_escape_at <= 0.8:
                event.stop()
                event.prevent_default()
                self._last_escape_at = 0.0
                self.post_message(self.RewindRequested())
                return
            self._last_escape_at = now
```

In `src/coding_agent/tui/app.py`, add the handlers (near the other composer handlers, e.g. near `_composer_history`):

```python
    def on_submit_text_area_rewind_requested(self, event) -> None:
        event.stop()
        if self.state.status != "idle":
            self._show_notice("A run is active; cannot rewind")
            return
        rows = self._rewind_rows()
        if not rows:
            self._show_notice("No earlier user messages to rewind to")
            return
        self.push_screen(RewindPicker(rows), callback=self._rewind_selected)

    def _rewind_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for item in self.state.transcript:
            if item.kind == "user":
                preview = item.text[:60] or "(empty prompt)"
                rows.append((item.item_id, preview, _relative_time(item.timestamp)))
        return rows

    def _rewind_selected(self, message_id: str | None) -> None:
        if message_id:
            self.run_worker(
                self._fork_from_message(message_id),
                name="rewind-fork",
                group="runtime",
                exit_on_error=False,
            )

    async def _fork_from_message(self, message_id: str) -> None:
        try:
            prompt = await self.runtime.fork_at(message_id)
            composer = self.query_one("#composer-input", SubmitTextArea)
            composer.text = prompt
        except Exception as exc:  # noqa: BLE001 - user-facing fork errors
            self._show_notice(f"rewind failed: {exc}", level="error")
```

(Ensure `RewindPicker` and `_relative_time` are imported into `app.py` — add them to the existing `from coding_agent.tui.widgets import (...)` import list.)

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_history_backtrack.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the full suite and linters**

Run: `pytest -q` then `ruff check src tests` then `ruff format --check src tests`
Expected: 547 + ~13 new test cases pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/coding_agent/tui/widgets.py src/coding_agent/tui/app.py tests/test_history_backtrack.py
git commit -m "Wire double-Esc rewind picker and fork refill into the TUI"
```

---

## Self-Review

- Spec §11 history-backtracking covered: runtime fork (T1), timestamp (T2), picker (T3), app wiring (T4). ✅
- Original session never modified; fork creates a new store truncated at the user message. ✅
- Single-Esc behavior unchanged; double-Esc only on empty composer while idle. ✅
- Fork rejected while a run is active; stale selection re-validated (fork_at raises → error notice + no composer fill). ✅
- New tests are pure-function friendly plus `run_test` integration coverage. ✅
