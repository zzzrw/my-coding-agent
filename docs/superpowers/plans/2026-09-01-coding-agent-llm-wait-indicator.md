# LLM-Wait Indicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** While waiting for the first streamed token of an assistant reply, show an animated `thinking…` placeholder (spinner frame + elapsed seconds) in the conversation area, replacing today's empty assistant row.

**Architecture:** `assistant_started` already inserts an empty assistant `TranscriptItem`. Mark that row `pending` with a `started_at` anchor. While pending, the row renders `⠹ thinking… (3s)` using the existing `SPINNER_FRAMES`; the first `assistant_delta` clears `pending` and the placeholder yields to streamed text. Rendering stays a pure function of the state snapshot; the 0.2s statusline tick refreshes only the pending row's renderable in place (no full-transcript re-mount).

**Tech Stack:** Python 3.11+, Textual 8.2.8, pydantic.

**Spec:** `docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md` §11 (Runtime Events and TUI).

## Global Constraints

- No new runtime events, no protocol changes: the indicator is derived purely from `assistant_started` / `assistant_delta` / `assistant_finished`.
- The statusline spinner, `format_statusline`, and its tests are unchanged.
- `TranscriptRow.render()` keeps returning the raw renderable for testability.
- Existing 535 tests stay green; `ruff check` and `ruff format --check` clean.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/tui/state.py` (modify): add `pending`, `started_at` to `TranscriptItem`.
- `src/coding_agent/tui/reducer.py` (modify): set/clear `pending` on assistant lifecycle events.
- `src/coding_agent/tui/widgets.py` (modify): `_pending_text`, `_row_text`, `TranscriptRow`, `TranscriptView.render_state`, `TranscriptView.update_pending`.
- `src/coding_agent/tui/app.py` (modify): pass frame/now into the transcript render, refresh the pending row on the spinner tick.
- `tests/test_llm_wait_indicator.py` (new): all new tests.

---

### Task 1: Mark pending assistant rows in state and reducer

**Files:** Modify `src/coding_agent/tui/state.py`, `src/coding_agent/tui/reducer.py`; test `tests/test_llm_wait_indicator.py` (new).

**Interfaces:**
- Consumes: `TranscriptItem`, `reduce`, `initial_state`, `time.monotonic`.
- Produces: `TranscriptItem.pending: bool = False`, `TranscriptItem.started_at: float | None = None`; the reducer sets `pending=True` + `started_at` on `assistant_started`, and clears both on the first `assistant_delta` and on `assistant_finished`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_wait_indicator.py`:

```python
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import initial_state


def event(event_type: str, payload: dict) -> RuntimeEvent:
    return RuntimeEvent(type=event_type, run_id="r", payload=payload)


def test_assistant_started_marks_row_pending_with_anchor() -> None:
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("assistant_started", {"message_id": "m1"}))
    row = state.transcript[-1]
    assert row.kind == "assistant"
    assert row.pending is True
    assert row.started_at is not None


def test_first_assistant_delta_clears_pending() -> None:
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("assistant_started", {"message_id": "m1"}))
    state = reduce(state, event("assistant_delta", {"message_id": "m1", "text": "hi"}))
    row = state.transcript[-1]
    assert row.pending is False
    assert row.started_at is None
    assert row.text == "hi"


def test_assistant_finished_without_delta_clears_pending() -> None:
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("assistant_started", {"message_id": "m1"}))
    state = reduce(state, event("assistant_finished", {"message_id": "m1"}))
    row = state.transcript[-1]
    assert row.pending is False
    assert row.started_at is None
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_llm_wait_indicator.py -q`
Expected: all three FAIL with a pydantic ValidationError (no `pending`/`started_at` fields yet).

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/state.py`, `TranscriptItem` (insert after `text: str = ""`):

```python
    pending: bool = False
    started_at: float | None = None
```

In `src/coding_agent/tui/reducer.py`:

`assistant_started` branch — mark the row pending:

```python
    elif event.type == "assistant_started":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id:
            _append_or_update(
                transcript,
                TranscriptItem(
                    kind="assistant",
                    item_id=message_id,
                    pending=True,
                    started_at=time.monotonic(),
                ),
            )
```

`assistant_delta` existing-row update — clear pending:

```python
            else:
                row = transcript[index]
                transcript[index] = row.model_copy(
                    update={
                        "text": row.text + _text(payload.get("text")),
                        "pending": False,
                        "started_at": None,
                    }
                )
```

`assistant_finished` — clear pending on an existing row (keep the current append-if-missing behavior):

```python
    elif event.type == "assistant_finished":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id is not None:
            index = _find_assistant(transcript, message_id)
            if index is None:
                transcript.append(
                    TranscriptItem(kind="assistant", item_id=message_id)
                )
            else:
                row = transcript[index]
                if row.pending:
                    transcript[index] = row.model_copy(
                        update={"pending": False, "started_at": None}
                    )
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_llm_wait_indicator.py tests/test_reducer.py -q`
Expected: all PASS, including the existing `test_assistant_deltas_merge_by_message_id` and `test_assistant_delta_without_started_row_creates_one_and_empty_ids_do_not_match`.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/state.py src/coding_agent/tui/reducer.py tests/test_llm_wait_indicator.py
git commit -m "Mark pending assistant rows while waiting for the first streamed token"
```

---

### Task 2: Pure render helpers for the thinking placeholder

**Files:** Modify `src/coding_agent/tui/widgets.py`; test `tests/test_llm_wait_indicator.py`.

**Interfaces:**
- Consumes: `SPINNER_FRAMES`, `TranscriptItem`, `_row_text`.
- Produces: `_pending_text(item, spinner_frame, now) -> str`; `_row_text(item, *, spinner_frame=0, now=0.0)` renders the placeholder for pending rows.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_wait_indicator.py`:

```python
from coding_agent.tui.state import TranscriptItem
from coding_agent.tui.widgets import SPINNER_FRAMES, _row_text


def test_pending_row_text_shows_frame_thinking_and_elapsed() -> None:
    item = TranscriptItem(
        kind="assistant", item_id="m1", pending=True, started_at=100.0
    )
    text = _row_text(item, spinner_frame=2, now=105.0)
    assert SPINNER_FRAMES[2] in text
    assert "thinking" in text
    assert "5s" in text


def test_pending_row_text_wraps_frame_index() -> None:
    item = TranscriptItem(
        kind="assistant", item_id="m1", pending=True, started_at=100.0
    )
    assert SPINNER_FRAMES[0] in _row_text(item, spinner_frame=10, now=101.0)


def test_pending_row_with_text_renders_text_not_placeholder() -> None:
    item = TranscriptItem(
        kind="assistant", item_id="m1", pending=True, started_at=100.0, text="hello"
    )
    assert _row_text(item) == "hello"


def test_normal_assistant_row_is_unchanged() -> None:
    item = TranscriptItem(kind="assistant", item_id="m1", text="hello")
    assert _row_text(item) == "hello"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_llm_wait_indicator.py -q`
Expected: the four new tests FAIL (no `_pending_text`; `_row_text` has no `spinner_frame`/`now` parameters).

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/widgets.py`, add this function near `SPINNER_FRAMES`:

```python
def _pending_text(item: TranscriptItem, spinner_frame: int, now: float) -> str:
    """Placeholder for an assistant row that has not received its first token."""
    frame = SPINNER_FRAMES[spinner_frame % len(SPINNER_FRAMES)]
    elapsed = max(0, int(now - (item.started_at or now)))
    return f"{frame} thinking… ({elapsed}s)"
```

Update `_row_text`:

```python
def _row_text(
    item: TranscriptItem, *, spinner_frame: int = 0, now: float = 0.0
) -> str:
    if item.kind == "user":
        return f"> {item.text}"
    if item.kind == "local_command":
        return f"$ {item.text}"
    if item.kind == "assistant":
        if item.pending and not item.text:
            return _pending_text(item, spinner_frame, now)
        return item.text
    if item.kind == "tool":
        return _tool_row_text(item)
    return f"[{item.level or 'notice'}] {item.text}"
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_llm_wait_indicator.py tests/test_tui_display_regression.py tests/test_tui_visual_refresh.py -q`
Expected: all PASS (existing single-argument `_row_text(item)` callers keep working through the defaults).

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/widgets.py tests/test_llm_wait_indicator.py
git commit -m "Render a thinking placeholder for pending assistant rows"
```

---

### Task 3: Wire the placeholder into the transcript and the spinner tick

**Files:** Modify `src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/app.py`; test `tests/test_llm_wait_indicator.py`.

**Interfaces:**
- Consumes: `TranscriptRow`, `TranscriptView`, `SPINNER_FRAMES`, `CodingAgentApp`, `initial_state`, `RuntimeEvent`.
- Produces: `TranscriptRow(item, *, index, spinner_frame=0, now=0.0)`; `TranscriptView.render_state(items, *, spinner_frame=0, now=0.0)`; `TranscriptView.update_pending(spinner_frame, now) -> None`; `_tick_spinner` refreshes the pending row.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_wait_indicator.py`:

```python
import pytest

from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.state import TranscriptItem, initial_state
from coding_agent.tui.widgets import (
    SPINNER_FRAMES,
    TranscriptRow,
    TranscriptView,
)


class _FakeRuntime:
    """Minimal runtime enough for the widget test (no submission paths used)."""

    status: str | None = None

    def subscribe(self, callback):
        return lambda: None


def test_transcript_row_renders_placeholder_for_pending_item() -> None:
    item = TranscriptItem(
        kind="assistant", item_id="m1", pending=True, started_at=100.0
    )
    row = TranscriptRow(item, index=0, spinner_frame=3, now=105.0)
    rendered = str(row.render())
    assert SPINNER_FRAMES[3] in rendered
    assert "thinking" in rendered
    assert "5s" in rendered


@pytest.mark.asyncio
async def test_tick_refreshes_pending_row_renderable() -> None:
    app = CodingAgentApp(
        runtime=_FakeRuntime(),
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )
    async with app.run_test(size=(100, 24)):
        app._apply_event(
            RuntimeEvent(
                type="run_started",
                run_id="r",
                turn_id="t",
                payload={"session_id": "s", "model": "fake", "policy": "default"},
            )
        )
        app._apply_event(
            RuntimeEvent(
                type="assistant_started",
                run_id="r",
                turn_id="t",
                payload={"message_id": "m1"},
            )
        )
        transcript = app.query_one("#transcript", TranscriptView)
        row = next(
            r for r in transcript.query(TranscriptRow) if r.item.item_id == "m1"
        )
        before = str(row.render())
        assert "thinking" in before

        app.state = app.state.model_copy(update={"spinner_frame": 4})
        app._tick_spinner()

        after = str(row.render())
        assert SPINNER_FRAMES[4] in after
        assert after != before
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_llm_wait_indicator.py -q`
Expected: `test_transcript_row_renders_placeholder_for_pending_item` FAILS (no `spinner_frame`/`now` on `TranscriptRow`); the tick test fails because `_tick_spinner` does not yet update the pending row.

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/widgets.py`, update `TranscriptRow.__init__`:

```python
    def __init__(
        self,
        item: TranscriptItem,
        *,
        index: int,
        spinner_frame: int = 0,
        now: float = 0.0,
    ) -> None:
        row_id = _row_id(item, index)
        if item.kind == "assistant":
            if item.pending and not item.text:
                self._renderable = _pending_text(item, spinner_frame, now)
            else:
                self._renderable = markdown_to_text(item.text)
        else:
            self._renderable = _row_text(item)
        super().__init__(
            self._renderable,
            id=row_id,
            markup=False,
            classes=f"row row-{item.kind}",
        )
        self.item = item
```

Update `TranscriptView.render_state` to thread frame/now and add `update_pending`:

```python
    async def render_state(
        self,
        items: Iterable[TranscriptItem],
        *,
        spinner_frame: int = 0,
        now: float = 0.0,
    ) -> None:
        """Replace rendered rows with a state snapshot."""
        await self.remove_children()
        rows = [
            TranscriptRow(
                item, index=index, spinner_frame=spinner_frame, now=now
            )
            for index, item in enumerate(items)
        ]
        self._rendered_text = "\n".join(
            _row_text(row.item, spinner_frame=spinner_frame, now=now)
            for row in rows
        )
        if rows:
            await self.mount_all(rows)

    def update_pending(self, spinner_frame: int, now: float) -> None:
        """Refresh the renderable of the pending assistant row, if any.

        Called from the statusline spinner tick so the placeholder animates in
        place without re-mounting the whole transcript.
        """
        for row in self.query(TranscriptRow):
            if row.item.kind == "assistant" and row.item.pending:
                row._renderable = _pending_text(row.item, spinner_frame, now)
                row.refresh()
```

In `src/coding_agent/tui/app.py`:

- In `_refresh_widgets`, pass frame/now into the transcript render (replace
  `await transcript.render_state(self.state.transcript)`):

```python
            await transcript.render_state(
                self.state.transcript,
                spinner_frame=self.state.spinner_frame,
                now=time.monotonic(),
            )
```

- In `_tick_spinner`, after the statusline `render_state`, refresh the pending
  row:

```python
            self.query_one("#statusline", StatusLine).render_state(
                self.state, getattr(self.runtime, "status", None)
            )
            self.query_one("#transcript", TranscriptView).update_pending(
                self.state.spinner_frame, time.monotonic()
            )
```

  Ensure `time` is imported in `app.py` (add `import time` if missing).

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_llm_wait_indicator.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the full suite and linters**

Run: `pytest -q` then `ruff check src tests` then `ruff format --check src tests`
Expected: 535 + 9 new test cases pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/coding_agent/tui/widgets.py src/coding_agent/tui/app.py tests/test_llm_wait_indicator.py
git commit -m "Animate a thinking placeholder in the transcript while the LLM is pending"
```

---

## Self-Review

- Spec §11 LLM-wait indicator covered: pending state (T1), pure render (T2), tick wiring (T3). ✅
- No new runtime events or protocol changes; indicator derived from existing events. ✅
- Statusline spinner/`format_statusline` untouched. ✅
- New tests are pure-function friendly (reducer + `_row_text`) plus one `run_test` integration test. ✅
