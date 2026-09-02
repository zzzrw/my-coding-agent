# Render Performance Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Eliminate TUI freezes when large LLM replies and tool outputs stream. Two changes: (1) `TranscriptView.render_state` stops re-mounting the whole transcript on every event — it updates only changed rows in place; (2) tool-row command labels stop embedding large write/edit payloads (bounded, compact headers).

**Architecture:** `render_state` keeps one mounted `TranscriptRow` per item id. On each snapshot it diffs against the existing rows: rows whose `TranscriptItem` changed get their `_renderable` replaced and are refreshed in place (never re-created); newly appended rows are mounted once; a materially different id set (session switch) falls back to a full re-mount. Separately, `_command_text` in the reducer builds bounded tool labels that never include `content`/`old_text`/`new_text` payload values and caps the label length.

**Tech Stack:** Python 3.11+, Textual 8.2.8, pydantic.

**Spec:** `docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md` §11 (Runtime Events and TUI).

## Global Constraints

- No runtime events, protocol, or state-model changes.
- The pure-reducer pattern is unchanged; only the widget render path and the row-label builder change.
- `TranscriptRow.render()` keeps returning the raw renderable; `renderable_text` stays a correct plain-text snapshot.
- Existing 563 tests stay green; `ruff check` and `ruff format --check` clean.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/tui/widgets.py` (modify): `TranscriptRow._row_renderable`, incremental `TranscriptView.render_state`.
- `src/coding_agent/tui/reducer.py` (modify): compact `_command_text`.
- `tests/test_render_incremental.py` (new): widget identity/append/replace tests.
- `tests/test_reducer.py` (modify): `_command_text` compactness tests.

---

### Task 1: Incremental `TranscriptView.render_state`

**Files:** Modify `src/coding_agent/tui/widgets.py`; test `tests/test_render_incremental.py` (new).

**Interfaces:**
- Consumes: `TranscriptRow`, `TranscriptItem`, `_row_text`, `_pending_text`, `markdown_to_text`.
- Produces: `TranscriptRow._row_renderable(item, spinner_frame, now)` (static renderable builder); `TranscriptView.render_state(items, *, spinner_frame=0, now=0.0)` updates only changed rows in place and mounts only appended rows, falling back to a full re-render when the id set/order changes materially.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_incremental.py`:

```python
import pytest

from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.state import TranscriptItem, initial_state
from coding_agent.tui.widgets import TranscriptRow, TranscriptView


class _FakeRuntime:
    status = None

    def subscribe(self, callback):
        return lambda: None


def _make_app() -> CodingAgentApp:
    return CodingAgentApp(
        runtime=_FakeRuntime(),
        initial_state=initial_state("/tmp/project", "fake"),
        branch_detector=lambda workspace: None,
    )


def _assistant(i: int, text: str) -> TranscriptItem:
    return TranscriptItem(kind="assistant", item_id=f"m{i}", text=text)


@pytest.mark.asyncio
async def test_render_state_preserves_settled_rows_on_delta() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        items = [_assistant(0, "hello one"), _assistant(1, "hello two")]
        await view.render_state(items)
        rows = list(view.query(TranscriptRow))
        assert len(rows) == 2
        settled_widget = rows[0]
        settled_renderable = rows[0]._renderable

        # A delta only grows the last row: the settled row must be untouched.
        items[1] = _assistant(1, "hello two plus more text")
        await view.render_state(items)
        rows_after = list(view.query(TranscriptRow))
        assert len(rows_after) == 2
        assert rows_after[0] is settled_widget
        assert rows_after[0]._renderable is settled_renderable
        assert rows_after[1]._renderable is not rows[1]._renderable


@pytest.mark.asyncio
async def test_render_state_mounts_only_appended_rows() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        await view.render_state([_assistant(0, "a"), _assistant(1, "b")])
        before = list(view.query(TranscriptRow))
        assert len(before) == 2

        await view.render_state(
            [_assistant(0, "a"), _assistant(1, "b"), _assistant(2, "c")]
        )
        after = list(view.query(TranscriptRow))
        assert len(after) == 3
        assert after[0] is before[0]
        assert after[1] is before[1]
        assert after[2].item.item_id == "m2"


@pytest.mark.asyncio
async def test_render_state_full_replaces_on_material_change() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        await view.render_state([_assistant(0, "a")])
        old = list(view.query(TranscriptRow))[0]

        # A session switch replaces the id set entirely.
        await view.render_state([_assistant(9, "z")])
        new_rows = list(view.query(TranscriptRow))
        assert len(new_rows) == 1
        assert new_rows[0] is not old


@pytest.mark.asyncio
async def test_render_state_incremental_snapshot_matches_full_render() -> None:
    app = _make_app()
    async with app.run_test():
        view = app.query_one("#transcript", TranscriptView)
        await view.render_state([_assistant(0, "**a**"), _assistant(1, "b")])
        await view.render_state(
            [_assistant(0, "**a**"), _assistant(1, "b + delta"), _assistant(2, "c")]
        )
        incremental_text = view.renderable_text

        # A fresh full render of the same items yields the same snapshot.
        await view.render_state(
            [_assistant(0, "**a**"), _assistant(1, "b + delta"), _assistant(2, "c")]
        )
        assert view.renderable_text == incremental_text
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_render_incremental.py -q`
Expected: `test_render_state_preserves_settled_rows_on_delta` FAILS (current code re-mounts, so `rows_after[0] is not settled_widget`); the other tests may also fail or pass depending on widget reuse.

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/widgets.py`:

Extract the renderable builder on `TranscriptRow`:

```python
    @staticmethod
    def _row_renderable(
        item: TranscriptItem, spinner_frame: int = 0, now: float = 0.0
    ) -> object:
        """Build the row renderable for an item (markdown for assistant)."""
        if item.kind == "assistant":
            if item.pending and not item.text:
                return _pending_text(item, spinner_frame, now)
            return markdown_to_text(item.text)
        return _row_text(item)
```

Update `TranscriptRow.__init__` to use it:

```python
        row_id = _row_id(item, index)
        self._renderable = self._row_renderable(item, spinner_frame, now)
        super().__init__(
            self._renderable,
            id=row_id,
            markup=False,
            classes=f"row row-{item.kind}",
        )
        self.item = item
```

Replace `TranscriptView.render_state` with an incremental version:

```python
    async def render_state(
        self,
        items: Iterable[TranscriptItem],
        *,
        spinner_frame: int = 0,
        now: float = 0.0,
    ) -> None:
        """Synchronize rendered rows with a state snapshot.

        Only rows whose item changed are updated in place and refreshed; settled
        rows are never re-created. Newly appended rows are mounted. If the row
        id set or order changes materially (e.g. a session switch), the whole
        list is replaced.
        """
        items = list(items)
        existing = list(self.query(TranscriptRow))
        existing_ids = [row.item.item_id for row in existing]
        new_ids = [item.item_id for item in items]

        if new_ids[: len(existing_ids)] != existing_ids:
            # Materially different id set or order: full replace.
            await self.remove_children()
            rows = [
                TranscriptRow(
                    item, index=index, spinner_frame=spinner_frame, now=now
                )
                for index, item in enumerate(items)
            ]
            self._rendered_text = "\n".join(
                _row_text(item, spinner_frame=spinner_frame, now=now)
                for item in items
            )
            if rows:
                await self.mount_all(rows)
            return

        # Same prefix: update changed rows in place, mount only the new suffix.
        for index, item in enumerate(items[: len(existing_ids)]):
            row = existing[index]
            if row.item != item:
                row.item = item
                row._renderable = self._row_renderable(item, spinner_frame, now)
                row.refresh()
        self._rendered_text = "\n".join(
            _row_text(item, spinner_frame=spinner_frame, now=now) for item in items
        )
        if len(items) > len(existing_ids):
            rows = [
                TranscriptRow(
                    item, index=index, spinner_frame=spinner_frame, now=now
                )
                for index, item in enumerate(items)
                if index >= len(existing_ids)
            ]
            await self.mount_all(rows)
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_render_incremental.py tests/test_tui_visual_refresh.py tests/test_llm_wait_indicator.py -q`
Expected: all PASS, including the incremental identity tests and the existing transcript rendering tests.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/widgets.py tests/test_render_incremental.py
git commit -m "Render transcript incrementally: update changed rows in place"
```

---

### Task 2: Compact bounded tool-row command labels

**Files:** Modify `src/coding_agent/tui/reducer.py`; test `tests/test_reducer.py`.

**Interfaces:**
- Consumes: `arguments: dict`, `tool_name: str | None`.
- Produces: `_command_text` never embeds payload keys and caps the label length. New module constants `_TOOL_LABEL_MAX_CHARS = 160`, `_TOOL_PAYLOAD_KEYS = {"content", "old_text", "new_text"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reducer.py`:

```python
from coding_agent.tui.reducer import _command_text


def test_command_text_for_write_file_omits_content_payload() -> None:
    label = _command_text(
        {"path": "src/App.jsx", "content": "x" * 5000}, "write_file"
    )
    assert label == "path=src/App.jsx"
    assert "content" not in label


def test_command_text_for_edit_file_omits_old_and_new_text() -> None:
    label = _command_text(
        {"path": "a.py", "old_text": "old" * 200, "new_text": "new" * 200},
        "edit_file",
    )
    assert label == "path=a.py"


def test_command_text_read_file_keeps_small_args() -> None:
    label = _command_text(
        {"path": "big.py", "start_line": 10, "end_line": 20}, "read_file"
    )
    assert label == "path=big.py, start_line=10, end_line=20"


def test_command_text_bounds_overlong_values() -> None:
    label = _command_text({"command": "echo " + "a" * 500}, "run_command")
    assert label is not None
    assert len(label) <= 160
    assert label.endswith("…")
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_reducer.py -q`
Expected: the four new tests FAIL — current `_command_text` embeds `content=...`/`old_text=...` and does not bound length.

- [ ] **Step 3: Implement**

In `src/coding_agent/tui/reducer.py`, replace `_command_text` (and add the constants near it):

```python
_TOOL_LABEL_MAX_CHARS = 160
_TOOL_PAYLOAD_KEYS = {"content", "old_text", "new_text"}


def _command_text(arguments: object, tool_name: str | None) -> str | None:
    """Derive a compact tool-row command label from the call arguments.

    Never embeds large payload values (write_file content, edit old/new text);
    the label is bounded so a huge payload cannot bloat the transcript row.
    """
    if not isinstance(arguments, dict):
        return None
    if tool_name == "run_command":
        command = arguments.get("command")
        label = command if isinstance(command, str) and command.strip() else None
    else:
        pairs = [
            f"{key}={value}"
            for key, value in arguments.items()
            if key not in _TOOL_PAYLOAD_KEYS
        ]
        label = ", ".join(pairs) if pairs else None
    if label is None:
        return None
    if len(label) > _TOOL_LABEL_MAX_CHARS:
        label = label[:_TOOL_LABEL_MAX_CHARS].rstrip() + "…"
    return label
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_reducer.py tests/test_tui_display_regression.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the full suite and linters**

Run: `pytest -q` then `ruff check src tests` then `ruff format --check src tests`
Expected: 563 + new tests pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/coding_agent/tui/reducer.py tests/test_reducer.py
git commit -m "Keep tool-row command labels compact and bounded"
```

---

## Self-Review

- Spec §11 transcript-rendering note covered: incremental in-place updates (T1), compact bounded tool labels (T2). ✅
- No runtime events/protocol/state-model changes. ✅
- `render_state` keeps settled rows stable (identity tests), falls back on session switch, and produces a correct `renderable_text`. ✅
- `_command_text` omits write/edit payloads and bounds every label. ✅
