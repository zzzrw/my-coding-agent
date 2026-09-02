import pytest

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import TranscriptItem, initial_state
from coding_agent.tui.widgets import (
    SPINNER_FRAMES,
    TranscriptRow,
    TranscriptView,
    _row_text,
)


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


def test_pending_row_prefers_draft_caption_over_thinking() -> None:
    item = TranscriptItem(
        kind="assistant",
        item_id="m1",
        pending=True,
        started_at=100.0,
        draft_caption="drafting write_file · 40 chars",
    )
    text = _row_text(item, spinner_frame=2, now=105.0)
    assert "drafting write_file · 40 chars" in text
    assert "thinking" not in text


def test_normal_assistant_row_is_unchanged() -> None:
    item = TranscriptItem(kind="assistant", item_id="m1", text="hello")
    assert _row_text(item) == "hello"


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
    async with app.run_test(size=(100, 24)) as pilot:
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
        # Let the message pump mount the refreshed transcript rows before
        # querying them.
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        row = next(r for r in transcript.query(TranscriptRow) if r.item.item_id == "m1")
        before = str(row.render())
        assert "thinking" in before

        app.state = app.state.model_copy(update={"spinner_frame": 4})
        app._tick_spinner()

        after = str(row.render())
        # _tick_spinner advances the frame before re-rendering the pending row.
        assert SPINNER_FRAMES[5] in after
        assert after != before
