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
