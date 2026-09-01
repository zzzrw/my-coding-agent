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
