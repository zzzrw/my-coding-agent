from coding_agent.runtime.events import RuntimeEvent
from coding_agent.session.models import ApprovalRequest
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import TuiState, initial_state


def event(event_type: str, payload: dict, *, run_id: str | None = None) -> RuntimeEvent:
    return RuntimeEvent(type=event_type, run_id=run_id, payload=payload)


def approval_request() -> ApprovalRequest:
    return ApprovalRequest(
        request_id="a1",
        run_id="r1",
        tool_call_id="c1",
        tool_name="write_file",
        risk_level="mutate_file",
        arguments={"path": "main.py"},
        reason="write requested",
    )


def test_assistant_deltas_merge_by_message_id():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("assistant_started", {"message_id": "m1"}))
    state = reduce(state, event("assistant_delta", {"message_id": "m1", "text": "hel"}))
    state = reduce(state, event("assistant_delta", {"message_id": "m1", "text": "lo"}))

    assert [row.text for row in state.transcript if row.kind == "assistant"] == [
        "hello"
    ]


def test_assistant_delta_without_started_row_creates_one_and_empty_ids_do_not_match():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state, event("assistant_delta", {"message_id": "m1", "text": "hello"})
    )
    state = reduce(state, event("assistant_started", {"message_id": "m2"}))
    state = reduce(
        state, event("assistant_delta", {"message_id": "m2", "text": "world"})
    )

    assert [
        (row.item_id, row.text) for row in state.transcript if row.kind == "assistant"
    ] == [
        ("m1", "hello"),
        ("m2", "world"),
    ]


def test_tool_started_and_finished_update_one_row():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        event(
            "tool_started",
            {"tool_call_id": "c1", "tool_name": "read_file", "arguments": {}},
        ),
    )
    running = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert running.tool_status == "running"
    assert state.active_tool_call_id == "c1"

    state = reduce(
        state,
        event(
            "tool_finished",
            {
                "tool_call_id": "c1",
                "tool_name": "read_file",
                "ok": True,
                "content": "x",
            },
        ),
    )
    rows = [row for row in state.transcript if row.tool_call_id == "c1"]
    assert len(rows) == 1
    assert rows[0].tool_status == "success"
    assert rows[0].text == "x"
    assert state.active_tool_call_id is None


def test_tool_finish_without_start_creates_terminal_row_and_classifies_cancelled():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        event(
            "tool_finished",
            {
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "ok": False,
                "content": "stopped",
                "status": "cancelled",
            },
        ),
    )

    row = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert row.tool_status == "cancelled"


def test_stale_approval_resolution_does_not_change_state():
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event(
            "run_started", {"session_id": "s1", "model": "fake", "policy": "default"}
        ),
    )
    next_state = reduce(
        state,
        event(
            "approval_resolved",
            {"request_id": "stale", "decision": "approve", "status": "approved"},
        ),
    )

    assert next_state == state


def test_approval_and_run_outcomes_update_status():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        event(
            "run_started",
            {"session_id": "s1", "model": "fake", "policy": "default"},
            run_id="r1",
        ),
    )
    assert state.status == "running"
    assert state.active_run_id == "r1"
    state = reduce(state, event("approval_requested", {"request": approval_request()}))
    assert state.status == "waiting_approval"
    assert state.pending_approval == approval_request()
    state = reduce(
        state,
        event(
            "approval_resolved",
            {"request_id": "a1", "decision": "approve", "status": "approved"},
        ),
    )
    assert state.status == "running"
    assert state.pending_approval is None


def test_run_finished_aborted_is_terminal_aborted_state():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        reduce(
            state,
            event(
                "run_started",
                {"session_id": "s1", "model": "fake", "policy": "default"},
                run_id="r1",
            ),
        ),
        event(
            "run_finished",
            {"outcome": {"reason": "aborted", "steps": 1}, "steps": 1},
            run_id="r1",
        ),
    )

    assert state.status == "aborted"
    assert state.active_run_id is None
    assert state.active_turn_id is None
    assert state.active_tool_call_id is None


def test_run_finished_string_aborted_outcome_is_terminal():
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("run_finished", {"outcome": "aborted", "steps": 1}),
    )

    assert state.status == "aborted"
    assert state.active_run_id is None


def test_run_finished_error_outcome_remains_error():
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("run_finished", {"outcome": {"reason": "provider_error"}, "steps": 1}),
    )

    assert state.status == "error"


def test_run_error_adds_system_row_and_sets_error_status():
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("run_error", {"code": "provider_error", "message": "unavailable"}),
    )

    assert state.status == "error"
    row = state.transcript[-1]
    assert row.kind == "system"
    assert row.text == "unavailable"


def test_context_session_policy_and_notice_updates():
    state = initial_state(workspace="old", model="old-model")
    state = reduce(
        state,
        event(
            "context_updated",
            {"used_tokens": 12, "context_window": 100, "estimated": True},
        ),
    )
    state = reduce(state, event("policy_changed", {"policy": "workspace"}))
    state = reduce(
        state,
        event(
            "session_loaded",
            {"session_id": "s2", "workspace": "/new", "model": "new-model"},
        ),
    )
    state = reduce(state, event("notice", {"level": "warning", "message": "loaded"}))

    assert state.session_id == "s2"
    assert state.workspace == "/new"
    assert state.model == "new-model"
    assert state.policy == "default"
    assert state.context_used == 12
    assert state.context_window == 100
    assert state.context_estimated is True
    assert state.transcript[-1].kind == "system"
    assert state.transcript[-1].text == "loaded"


def test_reduce_does_not_mutate_input_state_or_nested_rows():
    original = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("assistant_started", {"message_id": "m1"}),
    )
    next_state = reduce(
        original, event("assistant_delta", {"message_id": "m1", "text": "hi"})
    )

    assert original.transcript[0].text == ""
    assert len(original.transcript) == 1
    assert next_state.transcript is not original.transcript
    assert next_state.transcript[0] is not original.transcript[0]
    assert isinstance(original, TuiState)
