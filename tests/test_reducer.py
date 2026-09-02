import pytest
from pydantic import ValidationError

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.session.models import ApprovalRequest
from coding_agent.tui.reducer import _command_text, reduce
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
    state = reduce(state, event("assistant_started", {"message_id": " "}))
    state = reduce(
        state, event("assistant_delta", {"message_id": "\t", "text": "ignored"})
    )
    state = reduce(state, event("assistant_finished", {"message_id": ""}))

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


def test_tool_finish_requires_boolean_ok_and_terminal_status():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        event(
            "tool_finished",
            {
                "tool_call_id": "c1",
                "tool_name": "read_file",
                "ok": "false",
                "content": "failed",
                "status": "success",
            },
        ),
    )
    state = reduce(
        state,
        event(
            "tool_finished",
            {
                "tool_call_id": "c2",
                "tool_name": "read_file",
                "ok": True,
                "content": "ok",
                "status": "running",
            },
        ),
    )

    rows = {row.tool_call_id: row for row in state.transcript}
    assert rows["c1"].tool_status == "error"
    assert rows["c2"].tool_status == "success"


def test_tool_finished_uses_strict_boolean_ok_and_ignores_running_status():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        event(
            "tool_finished",
            {
                "tool_call_id": "c1",
                "tool_name": "read_file",
                "ok": "false",
                "content": "not successful",
                "status": "running",
            },
        ),
    )

    row = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert row.tool_status == "error"


def test_whitespace_assistant_ids_do_not_create_transcript_rows():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("assistant_started", {"message_id": "   "}))
    state = reduce(
        state,
        event("assistant_delta", {"message_id": "\t", "text": "ignored"}),
    )
    state = reduce(state, event("assistant_finished", {"message_id": ""}))

    assert not any(row.kind == "assistant" for row in state.transcript)


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


def test_resumed_tool_rows_preserve_validated_error_and_cancelled_statuses():
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event(
            "session_loaded",
            {
                "session_id": "s1",
                "workspace": "/tmp/project",
                "model": "fake",
                "context_window": 1000,
                "history": [
                    {
                        "record_id": "tool-error",
                        "turn_id": "t1",
                        "seq": 1,
                        "tool_status": "error",
                        "message": {
                            "role": "tool",
                            "content": "permission denied",
                            "tool_call_id": "c1",
                            "name": "read_file",
                        },
                    },
                    {
                        "record_id": "tool-cancelled",
                        "turn_id": "t1",
                        "seq": 2,
                        "tool_status": "cancelled",
                        "message": {
                            "role": "tool",
                            "content": "cancelled",
                            "tool_call_id": "c2",
                            "name": "run_command",
                        },
                    },
                ],
            },
        ),
    )

    rows = {row.tool_call_id: row for row in state.transcript}
    assert rows["c1"].tool_status == "error"
    assert rows["c2"].tool_status == "cancelled"


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
            {
                "session_id": "s2",
                "workspace": "/new",
                "model": "new-model",
                "context_window": 200,
                "history": [
                    {
                        "record_id": "u1",
                        "turn_id": "t1",
                        "seq": 1,
                        "message": {"role": "user", "content": "inspect"},
                    },
                    {
                        "record_id": "a1",
                        "turn_id": "t1",
                        "seq": 2,
                        "message": {"role": "assistant", "content": "reading"},
                    },
                    {
                        "record_id": "tool1",
                        "turn_id": "t1",
                        "seq": 3,
                        "message": {
                            "role": "tool",
                            "content": "contents",
                            "tool_call_id": "call-1",
                            "name": "read_file",
                        },
                    },
                ],
            },
        ),
    )
    state = reduce(state, event("notice", {"level": "warning", "message": "loaded"}))

    assert state.session_id == "s2"
    assert state.workspace == "/new"
    assert state.model == "new-model"
    assert state.policy == "workspace"
    assert state.context_used == 0
    assert state.context_window == 200
    assert state.context_estimated is False
    assert [
        (
            row.kind,
            row.item_id,
            row.text,
            row.tool_name,
            row.tool_call_id,
            row.tool_status,
        )
        for row in state.transcript[:-1]
    ] == [
        ("user", "u1", "inspect", None, None, None),
        ("assistant", "a1", "reading", None, None, None),
        ("tool", "tool1", "contents", "read_file", "call-1", "success"),
    ]
    assert state.transcript[-1].kind == "system"
    assert state.transcript[-1].text == "loaded"


def test_new_session_loaded_clears_visible_history_and_stale_context():
    state = reduce(
        initial_state(workspace="old", model="old-model"),
        event("user_message", {"message_id": "old-user", "text": "old"}),
    )
    state = reduce(
        state,
        event(
            "context_updated",
            {"used_tokens": 12, "context_window": 100, "estimated": True},
        ),
    )

    state = reduce(
        state,
        event(
            "session_loaded",
            {
                "session_id": "fresh",
                "workspace": "/fresh",
                "model": "fresh-model",
                "context_window": 300,
                "history": [],
            },
        ),
    )

    assert state.transcript == []
    assert state.context_used == 0
    assert state.context_window == 300
    assert state.context_estimated is False


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


def test_reduce_isolates_nested_approval_snapshots():
    request = approval_request()
    original = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("approval_requested", {"request": request}),
    )
    request.arguments["path"] = "changed-by-event-owner"
    next_state = reduce(original, event("notice", {"message": "still pending"}))

    assert original.pending_approval is not request
    assert original.pending_approval.arguments == {"path": "main.py"}
    assert next_state.pending_approval is not original.pending_approval
    next_state.pending_approval.arguments["path"] = "changed-in-next-snapshot"
    assert original.pending_approval.arguments == {"path": "main.py"}


def test_tui_state_rejects_coerced_context_fields():
    with pytest.raises(ValidationError):
        TuiState(workspace="/tmp/project", model="fake", context_used="12")
    with pytest.raises(ValidationError):
        TuiState(workspace="/tmp/project", model="fake", context_estimated="false")


def test_command_text_for_write_file_omits_content_payload() -> None:
    label = _command_text({"path": "src/App.jsx", "content": "x" * 5000}, "write_file")
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


def test_tool_draft_caption_grows_then_resolution_clears_it() -> None:
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("assistant_started", {"message_id": "m1"}),
    )
    state = reduce(
        state,
        event(
            "tool_draft",
            {
                "message_id": "m1",
                "tool_call_id": "c1",
                "tool_name": "write_file",
                "args_len": 12,
            },
        ),
    )
    row = next(item for item in state.transcript if item.kind == "assistant")
    assert row.draft_caption == "drafting write_file · 12 chars"

    state = reduce(
        state,
        event(
            "tool_draft",
            {
                "message_id": "m1",
                "tool_call_id": "c1",
                "tool_name": "write_file",
                "args_len": 40,
            },
        ),
    )
    row = next(item for item in state.transcript if item.kind == "assistant")
    assert row.draft_caption == "drafting write_file · 40 chars"

    state = reduce(state, event("assistant_finished", {"message_id": "m1"}))
    state = reduce(
        state,
        event(
            "tool_started",
            {
                "tool_call_id": "c1",
                "tool_name": "write_file",
                "arguments": {"path": "main.py", "content": "x" * 5},
            },
        ),
    )
    state = reduce(
        state,
        event(
            "tool_finished",
            {
                "tool_call_id": "c1",
                "tool_name": "write_file",
                "ok": True,
                "content": "saved",
            },
        ),
    )
    rows = state.transcript
    assert not any("drafting" in (item.draft_caption or "") for item in rows)
    assistant = next(item for item in rows if item.kind == "assistant")
    assert assistant.draft_caption is None
    assert assistant.pending is False
    tools = [item for item in rows if item.kind == "tool"]
    assert len(tools) == 1
    assert tools[0].tool_call_id == "c1"
    assert tools[0].text == "saved"


def test_tool_draft_ignored_once_assistant_row_has_prose() -> None:
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("assistant_started", {"message_id": "m1"}),
    )
    state = reduce(
        state, event("assistant_delta", {"message_id": "m1", "text": "hello"})
    )
    state = reduce(
        state,
        event(
            "tool_draft",
            {
                "message_id": "m1",
                "tool_call_id": "c1",
                "tool_name": "write_file",
                "args_len": 90,
            },
        ),
    )
    row = next(item for item in state.transcript if item.kind == "assistant")
    assert row.draft_caption is None
    assert row.text == "hello"


def test_tool_draft_run_command_surfaces_bounded_human_target() -> None:
    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        event("assistant_started", {"message_id": "m1"}),
    )
    state = reduce(
        state,
        event(
            "tool_draft",
            {
                "message_id": "m1",
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "args_len": 8,
                "target": "ls -la",
            },
        ),
    )
    row = next(item for item in state.transcript if item.kind == "assistant")
    assert row.draft_caption == "drafting run_command · ls -la"
