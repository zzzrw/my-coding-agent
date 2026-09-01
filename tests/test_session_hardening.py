"""Session recovery hardening tests.

These cover high-confidence defects in resume projection and session status:

1. A completed assistant with multiple tool calls but only some results must be
   discarded as a whole group, leaving no orphan tool result behind.
2. A ToolResult carrying both nonempty content and an error must preserve both
   in the projected tool message content in a stable, readable form.
3. Aborted turns must be excluded from resume projection like interrupted turns.
4. Multiple open turns must not retain stale open history during resume.
"""

from coding_agent.runtime.models import Message, ToolCall
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult


def _store(tmp_path):
    return SessionStore.create(tmp_path, workspace="w", model="m", context_window=1)


def test_partial_tool_results_discard_entire_assistant_group(tmp_path):
    store = _store(tmp_path)
    assistant = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
            ToolCall(id="c2", name="read_file", arguments={"path": "b.py"}),
        ],
    )
    store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="c1", tool_name="read_file", ok=True, content="a"
            )
        },
        turn_id="t1",
    )
    store.append_new("turn_end", {"reason": "completed"}, turn_id="t1")

    assert store.project_messages() == []


def test_partial_tool_results_keep_only_complete_groups(tmp_path):
    store = _store(tmp_path)
    complete = Message(
        role="assistant", tool_calls=[ToolCall(id="a1", name="read_file")]
    )
    store.append_new(
        "assistant_message", {"message": complete, "complete": True}, turn_id="t1"
    )
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="a1", tool_name="read_file", ok=True, content="ok"
            )
        },
        turn_id="t1",
    )
    incomplete = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="b1", name="read_file"),
            ToolCall(id="b2", name="read_file"),
        ],
    )
    store.append_new(
        "assistant_message", {"message": incomplete, "complete": True}, turn_id="t1"
    )
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="b1", tool_name="read_file", ok=True, content="partial"
            )
        },
        turn_id="t1",
    )

    projected = store.project_messages()

    assert [item.message.role for item in projected] == ["assistant", "tool"]
    assert projected[0].message.tool_calls[0].id == "a1"
    assert projected[1].message.tool_call_id == "a1"


def test_tool_result_preserves_both_content_and_error(tmp_path):
    store = _store(tmp_path)
    assistant = Message(
        role="assistant", tool_calls=[ToolCall(id="c1", name="run_command")]
    )
    store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="c1",
                tool_name="run_command",
                ok=False,
                content="partial stdout",
                error="command failed",
            )
        },
        turn_id="t1",
    )

    content = store.project_messages()[-1].message.content

    assert content == "partial stdout\n\n[error] command failed"


def test_aborted_turn_is_excluded_from_projection(tmp_path):
    store = _store(tmp_path)
    store.append_new("turn_start", {"turn_id": "t1"}, turn_id="t1")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="cancelled prompt")},
        turn_id="t1",
    )
    store.append_new(
        "assistant_message",
        {"message": Message(role="assistant", content="partial answer"), "complete": True},
        turn_id="t1",
    )
    store.append_new("turn_end", {"reason": "aborted"}, turn_id="t1")

    assert store.project_messages() == []
    assert store.project_messages(include_open_turn=True) == []


def test_multiple_open_turns_do_not_retain_stale_history(tmp_path):
    store = _store(tmp_path)
    store.append_new("turn_start", {"turn_id": "stale"}, turn_id="stale")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="stale prompt")},
        turn_id="stale",
    )
    store.append_new("turn_start", {"turn_id": "current"}, turn_id="current")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="current prompt")},
        turn_id="current",
    )

    projected = store.project_messages(include_open_turn=True)

    assert [(item.message.role, item.message.content) for item in projected] == [
        ("user", "current prompt")
    ]


def test_mark_open_final_turn_interrupted_marks_all_open_turns(tmp_path):
    store = _store(tmp_path)
    store.append_new("turn_start", {"turn_id": "t1"}, turn_id="t1")
    store.append_new("turn_start", {"turn_id": "t2"}, turn_id="t2")
    store.mark_open_final_turn_interrupted()

    reasons = [
        record.payload.get("reason")
        for record in store.records()
        if record.type == "turn_end"
    ]
    assert reasons == ["interrupted", "interrupted"]
    assert store.has_interrupted_turn()
    assert store.project_messages(include_open_turn=True) == []
