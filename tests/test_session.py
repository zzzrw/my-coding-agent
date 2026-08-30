import pytest

from coding_agent.runtime.models import Message, ToolCall
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult


def test_append_assigns_sequence_and_parent(tmp_path):
    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    first = store.append_new(
        "user_message", {"message": Message(role="user", content="hi")}
    )
    second = store.append_new("turn_start", {"turn_id": "t1"})
    assert first.seq == 0
    assert second.seq == 1
    assert second.parent_id == first.id


def test_header_round_trips_workspace_model_title_and_window(tmp_path):
    store = SessionStore.create(
        tmp_path,
        workspace=str(tmp_path),
        model="fake-model",
        context_window=128_000,
        title="Demo page",
    )
    reopened = SessionStore.open(tmp_path, store.session_id)
    assert reopened.header.workspace == str(tmp_path)
    assert reopened.header.model == "fake-model"
    assert reopened.header.title == "Demo page"
    assert reopened.header.context_window == 128_000


def test_create_rejects_existing_session_id(tmp_path):
    SessionStore.create(
        tmp_path, session_id="fixed", workspace="w", model="m", context_window=1
    )
    with pytest.raises(FileExistsError):
        SessionStore.create(
            tmp_path, session_id="fixed", workspace="w", model="m", context_window=1
        )


def test_projection_does_not_duplicate_audit_tool_call(tmp_path):
    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
    )
    store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    store.append_new("tool_call", {"tool_call": assistant.tool_calls[0]}, turn_id="t1")
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="c1", tool_name="read_file", ok=True, content="x"
            )
        },
        turn_id="t1",
    )
    assert [item.message.role for item in store.project_messages()] == [
        "assistant",
        "tool",
    ]


def test_projection_rejects_unmatched_tool_result(tmp_path):
    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    assistant = Message(
        role="assistant", tool_calls=[ToolCall(id="c1", name="read_file")]
    )
    store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="unknown", tool_name="read_file", ok=True, content="x"
            )
        },
        turn_id="t1",
    )
    with pytest.raises(ValueError, match="tool_result"):
        store.project_messages()


def test_projection_excludes_dangling_tool_assistant(tmp_path):
    store = SessionStore.create(tmp_path, workspace="w", model="m", context_window=1)
    assistant = Message(
        role="assistant", tool_calls=[ToolCall(id="c1", name="read_file")]
    )
    store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    assert store.project_messages() == []


def test_open_rejects_broken_sequence_chain(tmp_path):
    store = SessionStore.create(tmp_path, workspace="w", model="m", context_window=1)
    store.append_new("user_message", {"message": Message(role="user", content="x")})
    text = store.path.read_text(encoding="utf-8").replace('"seq": 0', '"seq": 4')
    store.path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="sequence"):
        SessionStore.open(tmp_path, store.session_id)


def test_reopened_header_reflects_latest_append_timestamp(tmp_path):
    store = SessionStore.create(tmp_path, workspace="w", model="m", context_window=1)
    created = store.header.updated_at
    store.append_new("user_message", {"message": Message(role="user", content="x")})
    reopened = SessionStore.open(tmp_path, store.session_id)
    assert reopened.header.updated_at >= created
    assert reopened.header.updated_at == reopened.records()[-1].timestamp


def test_open_turn_is_interrupted_without_replay(tmp_path):
    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    store.append_new("turn_start", {"turn_id": "t1"}, turn_id="t1")
    reopened = SessionStore.open(tmp_path, store.session_id)
    assert reopened.has_interrupted_turn()
    assert reopened.project_messages() == []


def test_malformed_final_line_is_usable_and_reports_notice(tmp_path):
    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    store.append_new("user_message", {"message": Message(role="user", content="hi")})
    store.path.write_text(
        store.path.read_text(encoding="utf-8") + "{bad json\n", encoding="utf-8"
    )
    reopened = SessionStore.open(tmp_path, store.session_id)
    assert len(reopened.records()) == 1
    assert "corrupt" in (reopened.load_notice or "")
