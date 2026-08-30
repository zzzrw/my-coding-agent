from coding_agent.context.truncate import TruncatePolicy
from coding_agent.runtime.models import Message, ToolCall, Usage
from coding_agent.session.models import SessionMessage

SYSTEM = Message(role="system", content="You are a coding agent.")


def item(seq, turn, message):
    return SessionMessage(record_id=f"r{seq}", turn_id=turn, seq=seq, message=message)


def history_fixture():
    return [
        item(0, "t1", Message(role="user", content="old question")),
        item(
            1,
            "t1",
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={})],
            ),
        ),
        item(2, "t1", Message(role="tool", content="old result", tool_call_id="c1")),
        item(3, "t2", Message(role="user", content="current question")),
        item(4, "t2", Message(role="assistant", content="current answer")),
    ]


def test_under_budget_preserves_all_messages():
    history = history_fixture()
    view = TruncatePolicy(budget=1000).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert view.compacted is False
    assert view.messages[0] == SYSTEM
    assert len(view.messages) == len(history) + 1


def test_over_budget_removes_complete_turns_only():
    view = TruncatePolicy(budget=25).prepare(
        history_fixture(), system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert view.compacted is True
    assert view.removed_turns == 1
    assert not any(message.tool_call_id == "c1" for message in view.messages)
    assert any(
        "current question" in (message.content or "") for message in view.messages
    )


def test_force_under_budget_compacts_when_turn_can_be_removed():
    view = TruncatePolicy(budget=1000).prepare(
        history_fixture(),
        system_prompt=SYSTEM,
        context_window=1000,
        usage=None,
        force=True,
    )
    assert view.compacted is True
    assert "compacted" in (view.messages[1].content or "")


def test_current_turn_overflow_is_reported():
    current = [item(0, "t1", Message(role="user", content="x" * 1000))]
    view = TruncatePolicy(budget=10).prepare(
        current, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert view.overflow is True
    assert view.compacted is False


def test_usage_is_preferred_and_fallback_is_deterministic():
    history = history_fixture()
    provider_view = TruncatePolicy(budget=1000).prepare(
        history,
        system_prompt=SYSTEM,
        context_window=1000,
        usage=Usage(input_tokens=123),
    )
    estimated_view = TruncatePolicy(budget=1000).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert provider_view.used_tokens == 123
    assert provider_view.estimated is False
    assert estimated_view.used_tokens == TruncatePolicy.estimate_tokens(
        estimated_view.messages
    )
    assert estimated_view.estimated is True


def test_prepare_does_not_mutate_history():
    history = history_fixture()
    before = [entry.model_dump() for entry in history]
    TruncatePolicy(budget=25).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert [entry.model_dump() for entry in history] == before
