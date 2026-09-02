from coding_agent.context.truncate import TruncatePolicy
from coding_agent.runtime.models import Message, ToolCall, Usage
from coding_agent.session.models import SessionMessage
from coding_agent.tools.models import ToolSchema

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
        usage=Usage(input_tokens=120, output_tokens=3, total_tokens=123),
    )
    estimated_view = TruncatePolicy(budget=1000).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    # The displayed "used" is the full request total (input + output), not the
    # input-only number the old meter reported.
    assert provider_view.used_tokens == 123
    assert provider_view.estimated is False
    assert estimated_view.used_tokens == TruncatePolicy.estimate_tokens(
        estimated_view.messages
    )
    assert estimated_view.estimated is True


def test_used_prefers_provider_total_when_nothing_was_appended():
    history = history_fixture()
    usage = Usage(input_tokens=200, output_tokens=40, total_tokens=240)
    view = TruncatePolicy(budget=1000).prepare(
        history,
        system_prompt=SYSTEM,
        context_window=1000,
        usage=usage,
    )
    # Every projected message was already part of the measured request, so the
    # appended estimate is zero and the meter shows the provider total alone.
    assert view.used_tokens == 240
    assert view.estimated is False


def test_used_adds_estimate_of_items_appended_since_the_response():
    history = history_fixture()
    appended = item(5, "t3", Message(role="user", content="y" * 2000))
    usage = Usage(input_tokens=200, output_tokens=40, total_tokens=240)
    view = TruncatePolicy(budget=1000).prepare(
        history + [appended],
        system_prompt=SYSTEM,
        context_window=1000,
        usage=usage,
    )
    # The measured response is the history's last assistant message (item 4);
    # only the user message that arrived strictly after it is appended, so the
    # meter adds just that suffix -- never the response text already inside the
    # provider total (its output_tokens) nor any earlier history.
    suffix = [appended.message]
    assert view.used_tokens == 240 + TruncatePolicy.estimate_tokens(suffix)
    assert view.estimated is False
    assert view.used_tokens > 240


def test_used_never_recounts_the_measured_response_output():
    # The assistant response (item 1) is the request the usage measured: its own
    # output tokens are already inside total_tokens. The meter must not add an
    # estimate of that response back on top (the old whole-history minus
    # input_tokens fallback did), or it inflates above the provider total and
    # later DROPS when the next measured total re-emits.
    history = [
        item(0, "t1", Message(role="user", content="old question")),
        item(1, "t1", Message(role="assistant", content="x" * 2000)),
    ]
    usage = Usage(input_tokens=20, output_tokens=500, total_tokens=520)
    view = TruncatePolicy(budget=1000).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=usage
    )
    assert view.used_tokens == 520
    assert view.estimated is False


def test_used_falls_back_to_input_plus_output_when_total_is_omitted():
    # Streaming OpenAI-compatible endpoints may omit total_tokens while still
    # reporting prompt/completion tokens; the measured total is then their sum
    # and must not fall back to a pure estimate (nor drop to zero).
    history = history_fixture()
    usage = Usage(input_tokens=1000, output_tokens=200, total_tokens=0)
    view = TruncatePolicy(budget=1000).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=usage
    )
    assert view.used_tokens == 1200
    assert view.estimated is False


def test_estimate_tokens_counts_cjk_and_emoji_per_codepoint():
    text = "你好" * 40 + "🎉" * 10  # 80 CJK + 10 emoji codepoints
    estimate = TruncatePolicy.estimate_tokens([Message(role="user", content=text)])
    # Non-ASCII text costs about one token per codepoint (never fewer), and the
    # structural overhead stays small: it must not inflate to chars/4 escapes.
    assert estimate >= 90
    assert estimate < 130


def test_estimate_tokens_includes_tool_and_skill_schema_overhead():
    messages = [Message(role="user", content="list the files")]
    schema = ToolSchema(
        name="read_file",
        description="Read a file from disk.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
        risk_level="read",
    )
    without_tools = TruncatePolicy.estimate_tokens(messages)
    with_tools = TruncatePolicy.estimate_tokens(messages, tools=[schema])
    # The schemas ride on the wire with the messages, so a tool-bearing request
    # must never be estimated lower than a bare message estimate.
    assert with_tools > without_tools


def test_auto_compaction_fires_at_headroom_not_full_window():
    # window 20000 => reserve max(12000, 5%) = 12000 => compact when the
    # estimate passes 8000, well before the 20000-token window would overflow.
    history = [
        item(0, "t1", Message(role="user", content="x" * 30000)),
        item(1, "t2", Message(role="user", content="y" * 8000)),
    ]
    view = TruncatePolicy().prepare(
        history, system_prompt=SYSTEM, context_window=20000, usage=None
    )
    assert view.compacted is True
    assert view.removed_turns == 1
    assert not any("x" * 30000 in (message.content or "") for message in view.messages)
    assert any("y" * 8000 in (message.content or "") for message in view.messages)


def test_auto_compaction_holds_below_headroom_threshold():
    history = [
        item(0, "t1", Message(role="user", content="a" * 20000)),
        item(1, "t2", Message(role="user", content="b" * 2000)),
    ]
    view = TruncatePolicy().prepare(
        history, system_prompt=SYSTEM, context_window=20000, usage=None
    )
    # ~5500 tokens of messages is comfortably under the 8000 headroom threshold:
    # nothing is dropped even though the estimate already exceeds the old
    # chars/4 feel for the window.
    assert view.compacted is False
    assert len(view.messages) == len(history) + 1


def test_small_window_still_keeps_the_current_turn():
    # A tiny window cannot reserve 12000 tokens of headroom; compaction must
    # not race below the current turn even when that turn alone overflows.
    history = [
        item(0, "t1", Message(role="user", content="z" * 20000)),
        item(1, "t2", Message(role="user", content="w" * 20000)),
    ]
    view = TruncatePolicy().prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert view.removed_turns == 1
    assert any("w" * 20000 in (message.content or "") for message in view.messages)


def test_prepare_fallback_estimate_includes_tools():
    history = history_fixture()
    schema = ToolSchema(
        name="read_file",
        description="Read a file from disk.",
        parameters={"type": "object"},
        risk_level="read",
    )
    with_tools = TruncatePolicy(budget=1000).prepare(
        history,
        system_prompt=SYSTEM,
        context_window=1000,
        usage=None,
        tools=[schema],
    )
    without_tools = TruncatePolicy(budget=1000).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert with_tools.used_tokens > without_tools.used_tokens


def test_prepare_does_not_mutate_history():
    history = history_fixture()
    before = [entry.model_dump() for entry in history]
    TruncatePolicy(budget=25).prepare(
        history, system_prompt=SYSTEM, context_window=1000, usage=None
    )
    assert [entry.model_dump() for entry in history] == before
