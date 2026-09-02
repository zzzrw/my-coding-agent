import asyncio
import json

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.runtime.models import LLMEvent, Message
from coding_agent.runtime.runner import AgentRunner
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult
from coding_agent.tools.registry import ToolRegistry


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append(messages)
        for event in self.responses.pop(0):
            yield event


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, call, **kwargs):
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=True,
            content="file content",
        )


def tool_response(arguments='{"path":"main.py"}', *, call_id="c1"):
    # ``call_id`` must be unique across tool waves in the same turn: the session
    # store matches each tool_result to its preceding assistant tool call, so a
    # reused id makes the second projection fail.
    return [
        LLMEvent(type="tool_call_start", tool_call_id=call_id, tool_name="read_file"),
        LLMEvent(
            type="tool_call_delta", tool_call_id=call_id, arguments_delta=arguments
        ),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]


def text_response(text="done"):
    return [
        LLMEvent(type="text_delta", text=text),
        LLMEvent(type="response_end", finish_reason="stop"),
    ]


def make_runner(tmp_path, provider, *, max_steps=None, executor=None):
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="fake",
        context_window=1000,
    )
    store.append_new("turn_start", {"turn_id": "t1"}, run_id="r1", turn_id="t1")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="inspect")},
        run_id="r1",
        turn_id="t1",
    )
    events = []

    async def sink(event):
        events.append(event)

    executor = executor or RecordingExecutor()
    runner = AgentRunner(
        provider=provider,
        registry=ToolRegistry(),
        executor=executor,
        context_policy=TruncatePolicy(1000),
        store=store,
        event_sink=sink,
        system_prompt=Message(role="system", content="system"),
        model="fake",
        context_window=1000,
        permission_mode="full",
        max_steps=max_steps,
    )
    return runner, store, events, executor


@pytest.mark.asyncio
async def test_tool_call_then_final_answer(tmp_path):
    provider = ScriptedProvider([tool_response(), text_response("ready")])
    runner, store, events, executor = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed" and outcome.final_text == "ready"
    assert len(executor.calls) == 1
    assert [event.type for event in events].count("tool_started") == 1
    assert any(
        message.message.role == "tool"
        for message in store.project_messages(include_open_turn=True)
    )
    assert provider.requests[0][1].content == "inspect"


@pytest.mark.asyncio
async def test_completed_tool_call_survives_usage_only_response_end(tmp_path):
    response = [
        *tool_response()[:-1],
        LLMEvent(type="response_end", finish_reason="tool_calls"),
        LLMEvent(type="response_end"),
    ]
    provider = ScriptedProvider([response, text_response("ready")])
    runner, _, _, executor = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert len(executor.calls) == 1
    assert outcome.reason == "completed" and outcome.final_text == "ready"


@pytest.mark.asyncio
async def test_tool_draft_events_report_growing_argument_lengths(tmp_path):
    arguments = '{"path":"main.py","content":"hi"}'
    response = [
        LLMEvent(type="tool_call_start", tool_call_id="c1", tool_name="write_file"),
        LLMEvent(
            type="tool_call_delta",
            tool_call_id="c1",
            arguments_delta=arguments[:10],
        ),
        LLMEvent(
            type="tool_call_delta",
            tool_call_id="c1",
            arguments_delta=arguments[10:],
        ),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]
    provider = ScriptedProvider([response, text_response("done")])
    runner, _, events, executor = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert len(executor.calls) == 1
    drafts = [event for event in events if event.type == "tool_draft"]
    assert len(drafts) == 2
    assert [event.payload["args_len"] for event in drafts] == sorted(
        event.payload["args_len"] for event in drafts
    )
    assert drafts[0].payload["args_len"] < drafts[1].payload["args_len"]
    assert all(event.payload["tool_name"] == "write_file" for event in drafts)
    assert all(event.payload["message_id"] for event in drafts)


@pytest.mark.asyncio
async def test_invalid_json_is_returned_to_model_without_execution(tmp_path):
    provider = ScriptedProvider([tool_response('{"path":'), text_response("invalid")])
    runner, store, _, executor = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed" and executor.calls == []
    projected = store.project_messages(include_open_turn=True)
    assert any(
        item.message.role == "tool" and "invalid" in (item.message.content or "")
        for item in projected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["[]", "null", '"text"'])
async def test_non_object_json_arguments_are_structured_tool_errors(
    tmp_path, arguments
):
    provider = ScriptedProvider([tool_response(arguments), text_response("recovered")])
    runner, store, _, executor = make_runner(tmp_path, provider)

    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    assert outcome.final_text == "recovered"
    assert executor.calls == []
    assert any(
        item.message.role == "tool"
        and "invalid tool arguments" in (item.message.content or "")
        for item in store.project_messages(include_open_turn=True)
    )
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_final_length_finish_reason_never_executes_tool_calls(tmp_path):
    response = [
        *tool_response()[:-1],
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
        LLMEvent(type="response_end", finish_reason="length"),
    ]
    runner, _, events, executor = make_runner(tmp_path, ScriptedProvider([response]))

    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    assert executor.calls == []
    assert any(
        event.type == "notice"
        and event.payload["message"] == "truncated tool call was not executed"
        for event in events
    )


@pytest.mark.asyncio
async def test_max_steps_stops_repeating_calls(tmp_path):
    provider = ScriptedProvider([tool_response(), tool_response()])
    runner, _, _, _ = make_runner(tmp_path, provider, max_steps=2)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "max_steps" and outcome.steps == 2


@pytest.mark.asyncio
async def test_truncated_tool_call_is_not_executed(tmp_path):
    response = tool_response()
    response[-1] = LLMEvent(type="response_end", finish_reason="length")
    runner, _, _, executor = make_runner(tmp_path, ScriptedProvider([response]))
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed" and executor.calls == []


@pytest.mark.asyncio
async def test_tool_call_without_explicit_completion_is_not_executed(tmp_path):
    response = tool_response()[:-1]
    runner, _, events, executor = make_runner(tmp_path, ScriptedProvider([response]))
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert executor.calls == []
    assert outcome.reason == "completed"
    assert any(
        event.type == "notice"
        and event.payload
        == {
            "level": "error",
            "message": "truncated tool call was not executed",
        }
        for event in events
    )


@pytest.mark.asyncio
async def test_provider_error_and_abort_are_structured(tmp_path):
    runner, _, _, _ = make_runner(
        tmp_path, ScriptedProvider([[LLMEvent(type="error", error="offline")]])
    )
    provider_error = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    signal = asyncio.Event()
    signal.set()
    aborted = await runner.run_turn("inspect", run_id="r2", turn_id="t2", signal=signal)
    assert provider_error.reason == "provider_error"
    assert aborted.reason == "aborted"


@pytest.mark.asyncio
async def test_unbounded_run_is_not_capped_at_old_default(tmp_path):
    # 21 distinct single-call waves then a final answer. With the old hidden
    # default of 20 this stops at max_steps; unbounded it completes.
    waves = [
        tool_response(json.dumps({"path": f"file{i}.py"}), call_id=f"c{i}")
        for i in range(21)
    ]
    provider = ScriptedProvider([*waves, text_response("done")])
    runner, _, _, executor = make_runner(tmp_path, provider, max_steps=None)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert outcome.steps > 20
    assert len(executor.calls) == 21


@pytest.mark.asyncio
async def test_int_cap_still_bounded(tmp_path):
    provider = ScriptedProvider([tool_response(), tool_response()])
    runner, _, _, _ = make_runner(tmp_path, provider, max_steps=2)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "max_steps" and outcome.steps == 2


@pytest.mark.asyncio
async def test_max_steps_cap_emits_warning_notice_before_returning(tmp_path):
    # A provider that never concludes, capped small: the outcome must be
    # preceded by a warning notice (no silent exit, session 09f88d4d).
    provider = ScriptedProvider([tool_response(), tool_response()])
    runner, _, events, _ = make_runner(tmp_path, provider, max_steps=2)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "max_steps"
    notices = [event for event in events if event.type == "notice"]
    assert notices
    last = notices[-1]
    assert last.payload["level"] == "warning"
    assert "max_steps" in last.payload["message"]
    # The notice is the last event emitted before the outcome.
    assert events.index(last) == len(events) - 1


@pytest.mark.asyncio
async def test_completing_on_last_capped_step_emits_no_max_steps_notice(tmp_path):
    # A bounded turn whose FINAL answer lands exactly on the last allowed step
    # (a tool wave, then the final text stream consumes the cap with no further
    # tool calls) is a normal completion, not a capped exit. It must return
    # reason == "completed" and never precede the outcome with the "reached the
    # max_steps limit" notice.
    provider = ScriptedProvider([tool_response(), text_response("done")])
    runner, _, events, executor = make_runner(tmp_path, provider, max_steps=2)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert outcome.steps == 2
    assert len(executor.calls) == 1
    assert not any(
        event.type == "notice" and "max_steps" in (event.payload.get("message") or "")
        for event in events
    )


class VersionedExecutor:
    """Returns distinct content per call, as if an external write changed the
    underlying file between identical-argument reads."""

    def __init__(self):
        self.calls = []
        self._n = 0

    async def execute(self, call, **kwargs):
        self.calls.append(call)
        self._n += 1
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=f"generation-{self._n}",
        )


class LongTailExecutor:
    """Returns content longer than the result-fingerprint cap whose head stays
    byte-identical across calls while only the region PAST the first 4096 chars
    changes -- simulating a write that edited the tail of a large file (read_file
    may return up to 20,000 chars)."""

    def __init__(self):
        self.calls = []
        self._n = 0

    async def execute(self, call, **kwargs):
        self.calls.append(call)
        self._n += 1
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=True,
            content="A" * 4096 + f"tail-generation-{self._n}",
        )


def _repeated_notice(events):
    return any(
        event.type == "notice"
        and event.payload.get("level") == "warning"
        and "repeated tool call" in (event.payload.get("message") or "")
        for event in events
    )


def _call_waves(arguments, count):
    """``count`` single-call read waves, each with a fresh tool call id so the
    session store can match every tool_result to its own assistant call."""
    return [tool_response(arguments, call_id=f"c{i}") for i in range(count)]


@pytest.mark.asyncio
async def test_distinct_pure_read_exploration_never_trips(tmp_path):
    # 20 distinct reads (pure exploration, zero writes) must never trip.
    waves = [
        tool_response(json.dumps({"path": f"src/mod{i}.py"}), call_id=f"c{i}")
        for i in range(20)
    ]
    provider = ScriptedProvider([*waves, text_response("done")])
    runner, _, events, _ = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert not _repeated_notice(events)


@pytest.mark.asyncio
async def test_interleaved_identical_result_repeats_trip(tmp_path):
    # A,B,A,B,A with identical arguments AND identical results each time: the
    # repeated A (non-consecutive) must trip the detector.
    a = json.dumps({"path": "main.py"})
    b = json.dumps({"path": "other.py"})
    provider = ScriptedProvider(
        [
            tool_response(a, call_id="c0"),
            tool_response(b, call_id="c1"),
            tool_response(a, call_id="c2"),
            tool_response(b, call_id="c3"),
            tool_response(a, call_id="c4"),
        ]
    )
    runner, _, events, _ = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "progress_loop"
    assert _repeated_notice(events)


@pytest.mark.asyncio
async def test_identical_result_repeats_trip_consecutively(tmp_path):
    provider = ScriptedProvider(_call_waves(json.dumps({"path": "main.py"}), 3))
    runner, _, events, _ = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "progress_loop"
    assert _repeated_notice(events)


@pytest.mark.asyncio
async def test_identical_args_with_changed_content_does_not_trip(tmp_path):
    # Re-reading the SAME path with identical arguments, but the executor's
    # content changes between reads (a write landed), is progress, not a loop.
    provider = ScriptedProvider(
        [
            *_call_waves(json.dumps({"path": "data.txt"}), 3),
            text_response("done"),
        ]
    )
    runner, _, events, _ = make_runner(tmp_path, provider, executor=VersionedExecutor())
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert not _repeated_notice(events)


@pytest.mark.asyncio
async def test_tail_only_change_beyond_fingerprint_cap_does_not_trip(tmp_path):
    # Three identical-argument reads whose content differs ONLY past the first
    # 4096 chars (the head window of the result fingerprint). The fingerprint
    # must still observe a tail edit so a genuinely progressing turn is not
    # reported as a repetition loop.
    provider = ScriptedProvider(
        [
            *_call_waves(json.dumps({"path": "big.txt"}), 3),
            text_response("done"),
        ]
    )
    runner, _, events, executor = make_runner(
        tmp_path, provider, executor=LongTailExecutor()
    )
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert len(executor.calls) == 3
    assert not _repeated_notice(events)


@pytest.mark.asyncio
async def test_calls_rejected_before_execution_are_never_counted(tmp_path):
    # Repeated invalid-argument calls never execute, so they must never enter
    # the repetition window: the turn survives to the scripted final text.
    provider = ScriptedProvider([*_call_waves("[]", 4), text_response("recovered")])
    runner, _, events, executor = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert executor.calls == []
    assert not _repeated_notice(events)
