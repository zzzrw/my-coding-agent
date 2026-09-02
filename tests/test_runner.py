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


def make_runner(tmp_path, provider, *, max_steps=None):
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

    executor = RecordingExecutor()
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
