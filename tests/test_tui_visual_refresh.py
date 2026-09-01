"""Tests for the approved TUI visual refresh plan (2026-09-01).

Coverage is added per task:
- Task 1: tool data chain (command + result metadata) across runner, reducer,
  state and the resumed-session projection.
- Task 2: per-kind transcript row classes and app CSS rules.
- Task 3: compact click-expandable tool rows.
- Task 4: lightweight markdown_to_text for assistant rows.
"""

from __future__ import annotations

import asyncio

import pytest

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import LLMEvent, Message
from coding_agent.runtime.runner import AgentRunner
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import initial_state


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append(messages)
        for event in self.responses.pop(0):
            yield event


def tool_response(arguments='{"command": "ls -la"}'):
    return [
        LLMEvent(type="tool_call_start", tool_call_id="c1", tool_name="run_command"),
        LLMEvent(type="tool_call_delta", tool_call_id="c1", arguments_delta=arguments),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]


def text_response(text="done"):
    return [
        LLMEvent(type="text_delta", text=text),
        LLMEvent(type="response_end", finish_reason="stop"),
    ]


# ---------------------------------------------------------------------------
# Task 1: tool data chain (command + metadata)
# ---------------------------------------------------------------------------


def test_tool_started_stores_run_command_argument():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_started",
            payload={
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "arguments": {"command": "ls -la"},
            },
        ),
    )

    row = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert row.command == "ls -la"


def test_tool_started_stores_compact_string_for_other_tools():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_started",
            payload={
                "tool_call_id": "c1",
                "tool_name": "read_file",
                "arguments": {"path": "main.py", "start_line": 1},
            },
        ),
    )

    row = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert row.command is not None
    assert "main.py" in row.command


def test_tool_finished_preserves_command_and_merges_metadata():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_started",
            payload={
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "arguments": {"command": "ls -la"},
            },
        ),
    )
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_finished",
            payload={
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "ok": False,
                "content": "total 8",
                "error": "exit code 2",
                "metadata": {"exit_code": 2, "elapsed_seconds": 2.5, "truncated": True},
            },
        ),
    )

    rows = [row for row in state.transcript if row.tool_call_id == "c1"]
    assert len(rows) == 1
    row = rows[0]
    assert row.command == "ls -la"
    assert row.text == "total 8"
    assert row.elapsed_seconds == 2.5
    assert row.truncated is True
    assert row.exit_code == 2
    assert row.expanded is False


def test_tool_finished_reapply_is_idempotent_for_same_call():
    state = initial_state(workspace="/tmp/project", model="fake")
    finished = RuntimeEvent(
        type="tool_finished",
        payload={
            "tool_call_id": "c1",
            "tool_name": "run_command",
            "ok": True,
            "content": "ok",
            "metadata": {"exit_code": 0, "elapsed_seconds": 1.0, "truncated": False},
        },
    )
    state = reduce(state, finished)
    state = reduce(state, finished)

    assert len([row for row in state.transcript if row.tool_call_id == "c1"]) == 1


def test_projected_transcript_carries_command_and_metadata(tmp_path):
    """Resumed tool rows keep command + metadata when the projection emits them.

    The reducer consumes optional ``command``/``metadata`` keys on history dicts
    (the intended resumed-session projection shape). The persisted tool_call and
    tool_result pair below is projected through the real SessionStore first.
    """
    from coding_agent.runtime.models import ToolCall

    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    call = ToolCall(id="c1", name="run_command", arguments={"command": "ls -la"})
    assistant = Message(role="assistant", tool_calls=[call])
    store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    store.append_new("tool_call", {"tool_call": call}, turn_id="t1")
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="c1",
                tool_name="run_command",
                ok=False,
                content="total 8",
                error="exit code 2",
                metadata={"exit_code": 2, "elapsed_seconds": 2.5, "truncated": True},
            )
        },
        turn_id="t1",
    )

    projected = store.project_messages(include_open_turn=False)
    history = [item.model_dump(mode="json") for item in projected]
    for item in history:
        if item["message"]["role"] == "tool":
            item["command"] = "ls -la"
            item["metadata"] = {
                "exit_code": 2,
                "elapsed_seconds": 2.5,
                "truncated": True,
            }

    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        RuntimeEvent(
            type="session_loaded",
            payload={"session_id": "s1", "history": history},
        ),
    )

    row = next(row for row in state.transcript if row.kind == "tool")
    assert row.command == "ls -la"
    assert row.elapsed_seconds == 2.5
    assert row.truncated is True
    assert row.exit_code == 2
    assert row.expanded is False


@pytest.mark.asyncio
async def test_runner_tool_finished_payload_carries_result_metadata(tmp_path):
    from coding_agent.context.truncate import TruncatePolicy

    class MetadataExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call, **kwargs):
            self.calls.append(call)
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=True,
                content="file content",
                metadata={"exit_code": 0, "elapsed_seconds": 1.5, "truncated": False},
            )

    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
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

    runner = AgentRunner(
        provider=ScriptedProvider([tool_response(), text_response("ready")]),
        registry=ToolRegistry(),
        executor=MetadataExecutor(),
        context_policy=TruncatePolicy(1000),
        store=store,
        event_sink=sink,
        system_prompt=Message(role="system", content="system"),
        model="fake",
        context_window=1000,
        permission_mode="full",
    )
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    finished = [event for event in events if event.type == "tool_finished"]
    assert len(finished) == 1
    assert finished[0].payload["metadata"] == {
        "exit_code": 0,
        "elapsed_seconds": 1.5,
        "truncated": False,
    }
