"""Focused hardening tests for command classification and tool-call execution.

Covers four safety/completion gaps:

1. ``git remote remove/rm/delete`` and ``git remote prune`` are catastrophic.
2. ``git -c`` / ``--config-env`` shell aliases cannot bypass catastrophic policy.
3. Nested command substitutions and process substitutions are inspected recursively.
4. Tool calls execute only after an explicit ``tool_calls`` completion, and
   malformed/non-object JSON arguments stay structured failed ToolResults.
"""

import asyncio
from pathlib import Path

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.policy.command import classify_command
from coding_agent.runtime.models import LLMEvent, Message
from coding_agent.runtime.runner import AgentRunner
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult, ToolSchema
from coding_agent.tools.registry import ToolRegistry

SHELL = ToolSchema(
    name="run_command", description="shell", parameters={}, risk_level="mutate_shell"
)


# ---------------------------------------------------------------------------
# Requirement 1: destructive git remote subcommands are catastrophic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git remote remove origin",
        "git remote rm origin",
        "git remote delete origin",
        "git remote prune origin",
    ],
)
def test_destructive_git_remote_subcommands_are_catastrophic(command):
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


@pytest.mark.parametrize(
    "command",
    [
        "git remote",
        "git remote -v",
        "git remote add origin https://example.com/repo.git",
        "git remote show origin",
        "git remote set-url origin https://example.com/repo.git",
        "git remote rename origin upstream",
    ],
)
def test_ordinary_git_remote_subcommands_are_not_catastrophic(command):
    assert classify_command(command).catastrophic is False


# ---------------------------------------------------------------------------
# Requirement 2: git config shell aliases cannot bypass catastrophic policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git -c 'alias.x=!rm -rf /' x",
        "git -c 'alias.x=!git push -f origin main' x",
        "git -c 'alias.x=!sh -c \"rm -rf /\"' x",
    ],
)
def test_git_config_shell_alias_bodies_are_catastrophic(command):
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


@pytest.mark.parametrize(
    "command",
    [
        "git -c core.editor=vim commit",
        "git -c x=y push origin main",
        "git -c 'alias.co=checkout' co",
        "git -c 'alias.st=!echo status' st",
    ],
)
def test_ordinary_git_config_values_are_not_catastrophic(command):
    assert classify_command(command).catastrophic is False


def test_git_config_env_is_conservatively_catastrophic():
    # --config-env resolves the injected value from an environment variable we
    # cannot inspect, so conservatively reject rather than risk a shell alias.
    command = "git --config-env=GIT_ALIAS_VALUE x"
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


# ---------------------------------------------------------------------------
# Requirement 3: recursive inspection of substitutions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "cat <(git push -f origin main)",
        "diff <(git push -f origin main) <(echo ok)",
        "echo >(git push -f origin main)",
        "cat <(rm -rf /)",
        "echo $(cat <(git push -f origin main))",
        'echo $(git push -f origin "main(x)")',
    ],
)
def test_nested_and_process_substitution_destructive_commands_are_catastrophic(command):
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


@pytest.mark.parametrize(
    "command",
    [
        "echo '$(git push -f origin main)'",
        "echo '`git push -f origin main`'",
        'echo "git push -f origin main"',
        'echo "<(git push -f origin main)"',
    ],
)
def test_quoted_substitution_literals_are_not_catastrophic(command):
    assert classify_command(command).catastrophic is False


@pytest.mark.parametrize(
    "command",
    [
        "echo $(echo hello)",
        "diff <(echo a) <(echo b)",
        "echo $(git status)",
        'echo "$(git status)"',
    ],
)
def test_safe_substitutions_are_not_catastrophic(command):
    assert classify_command(command).catastrophic is False


# ---------------------------------------------------------------------------
# Requirement 4: tool-call execution gated on final completion reason
# ---------------------------------------------------------------------------


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
            tool_call_id=call.id, tool_name=call.name, ok=True, content="ok"
        )


def tool_response(arguments='{"path":"main.py"}'):
    return [
        LLMEvent(type="tool_call_start", tool_call_id="c1", tool_name="read_file"),
        LLMEvent(type="tool_call_delta", tool_call_id="c1", arguments_delta=arguments),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]


def make_runner(tmp_path, provider, *, max_steps=20):
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
@pytest.mark.parametrize("reason", ["incomplete", "content_filter", "length"])
async def test_tool_calls_not_executed_after_later_response_end_reason(
    tmp_path, reason
):
    response = [
        *tool_response()[:-1],
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
        LLMEvent(type="response_end", finish_reason=reason),
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
async def test_tool_calls_not_executed_after_later_stop_reason(tmp_path):
    response = [
        *tool_response()[:-1],
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
        LLMEvent(type="response_end", finish_reason="stop"),
    ]
    runner, _, _, executor = make_runner(tmp_path, ScriptedProvider([response]))
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_tool_call_with_tool_calls_completion_still_executes(tmp_path):
    response = [
        *tool_response()[:-1],
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]
    provider = ScriptedProvider([response, [LLMEvent(type="text_delta", text="done")]])
    runner, _, _, executor = make_runner(tmp_path, provider)
    await runner.run_turn("inspect", run_id="r1", turn_id="t1", signal=asyncio.Event())
    assert len(executor.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{", "[]", "null", '"text"'])
async def test_malformed_json_is_structured_failed_tool_result(tmp_path, arguments):
    response = tool_response(arguments)
    provider = ScriptedProvider(
        [response, [LLMEvent(type="text_delta", text="recovered")]]
    )
    runner, store, _, executor = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert executor.calls == []
    projected = store.project_messages(include_open_turn=True)
    assert any(
        item.message.role == "tool"
        and "invalid tool arguments" in (item.message.content or "")
        for item in projected
    )
