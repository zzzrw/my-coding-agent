from datetime import UTC, datetime

import pytest

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import Message, TurnOutcome
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import initial_state


class _NoopRunner:
    async def run_turn(self, prompt, *, run_id, turn_id, signal):
        return TurnOutcome(reason="completed", final_text=prompt, steps=1)


def _store_with_turns(root, *, turns: int = 2) -> SessionStore:
    store = SessionStore.create(
        root, workspace=str(root), model="fake", context_window=1000
    )
    for index in range(turns):
        tid = f"t{index + 1}"
        store.append_new("turn_start", {"turn_id": tid}, run_id=f"r{index + 1}", turn_id=tid)
        store.append_new(
            "user_message",
            {"message": Message(role="user", content=f"prompt {index + 1}")},
            run_id=f"r{index + 1}",
            turn_id=tid,
        )
        store.append_new(
            "turn_end", {"reason": "completed", "turn_id": tid}, run_id=f"r{index + 1}", turn_id=tid
        )
    return store


def _runtime(store: SessionStore) -> AgentRuntime:
    return AgentRuntime(
        store=store,
        runner_factory=lambda *_: _NoopRunner(),
        context_policy_factory=lambda: TruncatePolicy(1000),
        approval_policy=DefaultApprovalPolicy(),
        system_prompt=Message(role="system", content="s"),
        model="fake",
    )


@pytest.mark.asyncio
async def test_fork_at_creates_truncated_new_session_and_returns_prompt(tmp_path):
    store = _store_with_turns(tmp_path)
    original_path = store.path
    original_text = original_path.read_text(encoding="utf-8")
    runtime = _runtime(store)
    original_session_id = runtime.session_id

    prompt = await runtime.fork_at("user-t1")

    assert prompt == "prompt 1"
    assert runtime.session_id != original_session_id
    assert [r.type for r in runtime.store.records()] == [
        "turn_start",
        "user_message",
        "turn_end",
    ]
    # Original session file is untouched and still holds every record.
    assert original_path.read_text(encoding="utf-8") == original_text
    # New session file is truncated at the fork point.
    new_text = runtime.store.path.read_text(encoding="utf-8")
    assert "prompt 1" in new_text
    assert "prompt 2" not in new_text


@pytest.mark.asyncio
async def test_fork_at_unknown_message_raises(tmp_path):
    store = _store_with_turns(tmp_path)
    runtime = _runtime(store)
    with pytest.raises(ValueError):
        await runtime.fork_at("user-nope")


@pytest.mark.asyncio
async def test_fork_at_latest_message_works_with_open_turn(tmp_path):
    store = _store_with_turns(tmp_path, turns=1)
    runtime = _runtime(store)
    prompt = await runtime.fork_at("user-t1")
    assert prompt == "prompt 1"
    assert [r.type for r in runtime.store.records()] == [
        "turn_start",
        "user_message",
        "turn_end",
    ]


def test_user_message_row_carries_timestamp() -> None:
    stamp = datetime.now(UTC)
    event = RuntimeEvent(
        type="user_message",
        run_id="r",
        payload={"message_id": "user-t1", "text": "hi"},
        timestamp=stamp,
    )
    state = reduce(initial_state(workspace="/tmp/project", model="fake"), event)
    row = state.transcript[-1]
    assert row.kind == "user"
    assert row.timestamp == stamp
