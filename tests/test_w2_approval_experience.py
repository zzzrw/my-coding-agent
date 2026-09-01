import asyncio
import os

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.policy.memory import DecisionMemory, signature
from coding_agent.runtime.models import ToolCall
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.filesystem import make_write_file_tool
from coding_agent.tools.registry import ToolRegistry


def test_signature_normalizes_arguments():
    assert signature("write_file", {"path": "a", "content": "x"}) == (
        "write_file",
        '{"content": "x", "path": "a"}',
    )
    assert signature("write_file", {"content": "x", "path": "a"}) == signature(
        "write_file", {"path": "a", "content": "x"}
    )


def test_remember_and_lookup_scopes(tmp_path):
    mem = DecisionMemory(always_path=tmp_path / "approvals.json")
    sig = signature("run_command", {"command": "ls"})
    assert mem.lookup(sig) is None
    mem.remember(sig, "allow", scope="turn")
    assert mem.lookup(sig) == "allow"
    mem.clear_turn()
    assert mem.lookup(sig) is None
    mem.remember(sig, "deny", scope="session")
    assert mem.lookup(sig) == "deny"
    mem.clear_session()
    assert mem.lookup(sig) is None


def test_always_persists_to_file(tmp_path):
    path = tmp_path / "approvals.json"
    mem = DecisionMemory(always_path=path)
    sig = signature("write_file", {"path": "x"})
    mem.remember(sig, "allow", scope="always")
    mem.persist_always()
    loaded = DecisionMemory(always_path=path)
    loaded.load_always()
    assert loaded.lookup(sig) == "allow"
    assert (os.stat(path).st_mode & 0o777) == 0o600


class _RecordingBroker:
    def __init__(self):
        self.requests = []
        self.decision = "deny"

    async def request(self, request):
        self.requests.append(request)
        return self.decision

    def cancel_all(self):
        pass


def _write_tool_call():
    return ToolCall(
        id="c1", name="write_file", arguments={"path": "a.txt", "content": "new"}
    )


async def test_remembered_allow_short_circuits_broker(tmp_path):
    registry = ToolRegistry()
    registry.register(make_write_file_tool())
    broker = _RecordingBroker()
    memory = DecisionMemory()
    mem_sig = signature("write_file", {"path": "a.txt", "content": "new"})
    memory.remember(mem_sig, "allow", scope="session")
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker, memory=memory)
    result = await executor.execute(
        _write_tool_call(),
        run_id="r",
        workspace=tmp_path,
        permission_mode="default",
        signal=asyncio.Event(),
    )
    assert result.ok
    assert broker.requests == []  # no approval asked


async def test_remembered_deny_short_circuits_broker(tmp_path):
    registry = ToolRegistry()
    registry.register(make_write_file_tool())
    broker = _RecordingBroker()
    memory = DecisionMemory()
    mem_sig = signature("write_file", {"path": "a.txt", "content": "new"})
    memory.remember(mem_sig, "deny", scope="session")
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker, memory=memory)
    result = await executor.execute(
        _write_tool_call(),
        run_id="r",
        workspace=tmp_path,
        permission_mode="default",
        signal=asyncio.Event(),
    )
    assert not result.ok
    assert "approval denied" in (result.error or "")
    assert broker.requests == []  # no approval asked


async def test_deny_with_feedback_reaches_model(tmp_path):
    registry = ToolRegistry()
    registry.register(make_write_file_tool())
    broker = _RecordingBroker()
    broker.decision = "deny"
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker)
    result = await executor.execute(
        _write_tool_call(),
        run_id="r",
        workspace=tmp_path,
        permission_mode="default",
        signal=asyncio.Event(),
        remember="turn",
        feedback="use relative path",
    )
    assert not result.ok
    assert "approval denied" in (result.error or "")
    assert "use relative path" in (result.error or "")


async def test_deny_without_feedback_keeps_reason(tmp_path):
    registry = ToolRegistry()
    registry.register(make_write_file_tool())
    broker = _RecordingBroker()
    broker.decision = "deny"
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker)
    result = await executor.execute(
        _write_tool_call(),
        run_id="r",
        workspace=tmp_path,
        permission_mode="default",
        signal=asyncio.Event(),
    )
    assert not result.ok
    assert "approval denied: mutation requires approval" in (result.error or "")


async def test_decision_recorded_on_resolve(tmp_path):
    registry = ToolRegistry()
    registry.register(make_write_file_tool())
    broker = _RecordingBroker()
    broker.decision = "approve"
    memory = DecisionMemory()
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker, memory=memory)
    await executor.execute(
        _write_tool_call(),
        run_id="r",
        workspace=tmp_path,
        permission_mode="default",
        signal=asyncio.Event(),
        remember="turn",
    )
    assert (
        memory.lookup(signature("write_file", {"path": "a.txt", "content": "new"}))
        == "allow"
    )


async def test_decision_recorded_on_deny(tmp_path):
    registry = ToolRegistry()
    registry.register(make_write_file_tool())
    broker = _RecordingBroker()
    broker.decision = "deny"
    memory = DecisionMemory()
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker, memory=memory)
    await executor.execute(
        _write_tool_call(),
        run_id="r",
        workspace=tmp_path,
        permission_mode="default",
        signal=asyncio.Event(),
        remember="session",
    )
    assert (
        memory.lookup(signature("write_file", {"path": "a.txt", "content": "new"}))
        == "deny"
    )
