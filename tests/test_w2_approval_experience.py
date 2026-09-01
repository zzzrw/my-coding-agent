import asyncio
import json
import os

from textual.widgets import Input, Select, Static

from coding_agent.app import create_app
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.policy.memory import DecisionMemory, signature
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import LLMEvent, ToolCall
from coding_agent.runtime.runtime import _ApprovalBroker
from coding_agent.session.models import ApprovalRequest
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.filesystem import make_write_file_tool
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.state import initial_state
from coding_agent.tui.widgets import ApprovalScreen, render_approval_diff


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


def _tool_response(call_id, name, arguments):
    return [
        LLMEvent(type="tool_call_start", tool_call_id=call_id, tool_name=name),
        LLMEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=json.dumps(arguments),
        ),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]


class _SequencedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append((messages, tools, model))
        for event in self.responses.pop(0):
            if signal.is_set():
                return
            yield event


async def test_resolve_approval_accepts_remember_and_feedback(tmp_path):
    provider = _SequencedProvider(
        [
            _tool_response(
                "write-1", "write_file", {"path": "a.txt", "content": "new\n"}
            ),
            [
                LLMEvent(type="text_delta", text="ok"),
                LLMEvent(type="response_end", finish_reason="stop"),
            ],
        ]
    )
    application = create_app(
        workspace=tmp_path,
        model="fake-model",
        session_dir=tmp_path / "sessions",
        context_window=2_000,
        provider=provider,
        permission_mode="default",
    )
    runtime = application.runtime
    store = runtime.store

    approval_requested = asyncio.Event()
    run_finished = asyncio.Event()
    captured = {}

    async def observe(event):
        if event.type == "approval_requested":
            captured["request"] = event.payload["request"]
            approval_requested.set()
        elif event.type == "run_finished":
            run_finished.set()

    runtime.subscribe(observe)
    await runtime.submit("write a.txt")
    await asyncio.wait_for(approval_requested.wait(), timeout=5)

    request = captured["request"]
    await runtime.resolve_approval(
        request.request_id, "deny", remember="turn", feedback="no"
    )
    await asyncio.wait_for(run_finished.wait(), timeout=5)

    assert runtime.last_outcome is not None
    assert runtime.last_outcome.reason == "completed"
    # The deny feedback reaches the model through the tool result error.
    tool_result = next(
        record for record in store.records() if record.type == "tool_result"
    )
    error = tool_result.payload["result"].error
    assert "approval denied" in error
    assert "no" in error

    approvals = [record for record in store.records() if record.type == "approval"]
    assert len(approvals) == 1
    record = approvals[0]
    assert record.payload["request_id"] == request.request_id
    assert record.payload["tool_name"] == "write_file"
    assert record.payload["decision"] == "deny"
    assert record.payload["scope"] == "turn"
    assert record.payload["feedback"] == "no"
    assert record.payload["tool_call_id"] == request.tool_call_id


async def test_approval_resolved_event_carries_remember_and_feedback():
    events = []

    async def publish(event):
        events.append(event)

    broker = _ApprovalBroker(publish, lambda status: None)
    request = ApprovalRequest(
        request_id="a1",
        run_id="r1",
        tool_call_id="c1",
        tool_name="write_file",
        risk_level="mutate_file",
    )
    pending = asyncio.create_task(broker.request(request))
    await asyncio.sleep(0)
    await broker.resolve("a1", "deny", remember="session", feedback="relocate")
    assert await pending == "deny"
    resolved = next(event for event in events if event.type == "approval_resolved")
    assert resolved.payload["remember"] == "session"
    assert resolved.payload["feedback"] == "relocate"


# ---------------------------------------------------------------------------
# Task 4: approval diff preview + remember/feedback UI
# ---------------------------------------------------------------------------


def test_render_approval_diff_write(tmp_path):
    (tmp_path / "a.txt").write_text("old\n")
    req = ApprovalRequest(
        request_id="1",
        run_id="r",
        tool_call_id="c",
        tool_name="write_file",
        risk_level="mutate_file",
        arguments={"path": "a.txt", "content": "new\n"},
    )
    text = render_approval_diff(req, workspace=tmp_path)
    assert "+new" in str(text)
    assert "-old" in str(text)


def test_render_approval_diff_edit(tmp_path):
    (tmp_path / "a.txt").write_text("first\nsecond\n")
    req = ApprovalRequest(
        request_id="2",
        run_id="r",
        tool_call_id="c",
        tool_name="edit_file",
        risk_level="mutate_file",
        arguments={"path": "a.txt", "old_text": "first", "new_text": "changed"},
    )
    text = render_approval_diff(req, workspace=tmp_path)
    assert "-first" in str(text)
    assert "+changed" in str(text)


def test_render_approval_diff_missing_file(tmp_path):
    # No current file -> the whole proposed content is added lines.
    req = ApprovalRequest(
        request_id="3",
        run_id="r",
        tool_call_id="c",
        tool_name="write_file",
        risk_level="mutate_file",
        arguments={"path": "new.txt", "content": "hello\n"},
    )
    text = render_approval_diff(req, workspace=tmp_path)
    assert "+hello" in str(text)


def test_render_approval_diff_truncates_long(tmp_path):
    (tmp_path / "a.txt").write_text("".join(f"line {i}\n" for i in range(100)))
    proposed = "".join(f"changed {i}\n" for i in range(100))
    req = ApprovalRequest(
        request_id="4",
        run_id="r",
        tool_call_id="c",
        tool_name="write_file",
        risk_level="mutate_file",
        arguments={"path": "a.txt", "content": proposed},
    )
    text = render_approval_diff(req, workspace=tmp_path)
    assert "more lines" in str(text)


class _ApprovalRuntime:
    """Minimal runtime for exercising the ApprovalScreen through the app."""

    def __init__(self):
        self.resolved = []
        self.sink = None

    def subscribe(self, sink):
        self.sink = sink
        return lambda: None

    async def resolve_approval(
        self, request_id, decision, remember="once", feedback=None
    ):
        self.resolved.append((request_id, decision, remember, feedback))


def _diff_approval_request():
    return ApprovalRequest(
        request_id="a1",
        run_id="r1",
        tool_call_id="c1",
        tool_name="write_file",
        risk_level="mutate_file",
        arguments={"path": "a.txt", "content": "new\n"},
        reason="write requested",
    )


async def test_approval_screen_has_remember_and_feedback(tmp_path):
    runtime = _ApprovalRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state(str(tmp_path), "fake", context_window=1000),
    )

    async with app.run_test() as pilot:
        await runtime.sink(
            RuntimeEvent(
                type="approval_requested",
                run_id="r1",
                payload={"request": _diff_approval_request()},
            )
        )
        await pilot.pause()

        screen = pilot.app.screen
        assert isinstance(screen, ApprovalScreen)
        diff = screen.query_one("#approval-diff", Static)
        assert "+new" in str(diff.render())
        remember = screen.query_one("#approval-remember", Select)
        assert remember.value == "once"
        assert isinstance(screen.query_one("#approval-feedback", Input), Input)


async def test_approval_screen_deny_carries_remember_and_feedback(tmp_path):
    runtime = _ApprovalRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state(str(tmp_path), "fake", context_window=1000),
    )

    async with app.run_test() as pilot:
        await runtime.sink(
            RuntimeEvent(
                type="approval_requested",
                run_id="r1",
                payload={"request": _diff_approval_request()},
            )
        )
        await pilot.pause()

        screen = pilot.app.screen
        screen.query_one("#approval-remember", Select).value = "session"
        screen.query_one("#approval-feedback", Input).value = "use relative path"
        screen.action_deny()
        await pilot.pause()

    assert runtime.resolved == [("a1", "deny", "session", "use relative path")]


async def test_approval_screen_approve_uses_default_remember(tmp_path):
    runtime = _ApprovalRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state(str(tmp_path), "fake", context_window=1000),
    )

    async with app.run_test() as pilot:
        await runtime.sink(
            RuntimeEvent(
                type="approval_requested",
                run_id="r1",
                payload={"request": _diff_approval_request()},
            )
        )
        await pilot.pause()

        pilot.app.screen.action_approve()
        await pilot.pause()

    assert runtime.resolved == [("a1", "approve", "once", None)]
