# Coding Agent MVP Implementation Plan

<!-- markdownlint-disable MD013 MD032 -->

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Python MVP: an interactive Textual TUI coding agent with a Pi-style Agent Loop, six local tools, JSONL sessions, deterministic context truncation, three permission modes, streaming events, and deterministic tests.

**Architecture:** The TUI talks only to `AgentRuntime`. `AgentRuntime` owns the active session, run task, approval broker, policy, and event subscribers; `AgentRunner` owns the turn/step model-tool loop. `LLMProvider`, `ToolRegistry`, `ToolExecutor`, `ContextPolicy`, and `SessionStore` are injected contracts so the runtime can be tested with a fake provider and temporary workspace.

**Tech Stack:** Python 3.11+, Pydantic 2, Textual 0.80+, OpenAI Python client for one OpenAI-compatible Chat Completions provider, `pytest`, `pytest-asyncio`, and `ruff`. No Agent framework or Agent SDK.

**Spec:** `docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md`

## Global Constraints

- Core agent logic must be implemented in this repository; do not use LangChain, LlamaIndex, OpenAI Agents SDK, Claude Agent SDK, AutoGen, CrewAI, or another Agent framework/SDK.
- The provider boundary is OpenAI-compatible Chat Completions; provider-specific wire shapes must not leak into `AgentRunner`.
- All durable and boundary models use Pydantic; all tool failures become structured `ToolResult(ok=False)` values.
- The TUI uses Textual but `AgentRuntime` and `AgentRunner` must not import Textual.
- The TUI layout is `Scrollable Transcript -> Input TextArea -> Bottom Statusline`.
- Only one run may be active at a time; `submit()` schedules an `asyncio.Task` and returns a `run_id` immediately.
- MVP tool calls execute sequentially; retain `is_parallel_safe` in `ToolSchema` for later read-only parallel execution.
- File writes use same-directory temporary files and atomic replacement; file paths use the mode-aware workspace policy.
- `default` and `workspace` require one-time approval for outside paths; `full` bypasses ordinary approval and path containment but never bypasses catastrophic-command denial.
- Resume restores conversation facts, workspace, and model; it starts with `default` permission and never replays old processes or tool side effects.
- Stream deltas are live events only; complete assistant messages and tool results are persisted, and incomplete tool calls are never executed.
- Default automated tests use Fake Provider and temporary workspaces. Real API
  calls are optional explicit live smoke checks or manual acceptance only.
- An opt-in DeepSeek live smoke test may be added, but it is never part of the
  default test suite or CI: require `RUN_LIVE_LLM_TESTS=1` plus
  `DEEPSEEK_API_KEY` in the process environment, use a minimal prompt with no
  workspace/tools, redact credentials from all output, and skip when the gate
  or key is absent. Never persist the key in source, `.env`, README, fixtures,
  session files, screenshots, or video.
- Keep assignment credentials out of Git, README files, and video; preserve double-blind anonymity in all deliverables.

Every implementation commit must use the repository Lore format. The short
intent line may match the task title, but the commit body must include all of:
`Constraint:`, `Rejected:`, `Confidence:`, `Scope-risk:`, `Directive:`,
`Tested:`, and `Not-tested:`. The command examples below use that convention;
the executor should fill the exact test names from the completed step.

---

## File Map

The implementation creates one installable package and focused test modules:

```text
pyproject.toml
src/coding_agent/
├── __init__.py
├── app.py
├── runtime/
│   ├── __init__.py
│   ├── models.py
│   ├── events.py
│   ├── hooks.py
│   ├── runner.py
│   └── runtime.py
├── llm/
│   ├── __init__.py
│   ├── protocol.py
│   └── openai_compatible.py
├── tools/
│   ├── __init__.py
│   ├── models.py
│   ├── registry.py
│   ├── executor.py
│   ├── filesystem.py
│   ├── search.py
│   └── shell.py
├── session/
│   ├── __init__.py
│   ├── models.py
│   └── store.py
├── context/
│   ├── __init__.py
│   ├── policy.py
│   └── truncate.py
├── policy/
│   ├── __init__.py
│   ├── approval.py
│   └── command.py
└── tui/
    ├── __init__.py
    ├── app.py
    ├── state.py
    ├── reducer.py
    ├── commands.py
    └── widgets.py
tests/
├── fakes.py
├── test_models.py
├── test_provider.py
├── test_live_provider.py
├── test_session.py
├── test_registry.py
├── test_tools_filesystem.py
├── test_tools_search.py
├── test_tools_shell.py
├── test_policy.py
├── test_executor.py
├── test_context.py
├── test_runner.py
├── test_runtime.py
├── test_reducer.py
├── test_tui.py
├── test_events.py
└── test_integration_flow.py
README.txt
```

The package layout is intentionally smaller than Pi or DeepSeek Harness. It
keeps their useful boundaries without introducing a plugin runtime, multi-lane
session engine, or OS sandbox.

### Test helper contract

The snippets below use deterministic helpers defined in the named test modules;
these are test-only utilities, not hidden production APIs:

```text
tests/fakes.py:
    FakeProvider(events), RepeatingToolProvider(tool_name),
    BlockingFakeProvider(), ApprovalBlockingProvider(),
    FakeTool(name), RecordingTool(name), FailingTool(), LargeOutputTool()
    SlowTool()
    assistant_with_tool(name, arguments), assistant_text(text),
    assistant_with_finish_reason(reason), assistant_with_truncated_tool_call(name),
    malformed_tool_call(name, raw_arguments)

tests/test_context.py:
    history, large_current_turn, SYSTEM, contains_dangling_tool_result()

tests/test_executor.py:
    make_executor(tool, approval_answer=None)

tests/test_runner.py:
    make_runner(provider, tools=None, max_steps=20)

tests/test_runtime.py:
    make_runtime(provider=None, workspace=None, session_root=None,
                 permission_mode="default"), wait_for_status(),
    wait_for_idle()

tests/test_reducer.py:
    initial_state(workspace, model), event(type, payload), approval_request()

tests/test_tui.py:
    make_app(session_summaries=None), summary(session_id)

tests/test_policy.py:
    read_schema, write_schema, shell_schema

tests/test_models.py:
    SYSTEM and model fixtures are local to the relevant test module.

tests/test_runner.py:
    make_runner() seeds the prompt record because AgentRuntime, not
    AgentRunner.run_turn(), owns the initial user_message append.
```

Each helper must be implemented before the test that first uses it and must
remain deterministic, network-free, and independent of Textual unless it is a
Pilot test helper.

## Task 1: Bootstrap the Python Package

**Files:**
- Create: `pyproject.toml`
- Create: `src/coding_agent/__init__.py`
- Create: `src/coding_agent/runtime/__init__.py`
- Create: `src/coding_agent/llm/__init__.py`
- Create: `src/coding_agent/tools/__init__.py`
- Create: `src/coding_agent/session/__init__.py`
- Create: `src/coding_agent/context/__init__.py`
- Create: `src/coding_agent/policy/__init__.py`
- Create: `src/coding_agent/tui/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Produces an installable `coding_agent` package and a pytest configuration for async tests.
- Does not yet expose runtime behavior.

- [ ] **Step 1: Write the failing import test**

```python
def test_package_imports():
    import coding_agent

    assert coding_agent.__version__
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_bootstrap.py -q`

Expected: FAIL because the package metadata and module do not exist.

- [ ] **Step 3: Add package metadata and dependencies**

Create `pyproject.toml` with:

```toml
[project]
name = "coding-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "openai>=2.50.0",
  "pydantic>=2.0",
  "textual>=0.80",
]

[project.scripts]
coding-agent = "coding_agent.app:main"

[dependency-groups]
dev = [
  "pytest>=8.0,<10.0",
  "pytest-asyncio>=0.23,<2.0",
  "ruff>=0.8,<1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/coding_agent"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["live: opt-in network smoke test; requires explicit environment gate"]
```

Create `src/coding_agent/__init__.py` with `__version__ = "0.1.0"` and empty
subpackage `__init__.py` files so every planned import path is an explicit
package.

- [ ] **Step 4: Run the test and package checks**

Run: `pytest tests/test_bootstrap.py -q`

Expected: PASS.

Run: `ruff check src tests`

Expected: PASS once the empty package is checked.

- [ ] **Step 5: Commit the bootstrap**

```bash
git add pyproject.toml src/coding_agent/__init__.py tests/conftest.py tests/test_bootstrap.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 2: Define Pydantic Domain Contracts

**Files:**
- Create: `src/coding_agent/runtime/models.py`
- Create: `src/coding_agent/runtime/events.py`
- Create: `src/coding_agent/tools/models.py`
- Create: `src/coding_agent/session/models.py`
- Test: `tests/test_models.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces `Message`, `ToolCall`, `Usage`, `LLMEvent`, `ToolSchema`, `ToolResult`, `SessionHeader`, `SessionRecord`, `SessionMessage`, `ContextView`, `SessionSummary`, `ApprovalRequest`, `RuntimeStatus`, `RuntimeEvent`, and `EventSink` before any runner or UI task begins.
- Later tasks import these models instead of creating ad hoc dictionaries.

- [ ] **Step 1: Write model validation tests**

```python
from datetime import datetime

import pytest
from pydantic import ValidationError

from coding_agent.runtime.models import Message, ToolCall, Usage
from coding_agent.tools.models import ToolResult, ToolSchema
from coding_agent.session.models import SessionRecord


def test_message_uses_internal_tool_call_shape():
    message = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
    )
    assert message.tool_calls[0].arguments == {"path": "a.py"}


def test_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        Usage(total_tokens=3, unexpected=1)


def test_session_record_requires_sequence_and_type():
    record = SessionRecord(
        id="r1",
        seq=0,
        timestamp=datetime.now(),
        type="user_message",
        payload={"message": Message(role="user", content="hi")},
    )
    assert record.parent_id is None


def test_tool_result_has_structured_error_fields():
    result = ToolResult(
        tool_call_id="c1",
        tool_name="read_file",
        ok=False,
        content="",
        error="missing",
    )
    assert result.ok is False


def test_provider_event_and_runtime_status_have_boundary_fields():
    event = LLMEvent(type="text_delta", text="hi")
    status = RuntimeStatus(status="idle")
    assert event.text == "hi"
    assert status.status == "idle"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest tests/test_models.py -q`

Expected: FAIL because the model modules do not exist.

- [ ] **Step 3: Implement the Pydantic contracts**

Use `ConfigDict(extra="forbid")` for all external/durable models. Define:

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    ok: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Define the normalized provider event model used by Task 3:

```python
class LLMEvent(BaseModel):
    type: Literal["text_delta", "tool_call_start", "tool_call_delta", "tool_call_end", "response_end", "error"]
    text: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    error: str | None = None
```

Also define the runtime status projection consumed by the TUI:

```python
class RuntimeStatus(BaseModel):
    status: Literal["idle", "running", "waiting_approval", "error", "aborted"] = "idle"
    run_id: str | None = None
    turn_id: str | None = None
    usage: Usage | None = None
    context_used: int = 0
    context_window: int | None = None
    context_estimated: bool = False
```

Use `ConfigDict(extra="forbid")` validation tests for both models, and keep
all optional event fields nullable so a parser can emit the smallest valid
event for each stream chunk.

`ToolSchema` has `name`, `description`, `parameters`, `risk_level`, and
`is_parallel_safe`. `SessionRecord` has `id`, `seq`, `timestamp`, the approved
record type literal, `parent_id`, `run_id`, `turn_id`, and `payload`. Define
`SessionMessage(record_id, turn_id, seq, message)` and `ContextView` with
`messages`, `used_tokens`, `context_window`, `estimated`, `compacted`,
`removed_turns`, and `overflow`.

Also define the runner result used by later tests:

```python
class TurnOutcome(BaseModel):
    reason: Literal["completed", "max_steps", "aborted", "provider_error", "session_error"]
    final_text: str = ""
    steps: int = 0
    usage: Usage | None = None
```

Define the persisted session header separately from conversation records:

```python
class SessionHeader(BaseModel):
    kind: Literal["header"] = "header"
    schema_version: int = 1
    session_id: str
    workspace: str
    model: str
    title: str = "New session"
    created_at: datetime
    updated_at: datetime
    context_window: int
```

Define `RuntimeEvent` with the approved event literal set and define
`EventSink = Callable[[RuntimeEvent], Awaitable[None]]` in
`runtime/events.py`. Add an event-envelope validation test in
`tests/test_events.py`; Task 10 can then consume this module without a
temporary contract.

- [ ] **Step 4: Run tests and type/lint checks**

Run: `pytest tests/test_models.py -q`

Expected: PASS.

Run: `ruff check src/coding_agent/runtime src/coding_agent/tools src/coding_agent/session tests/test_models.py`

Expected: PASS.

Run: `pytest tests/test_events.py -q`

Expected: PASS; the event contract is now available to later tasks.

- [ ] **Step 5: Commit the contracts**

```bash
git add src/coding_agent/runtime/models.py src/coding_agent/runtime/events.py src/coding_agent/tools/models.py src/coding_agent/session/models.py tests/test_models.py tests/test_events.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 3: Implement the Provider Boundary and Fake Provider

**Files:**
- Create: `src/coding_agent/llm/protocol.py`
- Create: `src/coding_agent/llm/openai_compatible.py`
- Create: `tests/fakes.py`
- Test: `tests/test_provider.py`
- Test: `tests/test_live_provider.py` (opt-in only)

**Interfaces:**
- Consumes: `Message`, `ToolSchema`, `LLMEvent`, and `Usage` from Task 2.
- Produces:

```python
class LLMProvider(Protocol):
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        model: str,
        signal: asyncio.Event,
    ) -> AsyncIterator[LLMEvent]: ...
```

`OpenAICompatibleProvider` converts Chat Completions SSE chunks into
`text_delta`, `tool_call_start`, `tool_call_delta`, `tool_call_end`,
`response_end`, and `error` events. `FakeProvider` yields a configured event
sequence and records every request for deterministic assertions.

The parser keeps a map from stream `index` to call id. A later chunk that has
only an index reuses the id/name from `tool_call_start`. `[DONE]` closes the
stream without fabricating content. A final usage chunk is attached to
`response_end`. A `finish_reason="length"` is preserved so the runner can
reject every incomplete tool call.

- [ ] **Step 1: Write provider conversion tests**

```python
import asyncio

from coding_agent.llm.openai_compatible import ChatChunkParser
from coding_agent.runtime.models import LLMEvent


def test_text_chunk_becomes_text_delta():
    event = ChatChunkParser().parse({"choices": [{"delta": {"content": "hello"}}]})
    assert event == LLMEvent(type="text_delta", text="hello")


def test_tool_argument_chunks_preserve_call_identity():
    event = ChatChunkParser().parse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "read_file", "arguments": "{\"pa"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert event.type == "tool_call_delta"
    assert event.tool_call_id == "call-1"
    assert event.arguments_delta == '{"pa'


def test_tool_start_and_end_events_are_explicit():
    parser = ChatChunkParser()
    start = parser.parse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file"}}]}}]})
    end = parser.parse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    assert start.type == "tool_call_start"
    assert start.tool_call_id == "c1"
    assert end.type == "tool_call_end"


def test_malformed_tool_arguments_are_reported_by_aggregation_fixture():
    provider = FakeProvider([
        LLMEvent(type="tool_call_start", tool_call_id="c1", tool_name="read_file"),
        LLMEvent(type="tool_call_delta", tool_call_id="c1", arguments_delta='{"path":'),
        LLMEvent(type="response_end", finish_reason="tool_calls"),
    ])
    assert provider.events[1].arguments_delta == '{"path":'


@pytest.mark.asyncio
async def test_fake_provider_yields_events_without_network():
    provider = FakeProvider([
        LLMEvent(type="text_delta", text="done"),
        LLMEvent(type="response_end", finish_reason="stop"),
    ])
    events = [event async for event in provider.stream([], [], model="fake", signal=asyncio.Event())]
    assert [event.type for event in events] == ["text_delta", "response_end"]
```

- [ ] **Step 2: Run tests and verify conversion fails**

Run: `pytest tests/test_provider.py -q`

Expected: FAIL because provider modules do not exist.

- [ ] **Step 3: Implement normalized provider events**

Keep provider-specific response parsing in `openai_compatible.py`. Parse
function arguments as raw strings and emit deltas; do not call tools or
construct `ToolResult` in the provider. Resolve API key/base URL/model from
explicit constructor configuration or environment variables, never from a
repository-local untrusted file.

- [ ] **Step 4: Implement FakeProvider and run tests**

`FakeProvider` must copy the input lists before storing them, check
`signal.is_set()` between events, and yield an `error` event only when the test
fixture explicitly includes one.

Run: `pytest tests/test_provider.py -q`

Expected: PASS.

- [ ] **Step 5: Add the opt-in live provider smoke test**

Create a `@pytest.mark.live` test that exits with `pytest.skip` unless both
`RUN_LIVE_LLM_TESTS=1` and `DEEPSEEK_API_KEY` are present. Construct the same
`OpenAICompatibleProvider` using an environment-configured base URL/model (with
DeepSeek defaults documented in the test), send only a fixed minimal prompt,
assert at least one normalized event and a terminating `response_end`, and
never include the key or full response in assertion messages. Keep this test
separate from all Fake Provider tests and do not execute it in CI by default.

Run the live check only when explicitly requested:

```bash
RUN_LIVE_LLM_TESTS=1 DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  pytest -m live tests/test_live_provider.py -q
```

The command must be process-scoped; do not add the key to shell startup files
or any repository-local file.

- [ ] **Step 6: Commit the provider boundary**

```bash
git add src/coding_agent/llm tests/fakes.py tests/test_provider.py tests/test_live_provider.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 4: Build the JSONL Session Store and Message Projection

**Files:**
- Modify: `src/coding_agent/session/models.py`
- Create: `src/coding_agent/session/store.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `SessionHeader`, `SessionRecord`, `SessionMessage`, `Message`, and `SessionSummary`.
- Produces:

```python
class SessionStore:
    @classmethod
    def create(
        cls,
        root: Path,
        *,
        session_id: str | None = None,
        workspace: str,
        model: str,
        context_window: int,
        title: str = "New session",
    ) -> "SessionStore": ...
    @classmethod
    def open(cls, root: Path, session_id: str) -> "SessionStore": ...
    @property
    def path(self) -> Path: ...
    @property
    def load_notice(self) -> str | None: ...
    @property
    def header(self) -> SessionHeader: ...
    def append(self, record: SessionRecord) -> None: ...
    def append_new(
        self,
        record_type: str,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
        turn_id: str | None = None,
    ) -> SessionRecord: ...
    def records(self) -> list[SessionRecord]: ...
    def project_messages(self) -> list[SessionMessage]: ...
    @classmethod
    def list_sessions(cls, root: Path) -> list[SessionSummary]: ...
```

The store file is `<session_root>/<session_id>.jsonl`; test fixtures use a
separate `session_root` and workspace directory so session files are never
inside the project being edited. The first line is a
validated `SessionHeader` object with `kind="header"`; subsequent lines are
`SessionRecord` objects. The store writes one JSON object per line through a
single append method, assigns monotonically increasing `seq` values and parent
ids, flushes the append, and rejects malformed records. The header is the
source for `workspace`, `model`, `title`, timestamps, and `context_window` used
by `list_sessions()` and `resume()`.

- [ ] **Step 1: Write session tests**

```python
def test_append_assigns_sequence_and_parent(tmp_path):
    store = SessionStore.create(tmp_path, workspace=str(tmp_path), model="fake", context_window=1000)
    first = store.append_new("user_message", {"message": Message(role="user", content="hi")})
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


def test_projection_does_not_duplicate_audit_tool_call(tmp_path):
    store = SessionStore.create(tmp_path, workspace=str(tmp_path), model="fake", context_window=1000)
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
    )
    store.append_new("assistant_message", {"message": assistant, "complete": True}, turn_id="t1")
    store.append_new("tool_call", {"tool_call": assistant.tool_calls[0]}, turn_id="t1")
    store.append_new(
        "tool_result",
        {"result": ToolResult(tool_call_id="c1", tool_name="read_file", ok=True, content="x")},
        turn_id="t1",
    )
    projected = store.project_messages()
    assert [item.message.role for item in projected] == ["assistant", "tool"]


def test_projection_rejects_duplicate_or_unmatched_tool_results(tmp_path):
    store = SessionStore.create(tmp_path, workspace=str(tmp_path), model="fake", context_window=1000)
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
    )
    store.append_new("assistant_message", {"message": assistant, "complete": True}, turn_id="t1")
    store.append_new(
        "tool_result",
        {"result": ToolResult(tool_call_id="unknown", tool_name="read_file", ok=True, content="x")},
        turn_id="t1",
    )
    with pytest.raises(ValueError, match="tool_result"):
        store.project_messages()


def test_open_turn_is_marked_interrupted_without_replay(tmp_path):
    store = SessionStore.create(tmp_path, workspace=str(tmp_path), model="fake", context_window=1000)
    store.append_new("turn_start", {"turn_id": "t1"}, turn_id="t1")
    reopened = SessionStore.open(tmp_path, store.session_id)
    assert reopened.has_interrupted_turn()
    assert reopened.project_messages() == []


def test_malformed_final_line_is_usable_and_reports_notice(tmp_path):
    store = SessionStore.create(tmp_path, workspace=str(tmp_path), model="fake", context_window=1000)
    store.append_new("user_message", {"message": Message(role="user", content="hi")})
    store.path.write_text(store.path.read_text(encoding="utf-8") + "{bad json\n", encoding="utf-8")
    reopened = SessionStore.open(tmp_path, store.session_id)
    assert len(reopened.records()) == 1
    assert "corrupt" in (reopened.load_notice or "")
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_session.py -q`

Expected: FAIL because `SessionStore` is not implemented.

- [ ] **Step 3: Implement append, load, and list behavior**

Use a per-store lock around append. Store files under a user/session directory,
not inside the target workspace. The first JSONL line is a `SessionHeader` with
`kind="header"`, `schema_version`, `session_id`, `workspace`, `model`, `title`,
`created_at`, `updated_at`, and `context_window`; each later line is a
`SessionRecord`. `list_sessions()` reads only valid headers, sorts by
`updated_at` descending, and returns a bounded title derived from the first user
prompt when the header still has the default title. A malformed final JSONL
line is ignored and reported as an interrupted/corrupt notice; earlier valid
records remain usable. Invalid headers fail closed and are excluded from the
selector.

Expose a read-only `load_notice: str | None` on an opened store. A malformed
final JSONL line is ignored only after all earlier lines validate; `open()` sets
`load_notice` to a corruption/interruption notice, while malformed headers or
non-final malformed records fail closed. `AgentRuntime.resume()` republishes
that notice as a `notice` event so the TUI can show it.

- [ ] **Step 4: Implement projection invariants**

Project only `user_message`, complete `assistant_message`, and `tool_result`.
Treat `tool_call` as audit-only. Require each projected tool result to match a
tool call in the preceding assistant message and occur at most once. Exclude
an incomplete assistant/tool pair from resumable context. Keep system prompt
outside the session message projection; `AgentRunner` prepends the configured
prompt later.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_session.py -q`

Expected: PASS.

```bash
git add src/coding_agent/session tests/test_session.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 5: Implement Tool Registry and Tool Context

**Files:**
- Create: `src/coding_agent/tools/registry.py`
- Modify: `src/coding_agent/tools/models.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `ToolSchema` and `ToolResult`.
- Produces:

```python
class ToolContext(BaseModel):
    workspace: Path
    permission_mode: Literal["default", "workspace", "full"]
    allow_outside_once: bool = False

class Tool(Protocol):
    schema: ToolSchema
    args_model: type[BaseModel]
    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolContext,
        signal: asyncio.Event,
    ) -> ToolResult: ...

class ToolRegistry:
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool | None: ...
    def schemas(self) -> list[ToolSchema]: ...
```

`ToolContext.allow_outside_once` is set only after an approval decision for a
single call. This is the explicit implementation of the approved
default/workspace outside-path behavior; it avoids a hidden global bypass.

- [ ] **Step 1: Write registry tests**

```python
@pytest.mark.asyncio
async def test_registry_rejects_duplicate_names():
    registry = ToolRegistry()
    registry.register(FakeTool("read_file"))
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(FakeTool("read_file"))


def test_registry_schema_order_is_registration_order():
    registry = ToolRegistry()
    registry.register(FakeTool("read_file"))
    registry.register(FakeTool("write_file"))
    assert [schema.name for schema in registry.schemas()] == ["read_file", "write_file"]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_registry.py -q`

Expected: FAIL because registry modules do not exist.

- [ ] **Step 3: Implement registry and context**

Use a private ordered dictionary keyed by tool name. Each tool's
`args_model` is the Pydantic class used to validate raw arguments; its JSON
schema is exposed through `ToolSchema.parameters`. Return defensive copies of
schema lists. Do not execute tools from the registry; execution belongs to
Task 8's `ToolExecutor`.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/test_registry.py -q`

Expected: PASS.

```bash
git add src/coding_agent/tools/registry.py src/coding_agent/tools/models.py tests/test_registry.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 6: Implement Filesystem and Search Tools

**Files:**
- Create: `src/coding_agent/tools/filesystem.py`
- Create: `src/coding_agent/tools/search.py`
- Create: `src/coding_agent/tools/shell.py`
- Test: `tests/test_tools_filesystem.py`
- Test: `tests/test_tools_search.py`
- Test: `tests/test_tools_shell.py`

**Interfaces:**
- Consumes: `Tool`, `ToolContext`, `ToolSchema`, and `ToolResult`.
- Produces all six built-in tool factories:

```python
def make_read_file_tool() -> Tool: ...
def make_write_file_tool() -> Tool: ...
def make_edit_file_tool() -> Tool: ...
def make_list_files_tool() -> Tool: ...
def make_grep_files_tool() -> Tool: ...
def make_run_command_tool() -> Tool: ...
```

Use one path helper:

```python
def resolve_tool_path(
    workspace: Path,
    user_path: str,
    *,
    permission_mode: PermissionMode,
    allow_outside_once: bool,
) -> Path: ...
```

- [ ] **Step 1: Write filesystem and search tests**

```python
@pytest.mark.asyncio
async def test_read_file_is_line_bounded(tmp_path):
    path = tmp_path / "main.py"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    result = await make_read_file_tool().execute(
        {"path": "main.py", "start_line": 2, "end_line": 2},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert "b" in result.content
    assert "a" not in result.content


@pytest.mark.asyncio
async def test_edit_requires_exactly_one_match(tmp_path):
    (tmp_path / "a.txt").write_text("x\nx\n", encoding="utf-8")
    result = await make_edit_file_tool().execute(
        {"path": "a.txt", "old_text": "x", "new_text": "y"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "exactly once" in (result.error or "")


@pytest.mark.asyncio
async def test_path_escape_is_rejected_without_override(tmp_path):
    result = await make_read_file_tool().execute(
        {"path": "../outside.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="workspace"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "workspace" in (result.error or "")


@pytest.mark.asyncio
async def test_list_and_grep_skip_build_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("needle\n", encoding="utf-8")
    context = ToolContext(workspace=tmp_path, permission_mode="full")
    listing = await make_list_files_tool().execute({}, context=context, signal=asyncio.Event())
    matches = await make_grep_files_tool().execute(
        {"pattern": "needle"}, context=context, signal=asyncio.Event()
    )
    assert "node_modules" not in listing.content
    assert "src/main.py" in matches.content
    assert "ignored.js" not in matches.content


@pytest.mark.asyncio
async def test_run_command_uses_workspace_cwd_and_timeout(tmp_path):
    result = await make_run_command_tool().execute(
        {"command": "pwd", "timeout_seconds": 5},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert str(tmp_path) in result.content
    assert result.metadata["exit_code"] == 0
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_tools_filesystem.py tests/test_tools_search.py tests/test_tools_shell.py -q`

Expected: FAIL because the tool factories do not exist.

- [ ] **Step 3: Implement safe path and file operations**

`read_file` uses 1-based inclusive lines and a bounded result. `write_file` and
`edit_file` use same-directory temp files, flush/fsync, and `os.replace`.
`edit_file` refuses zero or multiple matches. For `default`/`workspace`, an
outside path fails until `allow_outside_once=True`; `full` accepts it.

`run_command` executes `/bin/sh -c` with `cwd=context.workspace`, captures
stdout/stderr, returns exit code and elapsed time, applies the per-call timeout,
and starts a process group. It concurrently watches `signal.is_set()`; on
cancellation it sends SIGTERM to the process group, waits a short grace period,
then sends SIGKILL if needed, awaits process cleanup, and returns a cancelled
`ToolResult` without leaving descendants behind.

- [ ] **Step 4: Implement bounded list and grep**

Use deterministic sorted traversal, skip `.git`, `node_modules`, virtualenvs,
`__pycache__`, `dist`, `build`, `.next`, and `target`, do not follow symlinks,
skip binary files, cap entries/results, and return structured errors.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_tools_filesystem.py tests/test_tools_search.py tests/test_tools_shell.py -q`

Expected: PASS.

```bash
git add src/coding_agent/tools/filesystem.py src/coding_agent/tools/search.py src/coding_agent/tools/shell.py tests/test_tools_filesystem.py tests/test_tools_search.py tests/test_tools_shell.py
git commit  # use the Lore-format body described in Global Constraints
```

The shell test module must also cover timeout and cancellation: start a command
that spawns a child, set the cancellation event, await the result, and assert a
cancelled bounded error plus completed process-group cleanup.

## Task 7: Implement Command Classification and Permission Modes

**Files:**
- Create: `src/coding_agent/policy/command.py`
- Create: `src/coding_agent/policy/approval.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `ToolSchema`, command text, workspace path, and active permission mode.
- Produces:

```python
PermissionMode = Literal["default", "workspace", "full"]
DecisionKind = Literal["allow", "ask", "deny"]

class PermissionDecision(BaseModel):
    kind: DecisionKind
    reason: str
    category: str
    allow_outside_once: bool = False

class ApprovalPolicy(Protocol):
    def decide(
        self,
        tool: ToolSchema,
        arguments: dict[str, Any],
        *,
        workspace: Path,
        mode: PermissionMode,
    ) -> PermissionDecision: ...

class CommandClassification(BaseModel):
    catastrophic: bool = False
    outside_or_unknown: bool = False
    reason: str = ""

def classify_command(command: str) -> CommandClassification: ...
```

- [ ] **Step 1: Write policy tests**

```python
def test_default_allows_read_and_asks_for_mutations():
    policy = DefaultApprovalPolicy()
    assert policy.decide(read_schema, {}, workspace=Path("."), mode="default").kind == "allow"
    assert policy.decide(write_schema, {}, workspace=Path("."), mode="default").kind == "ask"


def test_workspace_allows_internal_write_but_asks_for_outside_path(tmp_path):
    policy = DefaultApprovalPolicy()
    inside = {"path": "src/index.html"}
    outside = {"path": "../secret.txt"}
    assert policy.decide(write_schema, inside, workspace=tmp_path, mode="workspace").kind == "allow"
    decision = policy.decide(write_schema, outside, workspace=tmp_path, mode="workspace")
    assert decision.kind == "ask"
    assert decision.allow_outside_once is True


def test_full_allows_ordinary_tools_but_catastrophic_shell_is_denied():
    policy = DefaultApprovalPolicy()
    assert policy.decide(write_schema, {"path": "/tmp/x"}, workspace=Path("."), mode="full").kind == "allow"
    decision = policy.decide(shell_schema, {"command": "rm -rf /"}, workspace=Path("."), mode="full")
    assert decision.kind == "deny"


def test_unknown_workspace_shell_requires_approval():
    decision = classify_command("python -c 'open(\"/tmp/out\", \"w\").write(\"x\")'")
    assert decision.outside_or_unknown is True
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_policy.py -q`

Expected: FAIL because policy modules do not exist.

- [ ] **Step 3: Implement explicit command rules**

Implement the approved catastrophic patterns for root removal, `mkfs`/device
writes, shutdown/reboot, force push, hard reset, destructive clean, and fork
bombs. Classify absolute paths, `..`, `cd`, `git -C`, external redirection,
command substitution, unknown scripts, and inline file-writing code as
`outside_or_unknown`. Unknown syntax is `ask` except for catastrophic matches,
which are always `deny`.

- [ ] **Step 4: Implement the three modes**

`default`: read allow, write/edit/shell ask, outside ask, catastrophic deny.
`workspace`: internal read/write/edit and simple safe shell allow, outside or
unknown ask, catastrophic deny. `full`: ordinary tools and outside paths allow,
catastrophic deny.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_policy.py -q`

Expected: PASS.

```bash
git add src/coding_agent/policy tests/test_policy.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 8: Implement ToolExecutor and Approval Broker

**Files:**
- Create: `src/coding_agent/tools/executor.py`
- Create: `src/coding_agent/runtime/hooks.py`
- Modify: `src/coding_agent/runtime/models.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: `ToolRegistry`, `ApprovalPolicy`, `ToolContext`, `ToolCall`, and cancellation event.
- Produces:

```python
class ApprovalBroker(Protocol):
    async def request(self, request: ApprovalRequest) -> Literal["approve", "deny"]: ...
    def cancel_all(self) -> None: ...

class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: ApprovalPolicy,
        broker: ApprovalBroker,
        hooks: HookSet | None = None,
        default_timeout_seconds: float = 120.0,
    ) -> None: ...

    async def execute(
        self,
        call: ToolCall,
        *,
        run_id: str,
        workspace: Path,
        permission_mode: PermissionMode,
        signal: asyncio.Event,
    ) -> ToolResult: ...
```

`runtime/hooks.py` defines the minimum non-plugin hook set: async
`before_model`, `before_tool`, `after_tool`, and `on_error` callbacks. A hook
may return a replacement or rejection value, but hook execution is ordered and
does not create a dynamic plugin lifecycle.

Expose `MAX_TOOL_OUTPUT_CHARS` as the single result cap used by the executor
and its tests.

`AgentRuntime` supplies the broker. The executor is the only path that calls a
tool and performs argument validation, policy, approval, timeout, cancellation,
exception conversion, after-hook, and output truncation.

Outside-path approval is represented only by
`PermissionDecision.allow_outside_once=True`. After the broker approves that
request, `ToolExecutor` copies the active context with
`ToolContext.allow_outside_once=True` for that exact call. The flag is neither
stored globally nor reused by a later call; `resolve_tool_path` only consumes
the flag and never asks for approval itself.

- [ ] **Step 1: Write executor tests**

```python
@pytest.mark.asyncio
async def test_tool_exception_becomes_error_result():
    executor = make_executor(FailingTool())
    result = await executor.execute(
        ToolCall(id="c1", name="fail", arguments={}),
        run_id="r1", workspace=Path("."), permission_mode="full", signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_denied_approval_does_not_call_tool():
    tool = RecordingTool("write_file")
    executor = make_executor(tool, approval_answer="deny")
    result = await executor.execute(
        ToolCall(id="c1", name="write_file", arguments={"path": "a", "content": "x"}),
        run_id="r1", workspace=Path("."), permission_mode="default", signal=asyncio.Event(),
    )
    assert result.ok is False
    assert tool.calls == []


@pytest.mark.asyncio
async def test_approved_outside_path_is_one_call_only():
    tool = RecordingTool("write_file")
    executor = make_executor(tool, approval_answer="approve")
    call = ToolCall(id="c1", name="write_file", arguments={"path": "../outside", "content": "x"})
    result = await executor.execute(call, run_id="r1", workspace=Path("/tmp/project"), permission_mode="workspace", signal=asyncio.Event())
    assert result.ok is True
    assert tool.calls == [{"path": "../outside", "content": "x"}]


@pytest.mark.asyncio
async def test_output_is_bounded():
    result = await make_executor(LargeOutputTool()).execute(
        ToolCall(id="c1", name="read_file", arguments={}),
        run_id="r1", workspace=Path("."), permission_mode="full", signal=asyncio.Event(),
    )
    assert len(result.content) <= MAX_TOOL_OUTPUT_CHARS
    assert result.metadata["truncated"] is True


@pytest.mark.asyncio
async def test_abort_cancels_a_slow_tool_and_returns_cancelled_error():
    signal = asyncio.Event()
    executor = make_executor(SlowTool())
    task = asyncio.create_task(executor.execute(
        ToolCall(id="c1", name="slow", arguments={}),
        run_id="r1", workspace=Path("."), permission_mode="full", signal=signal,
    ))
    await asyncio.sleep(0)
    signal.set()
    result = await task
    assert result.ok is False
    assert result.metadata["cancelled"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_executor.py -q`

Expected: FAIL because executor and broker do not exist.

- [ ] **Step 3: Implement hook set, approval request, and broker**

Implement `HookSet` with ordered async callbacks and default no-op hooks. Use
one pending approval request at a time. `request_id` is unique. The public
`AgentRuntime.resolve_approval()` raises `RuntimeError("approval not pending")`
for unknown or already-resolved ids; an internal decision arriving after
`abort()` is ignored. `cancel_all()` resolves the pending request as cancelled
and wakes the executor without restarting the run. Add tests for both the
public unknown-id error and the internal late-decision race.

- [ ] **Step 4: Implement execution, timeout, and cancellation pipeline**

Resolve the tool, validate its arguments with its Pydantic args model, call the
approval policy, request approval when needed, construct `ToolContext` with a
one-call outside override when approved, execute with a timeout, catch every
exception, invoke the before/after hooks, and truncate combined output. Wrap
each tool in `asyncio.wait_for`; classify timeout as a bounded error result.
`run_command` must create a process group, send SIGTERM on cancellation, then
SIGKILL after a short grace period, and await process cleanup. Never raise
ordinary tool errors to `AgentRunner`.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_executor.py -q`

Expected: PASS.

```bash
git add src/coding_agent/tools/executor.py src/coding_agent/runtime/models.py tests/test_executor.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 9: Implement Deterministic ContextPolicy

**Files:**
- Create: `src/coding_agent/context/policy.py`
- Create: `src/coding_agent/context/truncate.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `list[SessionMessage]`, configured system prompt, `Usage`, and context window.
- Produces:

```python
class ContextPolicy(Protocol):
    def prepare(
        self,
        history: list[SessionMessage],
        *,
        system_prompt: Message,
        context_window: int,
        usage: Usage | None,
        force: bool = False,
    ) -> ContextView: ...

class TruncatePolicy(ContextPolicy):
    @staticmethod
    def estimate_tokens(messages: list[Message]) -> int: ...
```

- [ ] **Step 1: Write context tests**

```python
def test_under_budget_preserves_all_messages():
    view = TruncatePolicy(budget=1000).prepare(history, system_prompt=SYSTEM, context_window=1000, usage=None)
    assert view.compacted is False
    assert view.messages[0] == SYSTEM


def test_over_budget_removes_complete_turns_only():
    view = TruncatePolicy(budget=20).prepare(history, system_prompt=SYSTEM, context_window=20, usage=None)
    assert view.compacted is True
    assert view.messages[0] == SYSTEM
    assert not contains_dangling_tool_result(view.messages)


def test_force_under_budget_compacts_when_a_complete_turn_can_be_removed():
    view = TruncatePolicy(budget=1000).prepare(history, system_prompt=SYSTEM, context_window=1000, usage=None, force=True)
    assert view.compacted is True


def test_compaction_records_removed_turn_count_and_marker():
    view = TruncatePolicy(budget=20).prepare(history, system_prompt=SYSTEM, context_window=20, usage=None)
    assert view.removed_turns >= 1
    assert any(message.role == "system" and "compacted" in (message.content or "") for message in view.messages)


def test_current_turn_overflow_is_reported():
    view = TruncatePolicy(budget=10).prepare(large_current_turn, system_prompt=SYSTEM, context_window=10, usage=None)
    assert view.overflow is True


def test_usage_fallback_is_deterministic_and_marked_estimated():
    view = TruncatePolicy(budget=1000).prepare(history, system_prompt=SYSTEM, context_window=1000, usage=None)
    assert view.estimated is True
    assert view.used_tokens == TruncatePolicy.estimate_tokens(view.messages)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_context.py -q`

Expected: FAIL because `TruncatePolicy` does not exist.

- [ ] **Step 3: Implement token estimate and turn grouping**

Use provider `usage.input_tokens` when available for status accounting; use a
deterministic serialized-character estimate divided by four otherwise. Group
history by `turn_id`, remove oldest complete groups, retain the current turn,
prepend the configured system prompt, and insert a system compact marker. Never
mutate the source `SessionMessage` list or SessionStore.

- [ ] **Step 4: Implement forced compaction metadata**

`force=True` compacts only if there is a removable complete turn. Return
`overflow=True` when the current turn alone exceeds budget. The runner will make
at most one request with that view; a provider context-overflow error ends the
run instead of retrying the same view.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_context.py -q`

Expected: PASS.

```bash
git add src/coding_agent/context tests/test_context.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 10: Implement the Pi-Style AgentRunner

**Files:**
- Create: `src/coding_agent/runtime/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `LLMProvider`, `ToolRegistry`, `ToolExecutor`, `ContextPolicy`, `SessionStore`, `RuntimeEvent`, `EventSink`, system prompt, and an event emitter from Tasks 2, 4, 8, and 9.
- Produces:

```python
class AgentRunner:
    async def run_turn(
        self,
        prompt: str,
        *,
        run_id: str,
        turn_id: str,
        signal: asyncio.Event,
    ) -> TurnOutcome: ...
```

- [ ] **Step 1: Write deterministic loop tests**

```python
@pytest.mark.asyncio
async def test_tool_call_then_final_answer():
    provider = FakeProvider([
        assistant_with_tool_call("read_file", {"path": "main.py"}),
        assistant_text("The file is ready."),
    ])
    runner, store, events = make_runner(provider)
    outcome = await runner.run_turn("inspect main.py", run_id="r1", turn_id="t1", signal=asyncio.Event())
    assert outcome.reason == "completed"
    assert "The file is ready." in outcome.final_text
    assert [event.type for event in events].count("tool_started") == 1


@pytest.mark.asyncio
async def test_max_steps_stops_repeating_tool_calls():
    provider = RepeatingToolProvider("read_file")
    runner = make_runner(provider, max_steps=2)[0]
    outcome = await runner.run_turn("loop", run_id="r1", turn_id="t1", signal=asyncio.Event())
    assert outcome.reason == "max_steps"


@pytest.mark.asyncio
async def test_truncated_tool_call_is_not_executed():
    provider = FakeProvider([assistant_with_truncated_tool_call("write_file")])
    tool = RecordingTool("write_file")
    runner = make_runner(provider, tools=[tool])[0]
    outcome = await runner.run_turn("write", run_id="r1", turn_id="t1", signal=asyncio.Event())
    assert tool.calls == []
    assert outcome.reason == "completed"


@pytest.mark.asyncio
async def test_invalid_json_tool_call_is_returned_to_model_as_error():
    provider = FakeProvider([
        malformed_tool_call("read_file", '{"path":'),
        assistant_text("I could not parse the tool arguments."),
    ])
    runner = make_runner(provider)[0]
    outcome = await runner.run_turn("read", run_id="r1", turn_id="t1", signal=asyncio.Event())
    assert outcome.reason == "completed"
    assert any(message.role == "tool" and "invalid" in (message.content or "") for message in runner.store.project_messages())
```

Add two failure-path tests before implementation: a provider `error` event
produces `provider_error` without another retry, and setting the cancellation
event during a blocked provider produces `aborted` after the stream/task is
cleaned up. These complement the incomplete-call and `max_steps` cases above.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_runner.py -q`

Expected: FAIL because `AgentRunner` does not exist.

- [ ] **Step 3: Implement the step loop and stream aggregation**

For every step, load projected history, call `ContextPolicy.prepare` with the
configured system prompt, collect schemas, consume provider events, append text
to one assistant buffer, and aggregate tool argument fragments by call id.
Parse and validate arguments only after `response_end`. Emit the Task 2
`RuntimeEvent` values through the injected `EventSink`; do not define a second
event type in the runner.

- [ ] **Step 4: Persist and execute calls in order**

Append `assistant_message` before tool execution. Append a separate audit
`tool_call` record but never project it as a second model message. Execute calls
sequentially through `ToolExecutor`; append each `tool_result` before starting
the next model request. A failed append prevents the corresponding side effect
or ends the run with `session_error`.

- [ ] **Step 5: Implement termination and cancellation**

No tool calls means `completed`. `max_steps` ends the turn. Provider errors,
session errors, and `signal.is_set()` produce structured outcomes. A partial or
`finish_reason="length"` tool call becomes an error result and is not executed.
Invalid JSON arguments become an error tool result and allow the model to
re-issue the call; they do not crash the runner.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_runner.py -q`

Expected: PASS.

```bash
git add src/coding_agent/runtime/runner.py src/coding_agent/runtime/models.py tests/test_runner.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 11: Implement AgentRuntime and Runtime Events

**Files:**
- Create: `src/coding_agent/runtime/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `AgentRunner`, `SessionStore`, `ApprovalBroker`, `ApprovalPolicy`, and provider configuration.
- Produces:

```python
class AgentRuntime:
    def __init__(
        self,
        *,
        store: SessionStore,
        runner_factory: Callable[[SessionStore, ContextPolicy, ApprovalBroker], AgentRunner],
        context_policy_factory: Callable[[], ContextPolicy],
        approval_policy: ApprovalPolicy,
        system_prompt: Message,
        model: str,
        permission_mode: PermissionMode = "default",
    ) -> None: ...
    async def submit(self, prompt: str) -> str: ...
    async def new_session(self) -> str: ...
    async def list_sessions(self) -> list[SessionSummary]: ...
    async def abort(self, run_id: str) -> None: ...
    async def resolve_approval(self, request_id: str, decision: Literal["approve", "deny"]) -> None: ...
    async def resume(self, session_id: str) -> None: ...
    async def compact(self) -> None: ...
    async def set_permission(self, mode: PermissionMode) -> None: ...
    def subscribe(self, sink: EventSink) -> Callable[[], None]: ...
```

The runtime exposes read-only `session_id`, `permission_mode`, `status`, and
`last_outcome` properties for the TUI and deterministic tests. `status` is a
Pydantic model containing the current state, optional run/turn ids, and the
latest usage/context values.

`RuntimeEvent` has `event_id`, `timestamp`, `type`, optional `run_id`/`turn_id`,
and a Pydantic payload dictionary. Use the approved event names and payload
minimums from the spec. Add assertions that `run_started` includes
`session_id`, `model`, and `policy`; `assistant_delta` includes a non-empty
`message_id` and `text`; `tool_started` includes `tool_call_id`, `tool_name`,
and `arguments`; `approval_requested` includes the complete `ApprovalRequest`;
and `run_finished` includes `outcome` and `steps`.

- [ ] **Step 1: Write runtime lifecycle tests**

```python
@pytest.mark.asyncio
async def test_submit_is_non_blocking_and_rejects_busy_run():
    runtime = make_runtime(provider=BlockingFakeProvider())
    first = await runtime.submit("first")
    assert first
    with pytest.raises(RuntimeError, match="active run"):
        await runtime.submit("second")


@pytest.mark.asyncio
async def test_abort_cancels_pending_approval():
    runtime = make_runtime(provider=ApprovalBlockingProvider())
    run_id = await runtime.submit("write")
    await wait_for_status(runtime, "waiting_approval")
    await runtime.abort(run_id)
    assert runtime.status.status == "idle"


@pytest.mark.asyncio
async def test_resume_resets_permission_to_default(tmp_path):
    runtime = make_runtime(session_root=tmp_path)
    await runtime.set_permission("full")
    session_id = runtime.session_id
    await runtime.new_session()
    await runtime.resume(session_id)
    assert runtime.permission_mode == "default"


@pytest.mark.asyncio
async def test_unknown_or_late_approval_id_is_rejected():
    runtime = make_runtime(provider=ApprovalBlockingProvider())
    run_id = await runtime.submit("write")
    await wait_for_status(runtime, "waiting_approval")
    await runtime.abort(run_id)
    with pytest.raises(RuntimeError, match="not pending"):
        await runtime.resolve_approval("stale", "approve")
```

Add `test_compact_records_metadata_and_preserves_records`: seed at least two
complete turns, call `await runtime.compact()` while idle, assert one
`compaction` record whose payload contains `strategy`, `removed_turn_ids`,
`retained_turn_ids`, `tokens_before`, `tokens_after`, and `forced=True`, and
assert a `context_updated` event. A compact request with no removable complete
turn is idle-safe, emits a notice, and does not append an empty compaction
record. Add `test_compact_rejects_active_run` to prove it cannot race a run.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_runtime.py -q`

Expected: FAIL because the runtime implementation does not exist.

- [ ] **Step 3: Implement runtime ownership and non-blocking submit**

The constructor creates a session when none is supplied. `submit()` checks for
an active task, creates `run_id`/`turn_id`, appends `turn_start` and
`user_message`, publishes `run_started`, schedules `AgentRunner.run_turn` with
`asyncio.create_task`, and returns immediately. The completion callback appends
`turn_end`, publishes `run_finished` or `run_error`, refreshes git branch and
usage, and clears the active task.

The constructor signature above is the dependency-injection boundary. The
runtime creates the single `ApprovalBroker`; `runner_factory` receives that
broker plus the current store and context policy so `new_session` and `resume`
can rebuild the runner without leaking Textual or provider wire types into the
runtime. `context_policy_factory` creates a fresh policy for each new/resumed
session.

- [ ] **Step 4: Implement event subscriptions and approval resolution**

Publish events to a snapshot of subscribers so one faulty sink cannot prevent
other sinks. The approval broker has exactly one pending request. Unknown,
resolved, cancelled, or late request ids return/ignore a structured error and
never restart the run. `abort()` sets the event, cancels approvals, and awaits
provider/tool cleanup before publishing the final outcome.

- [ ] **Step 5: Implement session commands**

`new_session()` works only while idle, resets the context policy and visible
session state, and uses default permission. `list_sessions()` delegates to the
store. `resume()` loads the selected session, marks an open turn interrupted,
sets permission to default, rebuilds the runner, publishes `session_loaded`, and
republishes `store.load_notice` when present.

`compact()` works only while idle. It calls `ContextPolicy.prepare(force=True)`
on the projected history, publishes `context_updated`, and when a complete turn
was removed appends one `compaction` record with the exact metadata fields
`strategy`, `removed_turn_ids`, `retained_turn_ids`, `tokens_before`,
`tokens_after`, and `forced`. It never rewrites or deletes prior records.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_runtime.py -q`

Expected: PASS.

```bash
git add src/coding_agent/runtime tests/test_runtime.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 12: Implement TUI State and Pure Reducer

**Files:**
- Create: `src/coding_agent/tui/state.py`
- Create: `src/coding_agent/tui/reducer.py`
- Test: `tests/test_reducer.py`

**Interfaces:**
- Consumes: `RuntimeEvent`.
- Produces a Pydantic `TuiState` snapshot with session/model/context/workspace,
  policy/status, transcript rows, active tool, pending approval, and input text.

- [ ] **Step 1: Write reducer tests**

```python
def test_assistant_deltas_merge_by_message_id():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("assistant_started", {"message_id": "m1"}))
    state = reduce(state, event("assistant_delta", {"message_id": "m1", "text": "hel"}))
    state = reduce(state, event("assistant_delta", {"message_id": "m1", "text": "lo"}))
    assert [row.text for row in state.transcript if row.kind == "assistant"] == ["hello"]


def test_tool_started_and_finished_update_one_row():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("tool_started", {"tool_call_id": "c1", "tool_name": "read_file", "arguments": {}}))
    state = reduce(state, event("tool_finished", {"tool_call_id": "c1", "tool_name": "read_file", "ok": True, "content": "x"}))
    row = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert row.tool_status == "success"


def test_approval_and_run_outcomes_update_status():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(state, event("run_started", {"session_id": "s1", "model": "fake", "policy": "default"}))
    state = reduce(state, event("approval_requested", {"request": approval_request()}))
    assert state.status == "waiting_approval"
    state = reduce(state, event("approval_resolved", {"request_id": "a1", "decision": "approve", "status": "approved"}))
    assert state.status == "running"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_reducer.py -q`

Expected: FAIL because TUI state and reducer do not exist.

- [ ] **Step 3: Implement immutable Pydantic state**

Define `TranscriptItem`, `TuiState`, and `initial_state`. Use
`model_copy(update=...)` and copied lists for every reducer transition. Match
assistant rows by non-empty message id; update tools by `tool_call_id`; never
append a row for every delta.

- [ ] **Step 4: Implement all approved event mappings**

Handle run start/finish/error, user message, assistant start/delta/finish,
tool start/finish, approval requested/resolved, context update, session loaded,
policy changed, and notice. `run_finished` with `outcome="aborted"` produces an
aborted/idle terminal display rather than a missing event branch.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_reducer.py -q`

Expected: PASS.

```bash
git add src/coding_agent/tui/state.py src/coding_agent/tui/reducer.py tests/test_reducer.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 13: Build the Textual TUI Shell

**Files:**
- Create: `src/coding_agent/tui/app.py`
- Create: `src/coding_agent/tui/widgets.py`
- Test: `tests/test_tui.py`

**Interfaces:**
- Consumes: `AgentRuntime`, `RuntimeEvent`, `TuiState`, and reducer.
- Produces a Textual `CodingAgentApp` with `VerticalScroll` transcript,
  `TextArea` composer, statusline, approval modal, session selector, and command palette.

- [ ] **Step 1: Write Textual Pilot tests**

```python
@pytest.mark.asyncio
async def test_app_has_transcript_input_and_statusline():
    app = make_app()
    async with app.run_test() as pilot:
        assert app.query_one("#transcript")
        assert app.query_one("#composer")
        assert app.query_one("#statusline")


@pytest.mark.asyncio
async def test_runtime_queue_coalesces_deltas_but_preserves_lifecycle():
    app = make_app()
    async with app.run_test() as pilot:
        app.runtime.emit_many([
            event("assistant_delta", {"message_id": "m1", "text": "a"}),
            event("assistant_delta", {"message_id": "m1", "text": "b"}),
            event("run_finished", {"outcome": "completed", "steps": 1}),
        ])
        await pilot.pause()
        assert app.state.status == "idle"
        assert "ab" in next(row.text for row in app.state.transcript if row.kind == "assistant")


@pytest.mark.asyncio
async def test_enter_submits_prompt_when_idle():
    app = make_app()
    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.text = "inspect the project"
        await pilot.press("enter")
        await pilot.pause()
        assert app.runtime.submit_calls == ["inspect the project"]
```

Also cover the required safety controls with Pilot tests: Ctrl-C during a run
calls `runtime.abort()` and returns to idle, and an approval modal renders the
request then calls only `runtime.resolve_approval()` for Approve/Deny.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_tui.py -q`

Expected: FAIL because the Textual app does not exist.

- [ ] **Step 3: Implement the fixed layout**

Compose exactly:

```text
VerticalScroll#transcript
TextArea#composer
Static#statusline
```

Keep the statusline as the bottom fixed row. Do not add a footer row. Use
Textual workers/tasks to await runtime operations so the UI event loop remains
responsive. The runtime publishes into an asyncio queue owned by the TUI
bridge; the bridge drains that queue on the Textual event loop and applies the
reducer there. Runtime tasks never call widgets directly. Queue puts are
bounded; if the UI is slow, assistant deltas may be coalesced by message id,
while lifecycle/tool/approval events are preserved. Unsubscribe closes the
queue consumer and removes the sink; a faulty sink is isolated from other
subscribers.

- [ ] **Step 4: Connect runtime events through the reducer**

Subscribe at mount, reduce every event into a new `TuiState`, render transcript
rows, update the statusline, and auto-scroll only when the user is already at
the bottom. Runtime exceptions become system/error rows rather than uncaught
widget exceptions.

- [ ] **Step 5: Add approval modal**

Render tool name, arguments, category, reason, and Approve/Deny actions. The
modal calls only `runtime.resolve_approval()` and never executes a tool or reads
stdin directly. Ctrl-C denies/cancels the pending request.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_tui.py -q`

Expected: PASS.

```bash
git add src/coding_agent/tui/app.py src/coding_agent/tui/widgets.py tests/test_tui.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 14: Add Commands, Session Selector, and Statusline

**Files:**
- Create: `src/coding_agent/tui/commands.py`
- Modify: `src/coding_agent/tui/app.py`
- Modify: `src/coding_agent/tui/widgets.py`
- Test: `tests/test_tui.py`

**Interfaces:**
- Consumes: `AgentRuntime.list_sessions`, `new_session`, `resume`, `compact`, `set_permission`, and `ContextView` state.
- Produces local command parsing and the required video-facing status information.

- [ ] **Step 1: Write command and selector tests**

```python
def test_commands_are_local_and_do_not_become_prompts():
    command = parse_command("/permission workspace")
    assert command.name == "permission"
    assert command.args == ["workspace"]


@pytest.mark.asyncio
async def test_session_selector_resumes_selected_session():
    app = make_app(session_summaries=[summary("s1"), summary("s2")])
    async with app.run_test() as pilot:
        await pilot.press("/", "s", "e", "s", "s", "i", "o", "n")
        await pilot.press("enter")
        await pilot.press("down", "enter")
        assert app.runtime.resume_calls == ["s2"]


def test_unknown_command_and_ambiguous_resume_stay_local():
    assert parse_command("/unknown").name == "unknown"
    assert parse_command("/resume ab").args == ["ab"]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_tui.py -q`

Expected: FAIL because command parsing and selector behavior are incomplete.

- [ ] **Step 3: Implement local command registry**

Support exactly:

```text
/help
/new
/session
/resume <id-or-unique-prefix>
/compact
/context
/permission
/permission default
/permission workspace
/permission full
/clear
/quit
```

Commands produce TUI notices or invoke runtime methods; none are appended to
model history. Unknown commands produce an error notice. `/clear` clears only
visible transcript; `/new` creates a new persisted session.

- [ ] **Step 4: Implement session selector**

Use a Textual `ModalScreen` and `OptionList`, show short id, updated time,
workspace, and bounded title, sort newest first, and call `resume()` on Enter.
Reject ambiguous prefixes with a system notice.

- [ ] **Step 5: Implement statusline and keyboard controls**

Render model, reasoning or `-`, context used/remaining, context window when
known, input/output usage when known, workspace, git branch or `-`, session short
id, permission policy, and runtime status. Hide low-priority fields instead of
wrapping on narrow terminals. Enter submits only while idle; Ctrl-C aborts a
run, cancels approval, or exits/clears idle input according to current state.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_tui.py -q`

Expected: PASS.

```bash
git add src/coding_agent/tui tests/test_tui.py
git commit  # use the Lore-format body described in Global Constraints
```

## Task 15: Wire the CLI Entrypoint and End-to-End MVP Flow

**Files:**
- Create: `src/coding_agent/app.py`
- Create: `tests/test_integration_flow.py`
- Create: `README.txt`
- Create or modify: `README.md` (developer documentation; it is not one of the
  required submission files, but the repository should document development)

**Interfaces:**
- Consumes: all completed runtime, tool, session, policy, and TUI modules.
- Produces the `coding-agent` executable, a deterministic integration path, and
  assignment-compliant run documentation.

- [ ] **Step 1: Write the integration test**

```python
@pytest.mark.asyncio
async def test_fake_agent_creates_and_verifies_small_project(tmp_path):
    provider = FakeProvider([
        assistant_with_tool("write_file", {"path": "index.html", "content": "<h1>Demo</h1>"}),
        assistant_with_tool("run_command", {"command": "python -c 'print(\"ok\")'"}),
        assistant_text("Created and verified the page."),
    ])
    runtime = make_runtime(provider=provider, workspace=tmp_path, permission_mode="full")
    run_id = await runtime.submit("Create and verify a page")
    await wait_for_idle(runtime)
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<h1>Demo</h1>"
    assert runtime.last_outcome.reason == "completed"
```

- [ ] **Step 2: Run the integration test and verify failure**

Run: `pytest tests/test_integration_flow.py -q`

Expected: FAIL until the application wiring exists.

- [ ] **Step 3: Implement application construction**

`app.py` resolves the workspace, model, base URL, API key environment variable,
context window, and session directory; constructs the provider, six built-in
tools, registry, policy, executor, context policy, session store, runtime, and
Textual app. It must fail with a redacted configuration error when the model or
credential is missing.

- [ ] **Step 4: Implement the console entrypoint**

`main()` parses `--workspace`, `--model`, `--base-url`, `--session-dir`, and
`--context-window`, then starts Textual. It must not print secrets. The normal
path is interactive TUI; tests inject a Fake Provider and never start a real
network client. Add `if __name__ == "__main__": main()` so
`python -m coding_agent.app --help` works without credentials and missing-key
errors are only emitted after argument parsing when the app is actually run.

- [ ] **Step 5: Write assignment-compliant README.txt**

Keep `README.txt` below 1,000 Chinese characters and include:

```text
Git repository URL
installation and run commands
environment-variable credential setup
MVP feature summary
explicit note that core agent logic is self-implemented
```

Do not include a real key, personal name, undergraduate school, or other
identifying information. Update `README.md` with developer-oriented details,
not credentials. Populate the repository URL from `git remote get-url origin`
and render the actual public URL (for example, convert the `git@host:path`
form to `https://host/path`); if no origin exists, stop rather than writing a
placeholder.

- [ ] **Step 6: Run the full MVP verification**

Run:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
python -m coding_agent.app --help
```

Expected: all tests pass, lint/format pass, and the CLI prints redacted help.
Also run `markdownlint docs/superpowers/plans/2026-08-30-coding-agent-mvp.md docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md`.
The live provider test remains opt-in and is run separately only when network
credentials are intentionally available; its failure must not make the
network-free suite fail.
Verify assignment artifacts: `README.txt` is under 1,000 Chinese characters,
contains the non-placeholder URL returned by `git remote get-url origin` and
run instructions without credentials or
identity, and the demo MP4 exists, is at most two minutes, and is no larger
than 200 MB.

- [ ] **Step 7: Commit the wired MVP**

```bash
git add src/coding_agent/app.py tests/test_integration_flow.py README.txt README.md
git commit  # use the Lore-format body described in Global Constraints
```

## Task 16: Manual TUI Smoke Test and Video Readiness

**Files:**
- Modify only files required by failed smoke checks; do not broaden MVP scope.
- Test: manual local run in a temporary demo workspace.

**Interfaces:**
- Consumes the complete MVP executable from Task 15.
- Produces evidence for the assignment video and a short verification note in
  the final handoff; no new runtime feature is introduced in this task.

- [ ] **Step 1: Create a disposable demo workspace**

Use a temporary directory containing a small starter project, for example an
HTML/CSS/JavaScript page with one deliberate test or behavior defect. Do not
use personal paths or identifying content in the recording.

- [ ] **Step 2: Run the TUI with an approved local model configuration**

Run: `coding-agent --workspace /path/to/demo --model <model>`

Verify the bottom statusline shows model, context, workspace, branch/session,
policy, and status. Verify the transcript scrolls, the TextArea accepts a
multi-line prompt, and Enter starts a run without freezing the UI.

- [ ] **Step 3: Exercise the visible safety and session flow**

In the TUI:

```text
/permission default
submit a task that needs a write
approve the write in the modal
/permission workspace
/session
resume the session
/context
Ctrl-C during a long command
```

Verify no command enters model history, approval denial returns an error tool
result, Ctrl-C leaves the runtime idle, and resume starts with default policy.

- [ ] **Step 4: Record the real task demonstration**

Demonstrate a small web page or game: inspect files, write/edit files, run a
verification command, react to an error or follow-up request, and finish with
the TUI statusline and resulting files visible. Keep the MP4 at or below two
minutes and 200 MB, and keep the one-minute English introduction separate from
the video length requirement.

- [ ] **Step 5: Commit only verified fixes**

Run `pytest -q` and `ruff check src tests` after every smoke-test fix, then
commit each focused fix with the repository's required decision-record format.

## Post-MVP Implementation Backlog

These are deliberately separate from the MVP execution sequence:

1. **MCP stdio extension:** `.mcp.json`, `initialize`, `tools/list`,
   `tools/call`, external schemas adapted into `ToolRegistry`, connection and
   call failures converted to `ToolResult`.
2. **Subagent S1:** a foreground `task` tool creates an isolated child
   `SessionStore`, filters recursive/task tools, reuses `AgentRunner`, and
   returns a final summary. Do not add background task UI in S1.
3. **Skill loader:** discover `.agents/skills/*/SKILL.md` and expose a bounded
   index before loading full skill content.
4. **Read-only parallel tools:** execute only tools with
   `is_parallel_safe=True`, preserve model order, and keep writes/shell serial.
5. **Alternative context policies:** add Pi-style LLM summary and
   oh-my-cli-style receipt summary behind the same `ContextPolicy` contract.
6. **Session tree:** implement `rewind`, `fork`, active leaf, and branch
   summaries using the already-persisted `id`, `seq`, `parent_id`, and
   `turn_id` fields.
7. **Stronger execution isolation:** add a container/bubblewrap backend only as
   a separately documented security project; do not describe the MVP soft
   policy as an OS sandbox.

## Plan Self-Review

### Spec coverage

- Spec goal/scope: Tasks 1 and 15.
- Pydantic data models: Task 2.
- Provider and streaming delta protocol: Task 3.
- Agent turn/step loop and termination: Task 10.
- Tool registry and six tools: Tasks 5 and 6.
- Permission modes, outside-path approval, and catastrophic command denial:
  Tasks 7 and 8.
- JSONL session, resume, audit call records, and future tree fields: Task 4
  and Task 11.
- Deterministic ContextPolicy and compaction record: Task 9 and Task 11.
- Runtime event envelope and TUI reducer: Tasks 11 and 12.
- Textual layout, approval modal, session selector, commands, and statusline:
  Tasks 13 and 14.
- Fake Provider, temporary workspace, reducer, Pilot, and integration tests:
  Tasks 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, and 15.
- Assignment README/video/double-blind constraints: Tasks 15 and 16.
- MCP and Subagent staged extensions: Post-MVP backlog.

### Consistency checks

- `AgentRuntime.submit()` is explicitly non-blocking and has one active run.
- `run_finished` carries `outcome="aborted"`; no `run_aborted` event is used.
- `tool_call` is persisted for audit but excluded from Message projection.
- System prompt is configured by the runtime and prepended after projection; it
  has no `turn_id` and is not grouped by ContextPolicy.
- Outside paths in `default`/`workspace` require a per-call approval override;
  `full` bypasses ordinary containment; catastrophic commands remain denied.
- Session resume resets permission to `default` and never replays side effects.
- TUI depends on Runtime events and commands only; Runtime has no Textual import.

### Placeholder scan

The plan contains no placeholder markers or unspecified “add appropriate
handling” steps. Every task names files, interfaces, failing tests, commands,
implementation behavior, and a commit boundary.

### Fresh review status

PASS after the independent review: compact/runtime construction and corruption
notice contracts are explicit; shell cancellation and required failure-path
tests are named; the top-level test map is complete; and README URL validation
is executable. Re-run `markdownlint` and `git diff --check` after any plan edit.
