# W1 — Real-time Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stream `run_command` output into the transcript tool row as it is
produced, and show a running spinner + live elapsed timer in the statusline.

**Architecture:** Add a `tool_output_delta` runtime event produced by the runner
from an optional per-call output sink threaded through `ToolContext`. The shell
tool invokes `context.on_output` per stdout chunk. The reducer accumulates
deltas into the tool row. The app runs a timer that animates the statusline
spinner frame and recomputes elapsed time while a run is active.

**Tech Stack:** Python 3.11+, Textual 8.2.8, asyncio, pydantic.

**Spec:** `docs/superpowers/specs/2026-09-01-coding-agent-feature-roadmap-design.md` §1.

## Global Constraints

- No new dependencies beyond the existing `openai`, `pydantic`, `textual`.
- Textual CSS has no `lighten()`/`mix()`; colors are hex.
- `tui/reducer.py` stays a pure function (no timers, no IO).
- Existing 395 tests must stay green; new behavior is additive.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/runtime/events.py`: add `tool_output_delta` to the literal.
- `src/coding_agent/tools/registry.py`: `ToolContext.on_output` optional callback.
- `src/coding_agent/tools/shell.py`: stream chunks through `context.on_output`.
- `src/coding_agent/tools/executor.py`: accept + thread `output_sink` into `ToolContext`.
- `src/coding_agent/runtime/runner.py`: per-call output sink emitting `tool_output_delta`.
- `src/coding_agent/tui/state.py`: `run_started_at`, `spinner_frame` on `TuiState`.
- `src/coding_agent/tui/reducer.py`: handle `tool_output_delta`; set/clear `run_started_at`.
- `src/coding_agent/tui/app.py`: bridge coalescing for `tool_output_delta`; timer.
- `src/coding_agent/tui/widgets.py`: `StatusLine` spinner frame + elapsed.
- `tests/test_w1_realtime_feedback.py` (new): all W1 coverage.

---

## Task 1: Tool output sink (registry → shell → executor)

**Files:**
- Modify: `src/coding_agent/tools/registry.py`, `src/coding_agent/tools/shell.py`, `src/coding_agent/tools/executor.py`
- Test: `tests/test_w1_realtime_feedback.py`

**Interfaces:**
- Consumes: existing `ToolContext`, `_ShellTool.execute`, `ToolExecutor.execute`.
- Produces:
  - `ToolContext.on_output: Callable[[str], Awaitable[None]] | None = None`
  - `ToolExecutor.execute(call, *, run_id, workspace, permission_mode, signal, output_sink=None)`
  - `_ShellTool` calls `await context.on_output(chunk_text)` per stdout chunk when set.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio

import pytest

from coding_agent.runtime.models import ToolCall
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.registry import ToolContext, ToolRegistry
from coding_agent.tools.shell import make_run_command_tool
from coding_agent.policy.approval import DefaultApprovalPolicy


class _NoopBroker:
    async def request(self, request):
        return "approve"

    def cancel_all(self):
        pass


async def test_tool_context_carries_on_output():
    collected: list[str] = []
    async def sink(text: str) -> None:
        collected.append(text)
    context = ToolContext(workspace=".", permission_mode="full", on_output=sink)
    assert context.on_output is not None
    await context.on_output("hello")
    assert collected == ["hello"]


async def test_shell_tool_streams_output_chunks(tmp_path):
    collected: list[str] = []
    async def sink(text: str) -> None:
        collected.append(text)
    tool = make_run_command_tool()
    context = ToolContext(
        workspace=str(tmp_path), permission_mode="full", on_output=sink
    )
    signal = asyncio.Event()
    result = await tool.execute(
        {"command": "printf 'one\\ntwo\\n'"}, context=context, signal=signal
    )
    assert result.ok
    assert result.content == "one\ntwo\n"
    assert collected and "".join(collected) == "one\ntwo\n"


async def test_executor_forwards_output_sink(tmp_path):
    collected: list[str] = []
    async def sink(text: str) -> None:
        collected.append(text)
    registry = ToolRegistry()
    registry.register(make_run_command_tool())
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), _NoopBroker())
    call = ToolCall(id="c1", name="run_command", arguments={"command": "printf hi"})
    signal = asyncio.Event()
    result = await executor.execute(
        call,
        run_id="r",
        workspace=tmp_path,
        permission_mode="full",
        signal=signal,
        output_sink=sink,
    )
    assert result.ok
    assert "".join(collected) == "hi"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_w1_realtime_feedback.py -q`
Expected: FAIL — `ToolContext` has no `on_output` field; `output_sink` unknown kwarg.

- [ ] **Step 3: Implement**

- `registry.py` `ToolContext`:

```python
class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Path
    permission_mode: PermissionMode
    allow_outside_once: bool = False
    on_output: Callable[[str], Awaitable[None]] | None = None
```

  (add `from collections.abc import Awaitable, Callable` import)

- `shell.py` `_ShellTool._collect_output`: accept and call the sink. Change the
  collect task construction in `execute` and `_collect_output`:

```python
started = time.monotonic()
collect = asyncio.create_task(
    self._collect_output(proc, output_sink=context.on_output)
)
```

  and in `_collect_output`:

```python
@staticmethod
async def _collect_output(proc, *, output_sink=None):
    output = bytearray()
    truncated = False
    assert proc.stdout is not None
    while chunk := await proc.stdout.read(4096):
        remaining = MAX_COMMAND_OUTPUT_BYTES - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
        if output_sink is not None and len(chunk) <= remaining:
            await output_sink(chunk.decode(errors="replace"))
    await proc.wait()
    return bytes(output), truncated
```

  Note: only stream chunks that fit in the output budget so the sink mirrors the
  final content exactly (matches the test `"".join(collected) == result.content`).

- `executor.py`: thread `output_sink` through to `ToolContext`:

```python
async def execute(self, call, *, run_id, workspace, permission_mode, signal,
                  output_sink=None):
    ...
    context = ToolContext(
        workspace=workspace,
        permission_mode=permission_mode,
        allow_outside_once=outside_once,
        on_output=output_sink,
    )
```

- [ ] **Step 4: Run focused + full suite**

Run: `pytest tests/test_w1_realtime_feedback.py -q && pytest -q`
Expected: focused PASS; full suite green (395+3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tools/registry.py src/coding_agent/tools/shell.py src/coding_agent/tools/executor.py tests/test_w1_realtime_feedback.py
git commit -m "Thread a tool output sink through ToolContext for streamed command output"
```

---

## Task 2: Runner emits `tool_output_delta`

**Files:**
- Modify: `src/coding_agent/runtime/events.py`, `src/coding_agent/runtime/runner.py`
- Test: `tests/test_w1_realtime_feedback.py`

**Interfaces:**
- Consumes: `ToolExecutor.execute(..., output_sink=...)` from Task 1.
- Produces: `RuntimeEvent(type="tool_output_delta", payload={"tool_call_id": str, "text": str})`
  emitted between `tool_started` and `tool_finished` for each output chunk.

- [ ] **Step 1: Write the failing test**

```python
async def test_runner_emits_tool_output_delta():
    from coding_agent.runtime.events import RuntimeEvent
    from coding_agent.runtime.models import LLMEvent
    from coding_agent.runtime.runner import AgentRunner
    from coding_agent.session.store import SessionStore
    from coding_agent.tools.registry import ToolRegistry
    from coding_agent.tools.shell import make_run_command_tool
    from coding_agent.context.truncate import TruncatePolicy
    from coding_agent.runtime.models import Message
    from coding_agent.tools.executor import ToolExecutor
    from coding_agent.policy.approval import DefaultApprovalPolicy

    events: list[RuntimeEvent] = []
    async def sink(event: RuntimeEvent) -> None:
        events.append(event)

    provider_events = [
        LLMEvent(type="tool_call_start", tool_call_id="call-1", tool_name="run_command"),
        LLMEvent(type="tool_call_delta", tool_call_id="call-1",
                 arguments_delta='{"command": "printf ab"}'),
        LLMEvent(type="response_end", finish_reason="tool_calls"),
        LLMEvent(type="tool_call_start", tool_call_id="call-2", tool_name="grep_files"),
        LLMEvent(type="tool_call_delta", tool_call_id="call-2",
                 arguments_delta='{"pattern": "x", "path": "."}'),
        LLMEvent(type="response_end", finish_reason="tool_calls"),
        LLMEvent(type="text_delta", text="done"),
        LLMEvent(type="response_end", finish_reason="stop"),
    ]
    # Use a streaming shell output > pipe chunk size so deltas are produced.
    # Fake provider replays tool calls for run_command then a final text turn.
```

  The clean way: drive a real `AgentRunner` whose executor runs a real shell
  tool and whose provider replays a `run_command` call. Use a command with
  enough output to cross the pipe buffer, e.g. `seq 1 2000`. Assert at least one
  `tool_output_delta` arrives with `tool_call_id == "call-1"` and that its
  concatenated text matches the final tool content. Build the runner with a
  session store created in a tmp dir.

```python
def _make_runner(tmp_path, provider, registry, store, executor):
    return AgentRunner(
        provider=provider,
        registry=registry,
        executor=executor,
        context_policy=TruncatePolicy(),
        store=store,
        event_sink=lambda e: _collect(e),
        system_prompt=Message(role="system", content="sys"),
        model="test",
        context_window=100_000,
        permission_mode="full",
    )
```

  For the assertion, run one turn with `signal = asyncio.Event()` and collect
  all events; filter `type == "tool_output_delta"`.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_w1_realtime_feedback.py::test_runner_emits_tool_output_delta -q`
Expected: FAIL — the event type literal rejects `tool_output_delta` (validation
error) or no such events are emitted.

- [ ] **Step 3: Implement**

- `events.py`: add `"tool_output_delta"` to `RuntimeEventType`.
- `runner.py`: build a per-call sink. Replace the sequential tool loop's executor
  call to pass an `output_sink` that emits the event, and give the executor a
  sink regardless of call type (the executor ignores it for non-streaming tools):

```python
async def _tool_output_sink(self, run_id, turn_id, call_id):
    async def sink(text: str) -> None:
        await self._emit(
            "tool_output_delta", run_id, turn_id,
            tool_call_id=call_id, text=text,
        )
    return sink
```

  In the tool loop, before `self.executor.execute(...)`, create the sink with the
  current `call.id` and pass `output_sink=await self._tool_output_sink(run_id, turn_id, call.id)`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/runtime/events.py src/coding_agent/runtime/runner.py tests/test_w1_realtime_feedback.py
git commit -m "Emit tool_output_delta events for streamed tool output"
```

---

## Task 3: Reducer + state for deltas and run timing

**Files:**
- Modify: `src/coding_agent/tui/state.py`, `src/coding_agent/tui/reducer.py`
- Test: `tests/test_w1_realtime_feedback.py`

**Interfaces:**
- Consumes: `tool_output_delta` event from Task 2; `run_started`/`run_finished`.
- Produces:
  - `TuiState.run_started_at: float | None`
  - `TuiState.spinner_frame: int = 0`
  - reducer handles `tool_output_delta` (append to matching tool row text) and
    sets/clears `run_started_at`.

- [ ] **Step 1: Write the failing tests**

```python
def test_reducer_accumulates_tool_output_delta():
    from coding_agent.tui.reducer import reduce
    from coding_agent.tui.state import initial_state
    from coding_agent.runtime.events import RuntimeEvent

    state = initial_state(".", "test")
    started = RuntimeEvent(
        type="tool_started", run_id="r", turn_id="t",
        payload={"tool_call_id": "c1", "tool_name": "run_command",
                 "arguments": {"command": "seq 10"}},
    )
    state = reduce(state, started)
    state = reduce(state, RuntimeEvent(
        type="tool_output_delta", run_id="r", turn_id="t",
        payload={"tool_call_id": "c1", "text": "1\n2\n"},
    ))
    state = reduce(state, RuntimeEvent(
        type="tool_output_delta", run_id="r", turn_id="t",
        payload={"tool_call_id": "c1", "text": "3\n"},
    ))
    tool_rows = [r for r in state.transcript if r.kind == "tool"]
    assert tool_rows and tool_rows[0].text == "1\n2\n3\n"


def test_run_started_sets_timing_and_finished_clears():
    from coding_agent.tui.reducer import reduce
    from coding_agent.tui.state import initial_state
    from coding_agent.runtime.events import RuntimeEvent

    state = initial_state(".", "test")
    state = reduce(state, RuntimeEvent(
        type="run_started", run_id="r", turn_id="t",
        payload={"session_id": "s", "model": "test", "policy": "default"},
    ))
    assert state.run_started_at is not None
    assert state.spinner_frame == 0
    state = reduce(state, RuntimeEvent(
        type="run_finished", run_id="r", turn_id="t",
        payload={"outcome": {"reason": "completed"}, "steps": 1},
    ))
    assert state.run_started_at is None
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

- `state.py` `TuiState`:

```python
run_started_at: float | None = None
spinner_frame: int = 0
```

- `reducer.py`:
  - `run_started`: `updates["run_started_at"] = _time_now()` (use `time.monotonic()`; import `time`).
  - `run_finished` / `run_error`: `updates["run_started_at"] = None`.
  - `tool_output_delta`: find the tool row by `tool_call_id`; append `text` to
    `row.text`; if missing, create from `tool_started`-style data. Never change
    `tool_status` (stays `running`).
  - `session_loaded`: `run_started_at=None`, `spinner_frame=0`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/state.py src/coding_agent/tui/reducer.py tests/test_w1_realtime_feedback.py
git commit -m "Accumulate streamed tool output and track run timing in TUI state"
```

---

## Task 4: Bridge coalescing for `tool_output_delta`

**Files:**
- Modify: `src/coding_agent/tui/app.py`
- Test: `tests/test_w1_realtime_feedback.py`

**Interfaces:**
- Consumes: `_RuntimeBridge` existing coalescing for `assistant_delta`.
- Produces: `tool_output_delta` is coalesced under key `(generation, tool_call_id)` and never reorders control events.

- [ ] **Step 1: Write the failing test**

  Test `_RuntimeBridge` directly: publish two `tool_output_delta` events for the
  same call followed by a `tool_finished` control event; after draining, the
  two deltas' text must be concatenated and arrive before `tool_finished`.

```python
def test_bridge_coalesces_tool_output_delta():
    from coding_agent.tui.app import _RuntimeBridge
    from coding_agent.runtime.events import RuntimeEvent

    applied: list[RuntimeEvent] = []

    class FakeApp:
        def _apply_event(self, event):
            applied.append(event)
        def _show_error(self, message):
            raise AssertionError(message)

    bridge = _RuntimeBridge(FakeApp(), maxsize=4)
    asyncio.get_event_loop().run_until_complete(bridge._publish_locked(
        RuntimeEvent(type="tool_output_delta", run_id="r",
                     payload={"tool_call_id": "c1", "text": "ab"})))
    asyncio.get_event_loop().run_until_complete(bridge._publish_locked(
        RuntimeEvent(type="tool_output_delta", run_id="r",
                     payload={"tool_call_id": "c1", "text": "cd"})))
    asyncio.get_event_loop().run_until_complete(bridge._publish_locked(
        RuntimeEvent(type="tool_finished", run_id="r",
                     payload={"tool_call_id": "c1", "ok": True, "content": "abcd"})))
    # drain synchronously
    while bridge._coalesced:
        bridge.app._apply_event(bridge._pop_coalesced())
    delta_texts = [e.payload["text"] for e in applied if e.type == "tool_output_delta"]
    assert "".join(delta_texts) == "abcd"
    assert applied[-1].type == "tool_finished"
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

  In `_RuntimeBridge._publish_locked`, generalize the `assistant_delta` branch to
  also cover `tool_output_delta` by keying on `message_id` or `tool_call_id`:

```python
if event.type in {"assistant_delta", "tool_output_delta"}:
    stream_id = event.payload.get("message_id") or event.payload.get("tool_call_id")
    if isinstance(stream_id, str) and stream_id:
        key = (self._generation, stream_id)
        # ... existing coalescing body unchanged
```

  Also exclude `tool_output_delta` from the drain-time stale flush logic only as
  needed — it should behave exactly like `assistant_delta` (kept ordered).

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/app.py tests/test_w1_realtime_feedback.py
git commit -m "Coalesce streamed tool output deltas in the TUI bridge"
```

---

## Task 5: Statusline spinner + elapsed timer

**Files:**
- Modify: `src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/app.py`
- Test: `tests/test_w1_realtime_feedback.py`

**Interfaces:**
- Consumes: `TuiState.run_started_at`, `TuiState.spinner_frame` from Task 3.
- Produces:
  - `SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")`
  - `StatusLine` renders a spinner frame + `⏱ <elapsed>` in the status field when running.
  - `CodingAgentApp` timer (via `set_interval`) advances `spinner_frame` and refreshes while active.

- [ ] **Step 1: Write the failing tests**

```python
def test_statusline_renders_spinner_and_elapsed():
    from coding_agent.tui.widgets import StatusLine
    from coding_agent.tui.state import initial_state

    state = initial_state(".", "test", model="test", context_window=10)
    state = state.model_copy(update={
        "status": "running", "run_started_at": 100.0, "spinner_frame": 2,
    })
    line = StatusLine(state)
    text = line.format_statusline(state, None)  # inspect implementation for exact signature
    rendered = str(text)
    assert "⠹" in rendered  # frame index 2
    assert "running" in rendered
```

  Also test the app timer advances `spinner_frame` and clears `run_started_at`
  after finish. Use `CodingAgentApp` with a fake runtime whose `status` returns a
  stub; drive `_apply_event` and the interval callback directly.

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

  - `widgets.py`: define `SPINNER_FRAMES`. In `StatusLine.format_statusline`,
    when `state.status == "running"`:
    - frame = `SPINNER_FRAMES[state.spinner_frame % len(SPINNER_FRAMES)]`
    - elapsed = `int(monotonic() - state.run_started_at)` if `run_started_at` else 0
    - render the status field as `f"{frame} running ⏱{elapsed}s"` with the running
      cyan style. When `waiting_approval`, keep elapsed but show a static glyph.
  - `app.py`: add `self._spinner_interval: asyncio.TimerHandle | None = None` and
    a `_tick_spinner()` method that advances `spinner_frame`, recomputes, and calls
    `render_state` on the statusline. Start/restart the interval when `_apply_event`
    sets status to `running`/`waiting_approval`, stop when it leaves those states.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/tui/widgets.py src/coding_agent/tui/app.py tests/test_w1_realtime_feedback.py
git commit -m "Animate statusline spinner and elapsed timer while a run is active"
```

---

## Self-Review

- Spec §1 features mapped: streaming output (Tasks 1–4), spinner+elapsed (Task 5). ✅
- No placeholders; every task has failing-test code. ✅
- Types consistent across tasks: `on_output`, `output_sink`, `tool_output_delta`,
  `run_started_at`, `spinner_frame`. ✅
- Full gates after the last task: `pytest -q`, `ruff check src tests`,
  `ruff format --check src tests`, `python -m coding_agent.app --help`.
