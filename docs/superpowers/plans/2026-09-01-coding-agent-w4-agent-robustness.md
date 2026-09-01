# W4 — Agent Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Auto-retry transient tool failures, detect provider idle timeouts and
no-progress loops, and compress by summarizing dropped turns instead of silently
discarding them.

**Architecture:** `ToolExecutor` gains a retry loop (`max_retries`, backoff) with
a `_retryable` predicate. `AgentRunner` wraps the provider stream in an idle
watchdog and publishes a `heartbeat` event during slow output; it tracks recent
call signatures and returns `reason="progress_loop"` when ≥3 identical calls
repeat. `AgentRuntime.compact()` summarizes removed turns via
`LLMProvider.stream` and stores the summary in the `compaction` record;
`SessionStore.project_messages` prepends the summary as a system message.

**Tech Stack:** Python 3.11+, asyncio, pydantic.

**Spec:** `docs/superpowers/specs/2026-09-01-coding-agent-feature-roadmap-design.md` §4.

## Global Constraints

- Approval denials, cancellations, invalid-arguments, and exact-match errors are never retried.
- `compact` falls back to silent truncation when summarization is unavailable or fails.
- Summarization reuses `LLMProvider.stream` with an empty tool list; no protocol change.
- Existing 395+ tests stay green.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/tools/executor.py`: retry loop + `_retryable`.
- `src/coding_agent/runtime/runner.py`: idle watchdog, heartbeat, loop detection.
- `src/coding_agent/runtime/events.py`: `heartbeat` event type.
- `src/coding_agent/runtime/runtime.py`: compact summarization.
- `src/coding_agent/session/store.py`: prepend summary system message.
- `src/coding_agent/tui/reducer.py` / `widgets.py`: tool footer `· retried 2×`; notices.
- `tests/test_w4_agent_robustness.py` (new).

---

## Task 1: Tool retry with eligibility predicate

**Files:** Modify `src/coding_agent/tools/executor.py`; test `tests/test_w4_agent_robustness.py`.

**Interfaces:**
- Produces:
  - `ToolExecutor(..., max_retries: int = 2, retry_backoff_seconds: float = 1.0)`.
  - `_retryable(error: str) -> bool` (module-level, pure).
  - A failed-but-retryable result is re-run up to `max_retries` times with `retry_backoff_seconds * attempt` delay; final result carries `metadata["retries"]`.

- [ ] **Step 1: Failing tests**

```python
def test_retryable_predicate():
    from coding_agent.tools.executor import _retryable
    assert _retryable("tool timed out")
    assert _retryable("connection reset")
    assert not _retryable("approval denied")
    assert not _retryable("approval cancelled")
    assert not _retryable("invalid tool arguments")
    assert not _retryable("old_text must match exactly once")


async def test_executor_retries_transient_error_then_succeeds(tmp_path):
    calls = {"n": 0}
    class _FlakyTool:
        schema = ToolSchema(name="flaky", description="d", parameters={"type": "object"},
                            risk_level="read")
        args_model = type("A", (), {"model_validate": classmethod(lambda cls, a: a)})
        async def execute(self, arguments, *, context, signal):
            calls["n"] += 1
            if calls["n"] < 3:
                return ToolResult(tool_call_id="c", tool_name="flaky", ok=False,
                                  content="", error="tool timed out")
            return ToolResult(tool_call_id="c", tool_name="flaky", ok=True, content="ok")
    registry = ToolRegistry(); registry.register(_FlakyTool())
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), _NoopBroker(),
                            max_retries=2, retry_backoff_seconds=0.01)
    result = await executor.execute(ToolCall(id="c", name="flaky", arguments={}),
                                    run_id="r", workspace=tmp_path,
                                    permission_mode="full", signal=asyncio.Event())
    assert result.ok and calls["n"] == 3 and result.metadata.get("retries") == 2
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `_retryable(error)`: return False if any of `approval`, `cancelled`, `invalid`, `exactly once`, `old_text` in `error.lower()`; True otherwise (transient).
  - In `execute`, wrap the tool-run in a retry loop: after `_run_tool` returns a non-ok result, if `_retryable(error)` and attempts remain, sleep backoff and re-run; otherwise break. Track `retries` count; on final non-ok result, set `metadata["retries"] = attempts`.
  - Retry reuses the same approval decision (skip re-asking): the retry only re-enters the tool-run phase, not the approval phase.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Auto-retry transient tool failures with a retry eligibility predicate`

---

## Task 2: Provider idle watchdog + heartbeat + loop detection

**Files:** Modify `src/coding_agent/runtime/runner.py`, `src/coding_agent/runtime/events.py`; test `tests/test_w4_agent_robustness.py`.

**Interfaces:**
- Produces:
  - `AgentRunner(..., provider_idle_timeout_seconds: float = 90.0)`.
  - `RuntimeEvent("heartbeat", payload={"elapsed_seconds": float})` emitted while waiting on slow provider output.
  - `TurnOutcome(reason="provider_timeout")` when the stream is idle past the timeout.
  - `TurnOutcome(reason="progress_loop")` when ≥3 identical `(name, args)` signatures repeat with no differing call between them.

- [ ] **Step 1: Failing tests**

```python
async def test_provider_idle_timeout():
    # BlockingFakeProvider that never yields; runner with
    # provider_idle_timeout_seconds=0.05 should return provider_timeout.
    runner = _make_runner(store, BlockingFakeProvider(), registry, executor,
                          provider_idle_timeout_seconds=0.05)
    outcome = await runner.run_turn("hi", run_id="r", turn_id="t",
                                    signal=asyncio.Event())
    assert outcome.reason == "provider_timeout"


async def test_loop_detection():
    # RepeatingToolProvider("write_file") with executor returning ok each time;
    # a provider that re-emits the SAME tool call 3 times -> progress_loop.
    outcome = await runner.run_turn("go", run_id="r", turn_id="t",
                                    signal=asyncio.Event())
    assert outcome.reason == "progress_loop"
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `events.py`: add `"heartbeat"` to `RuntimeEventType`.
  - `runner.py`:
    - Loop detection: keep `last_signatures: deque(maxlen=3)` of `(call.name, json.dumps(call.arguments, sort_keys=True))`. After executing each wave, append the wave's signatures; if the deque length is 3 and all equal, emit a `notice` and return `TurnOutcome(reason="progress_loop")`.
    - Idle watchdog: in the provider stream loop, wrap `async for` with a per-event timeout. Use a helper that races `anext()` against a timeout via `asyncio.wait_for` on a task, and publishes a `heartbeat` event (with elapsed from `time.monotonic()` since turn start) on each timeout tick without aborting, until the idle window passes → break with `provider_timeout`. Reuse `provider_idle_timeout_seconds`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Detect provider idle timeouts and repeated no-progress tool calls`

---

## Task 3: Summary compression

**Files:** Modify `src/coding_agent/runtime/runtime.py`, `src/coding_agent/session/store.py`; test `tests/test_w4_agent_robustness.py`.

**Interfaces:**
- Produces:
  - `AgentRuntime.compact()` writes `summary` into the `compaction` record when a summarizer is available.
  - `AgentRuntime` gains `summarizer: Callable[[list[Message]], Awaitable[str]] | None` (injected; default uses `self._runner.provider.stream`).
  - `SessionStore.project_messages()` prepends a `system` message `"Summary of earlier conversation: <summary>"` from the latest `compaction` record's `summary`, when present.

- [ ] **Step 1: Failing tests**

```python
async def test_compact_stores_summary_and_project_prepends():
    # runtime with a fake summarizer returning "the summary"; store with a few
    # turns that will be truncated.
    await runtime.compact()
    records = store.records()
    compaction = [r for r in records if r.type == "compaction"]
    assert compaction and compaction[-1].payload.get("summary") == "the summary"
    messages = store.project_messages(include_open_turn=False)
    assert messages[0].message.role == "system"
    assert "the summary" in (messages[0].message.content or "")


async def test_compact_falls_back_without_summarizer():
    runtime = ...  # summarizer=None
    await runtime.compact()
    compaction = [r for r in store.records() if r.type == "compaction"]
    assert compaction and "summary" not in compaction[-1].payload
    messages = store.project_messages()
    assert not messages or messages[0].message.role != "system"
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `runtime.py`: `AgentRuntime.__init__` gains `summarizer: Callable | None = None`. In `_compact`, after computing `removed`/`retained` turns and before/at the `compaction` record write, if `removed` and a summarizer exists, build `removed_messages = [item.message for item in history if item.turn_id in removed]`, call the summarizer, and include `"summary": text` in the compaction payload. Wrap in try/except → on failure omit summary (fallback).
  - Default summarizer (when `None` is passed but a provider exists): a small async function that collects `text_delta`s from `provider.stream(messages, [], model=..., signal=...)` and returns the joined text; truncated to ~4000 chars.
  - `store.py`: in `project_messages`, scan records for the last `compaction` with a `summary`; if found, prepend a `Message(role="system", content=f"Summary of earlier conversation: {summary}")` as the first `SessionMessage` (record_id synthetic like `summary-<seq>`, turn_id None).
  - `app.py`: pass a summarizer when building the runtime (wired from the provider) unless disabled.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Summarize dropped turns during compaction and prepend the summary to context`

---

## Task 4: TUI surfacing (retry footer + notices)

**Files:** Modify `src/coding_agent/tui/reducer.py`, `src/coding_agent/tui/widgets.py`; test `tests/test_w4_agent_robustness.py`.

**Interfaces:**
- Consumes: `tool_finished` metadata `retries`; `notice` events for idle/loop.
- Produces: tool footer appends `· retried N×` when `metadata["retries"] > 0`; no-op handling of `heartbeat` in the reducer (ignored, state unchanged).

- [ ] **Step 1: Failing tests**

```python
def test_tool_footer_shows_retry_count():
    from coding_agent.tui.widgets import _tool_footer
    # existing _tool_footer(item) extended: item with metadata retries=2 renders
    # "· retried 2×" in the footer.


def test_heartbeat_is_noop_in_reducer():
    from coding_agent.tui.reducer import reduce
    state = initial_state(".", "test")
    before = state
    state = reduce(state, RuntimeEvent(type="heartbeat",
                                       payload={"elapsed_seconds": 5.0}))
    assert state == before
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `reducer.py`: ignore `heartbeat` (return state unchanged). In `tool_finished`, copy `metadata["retries"]` into the item (add `retries: int | None = None` to `TranscriptItem` in `state.py`).
  - `widgets.py`: `_tool_footer` appends `· retried {n}×` when `item.retries`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Surface retry counts in tool footers and ignore heartbeat events`

---

## Self-Review

- Spec §4 covered: retry (T1), watchdog/heartbeat/loop (T2), summary compression (T3), TUI surfacing (T4). ✅
- Types consistent: `_retryable`, `provider_idle_timeout_seconds`, `summarizer`, `heartbeat`. ✅
- `heartbeat` is a no-op for the reducer but W1's timer still drives the visible spinner. ✅
