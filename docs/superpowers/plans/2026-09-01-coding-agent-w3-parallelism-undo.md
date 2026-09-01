# W3 — Parallelism & Undo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run a turn's tool calls concurrently where safe (same-file mutations
serialized), add `/undo` to revert the latest write/edit, and show a plan banner
for multi-call turns.

**Architecture:** `AgentRunner` partitions parsed calls into ordered "waves"
(parallel-safe reads and distinct-path calls batch; same-path mutations
serialize) executed with `asyncio.gather`, collecting results back into call
order. A `MutationJournal` in `ToolExecutor` snapshots file state before each
`mutate_file` success; `AgentRuntime.undo()` restores the latest snapshot and
emits a notice. The runner emits a `plan_preview` event for 2+ calls; the
reducer inserts a plan transcript row.

**Tech Stack:** Python 3.11+, asyncio, Textual 8.2.8.

**Spec:** `docs/superpowers/specs/2026-09-01-coding-agent-feature-roadmap-design.md` §3.

## Global Constraints

- `signal` cancellation semantics unchanged; each call still returns a `ToolResult`.
- Event order per call (started → finished) is preserved; transcript keys on `tool_call_id`.
- `tui/reducer.py` stays pure.
- Existing 395+ tests stay green.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/runtime/runner.py`: wave partitioner + gather; `plan_preview` emission.
- `src/coding_agent/runtime/events.py`: `plan_preview` event type.
- `src/coding_agent/tools/executor.py`: `MutationJournal` (snapshot before mutate_file).
- `src/coding_agent/runtime/runtime.py`: `undo()` method.
- `src/coding_agent/tui/commands.py`: `undo` command.
- `src/coding_agent/tui/app.py`: `/undo` dispatch.
- `src/coding_agent/tui/reducer.py`: `plan_preview` row.
- `tests/test_w3_parallelism_undo.py` (new).

---

## Task 1: Wave partitioner + parallel execution

**Files:** Modify `src/coding_agent/runtime/runner.py`; test `tests/test_w3_parallelism_undo.py`.

**Interfaces:**
- Consumes: parsed `list[ToolCall]`, `ToolExecutor.execute`.
- Produces:
  - `partition_waves(calls: list[ToolCall]) -> list[list[ToolCall]]` (module-level, pure).
  - `AgentRunner` executes each wave with `asyncio.gather`, collects results into call order.

- [ ] **Step 1: Failing tests**

```python
def test_partition_waves_batches_parallel_safe_and_serializes_same_path():
    from coding_agent.runtime.runner import partition_waves
    from coding_agent.runtime.models import ToolCall
    calls = [
        ToolCall(id="1", name="read_file", arguments={"path": "a"}),      # parallel-safe
        ToolCall(id="2", name="read_file", arguments={"path": "b"}),      # parallel-safe
        ToolCall(id="3", name="write_file", arguments={"path": "x"}),     # mutate
        ToolCall(id="4", name="write_file", arguments={"path": "x"}),     # same path -> serial
        ToolCall(id="5", name="read_file", arguments={"path": "c"}),      # parallel-safe
    ]
    waves = partition_waves(calls)
    assert waves[0] == [calls[0], calls[1]]          # both reads batch
    assert calls[3] in waves[1] or calls[3] in waves[2]  # ordered relative to calls[2]
    # flatten preserves call order
    assert [c.id for w in waves for c in w] == [c.id for c in calls]
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

```python
def partition_waves(calls):
    waves: list[list] = []
    used_keys: set[str] = set()
    for call in calls:
        key = _mutation_key(call)
        can_join = (
            call_schema_parallel_safe(call.name)
            or (key is not None and key not in used_keys)
        )
        if can_join and waves and _compatible(waves[-1], call, key):
            waves[-1].append(call)
        else:
            waves.append([call])
            used_keys = set()
        if key is not None:
            used_keys.add(key)
    return waves
```

  `_mutation_key(call)`: for `write_file`/`edit_file`, the resolved path string
  (resolve relative under workspace); otherwise `None`. `_compatible` checks the
  call's key (or tool name for parallel-safe reads) doesn't collide with keys
  already in the wave. For non-parallel-safe, non-path tools, each starts its own
  wave. Note: a wave must not contain two calls whose mutation keys collide.

  In `run_turn`, replace the sequential loop with:

```python
for wave in partition_waves(parsed_calls):
    async def _run_call(call):
        ... append tool_call record, emit tool_started, build output_sink,
        execute, append tool_result, emit tool_finished, return (call.id, result)
    results = await asyncio.gather(*[_run_call(c) for c in wave])
```

  Order results by `call.id` within the wave and continue the loop as before.

- [ ] **Step 4: Run focused + full suite** (add a timing test: two slow tools on
  distinct paths complete before a serialized same-path third; control with
  `asyncio.sleep` in fake tools and `time.monotonic` deltas).

- [ ] **Step 5: Commit** — `Execute tool calls in parallel-safe waves within a turn`

---

## Task 2: `plan_preview` event + plan banner

**Files:** Modify `src/coding_agent/runtime/events.py`, `src/coding_agent/runtime/runner.py`, `src/coding_agent/tui/reducer.py`; test `tests/test_w3_parallelism_undo.py`.

**Interfaces:**
- Produces:
  - `RuntimeEvent("plan_preview", payload={"tool_calls": [{"name","arguments"}]})` emitted once per assistant response when `len(parsed_calls) >= 2`.
  - Reducer inserts a `system` transcript row `"→ N tool calls: write_file(a), run_command(b), ..."`.

- [ ] **Step 1: Failing tests**

```python
def test_reducer_renders_plan_banner():
    from coding_agent.tui.reducer import reduce
    from coding_agent.tui.state import initial_state
    from coding_agent.runtime.events import RuntimeEvent
    state = initial_state(".", "test")
    state = reduce(state, RuntimeEvent(
        type="plan_preview", run_id="r", turn_id="t",
        payload={"tool_calls": [
            {"name": "write_file", "arguments": {"path": "a"}},
            {"name": "run_command", "arguments": {"command": "ls"}},
        ]}))
    systems = [r for r in state.transcript if r.kind == "system"]
    assert systems and "write_file(a)" in systems[0].text
    assert "run_command(ls)" in systems[0].text
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `events.py`: add `"plan_preview"` to `RuntimeEventType`.
  - `runner.py`: after `parsed_calls` is finalized and `parsed_calls` non-empty and `len(parsed_calls) >= 2`, emit `plan_preview` with a compact call list (`{name, arguments}`).
  - `reducer.py`: on `plan_preview`, build the summary string and append a `system` row (id via `_next_suffixed_id(transcript, "plan-", kind="system")`), styled dim by the `system` row CSS.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Show a plan banner when a turn queues multiple tool calls`

---

## Task 3: `MutationJournal` + `/undo`

**Files:** Modify `src/coding_agent/tools/executor.py`, `src/coding_agent/runtime/runtime.py`, `src/coding_agent/tui/commands.py`, `src/coding_agent/tui/app.py`; test `tests/test_w3_parallelism_undo.py`.

**Interfaces:**
- Produces:
  - `class MutationJournal:` with `push(path, original: str | None)` and `pop() -> tuple[Path, str | None] | None`.
  - `ToolExecutor(journal: MutationJournal | None = None)`; snapshots before and pushes after successful `mutate_file` tools.
  - `AgentRuntime.undo() -> None` (reserve operation, pop, restore/unlink, emit `notice` with `command="undo <path>"`).
  - `parse_command("/undo")` → `Command("undo", [])`; app dispatch calls `runtime.undo()`.

- [ ] **Step 1: Failing tests**

```python
def test_undo_restores_overwritten_file(tmp_path):
    store = SessionStore.create(tmp_path, workspace=str(tmp_path), model="test",
                                context_window=10_000)
    # real runtime, write a.txt (requires approval -> full permission to skip),
    # then call runtime.undo(); file content restored.
```

  Concretely: build a runtime with `permission_mode="full"` and a FakeProvider
  that calls `write_file` twice with different content; after the turn,
  `a.txt` holds the second content; `await runtime.undo()` restores the first.
  Also test undo of a created file removes it, and undo with an empty journal is
  a no-op that still emits a notice.

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `executor.py`: `MutationJournal` (thread-safe list). In `execute`, before running a tool whose schema `risk_level == "mutate_file"`, resolve the path and read current content (or `None` if missing); after a successful result, `journal.push(path, original)`. Path resolution mirrors the tools (`workspace / path`, resolved).
  - `runtime.py`: `async def undo()` — `_reserve_operation`, pop journal (journal accessible via the executor — pass it into the runtime or share the executor instance; the runtime's `_runner` holds the executor; add `self._runner.executor.journal` access or store the journal on the runtime at construction via runner_factory closure). Restore content with atomic write; if original is `None`, unlink the file. Emit `notice` with `{"command": f"undo {path}", "message": f"undid write to {path}"}`. `_release_operation`.
  - `commands.py`: add `CommandSuggestion("undo", "Undo the last file write/edit", usage="/undo")` to `_COMMANDS`; `SUPPORTED_COMMANDS` auto-includes it.
  - `app.py`: dispatch `undo` → `self.run_worker(self._runtime_action("undo"), ...)`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Add /undo to restore the most recent file mutation`

---

## Self-Review

- Spec §3 covered: waves + gather (T1), plan preview (T2), journal + `/undo` (T3). ✅
- Types consistent: `partition_waves`, `MutationJournal.push/pop`, `runtime.undo()`. ✅
- W3 touches `runner.py` again after W1; it must build on the W1 output-sink loop. ✅
