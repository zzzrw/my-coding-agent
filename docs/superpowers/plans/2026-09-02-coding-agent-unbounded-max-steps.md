# Unbounded Run Loop and Windowed Repetition Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `AgentRunner` unbounded by default (`max_steps: int | None = None`,
`None` = unlimited) so genuine long work is never silently cut off, always surface
a `warning` notice before the capped `max_steps` return (fixing session
`09f88d4d`, where the UI went silent after step 20), and replace the
consecutive-3-identical no-progress detector with a sliding-window repetition
detector over *executed* calls keyed on `(tool, canonical args, result
fingerprint)` so pure-read exploration can never misfire while real unchanged
repeats (even interleaved) are caught.

**Architecture:**
- `AgentRunner.max_steps` becomes `int | None = None`. The per-turn loop runs
  from an infinite iterator when `None` and from `range(1, max_steps + 1)` when
  an int, so every internal `steps`/`step` accounting is byte-for-byte today's.
  No body re-indent is required (the loop header is the only structural change).
- The capped path emits a `notice` (`level="warning"`,
  `"reached the max_steps limit without a final answer"`) immediately before
  the `TurnOutcome(reason="max_steps", ...)` return. The TUI reducer already
  renders notices as system rows — no reducer/model change.
- The `last_signatures: deque(maxlen=3)` consecutive detector is replaced by a
  bounded `deque(maxlen=8)` window of executed-call signature hashes. Signature
  = sha256 over `tool_name`, canonical sorted-JSON arguments, and a result
  fingerprint of the persisted `content`+`error` (each field hashed input capped
  at 4096 chars). After each executed wave the window absorbs that wave's
  signatures; if any signature occurs >= 3 times in the window the runner emits
  `"repeated tool call without progress"` and returns
  `reason="progress_loop"`. Calls rejected before execution (invalid-argument
  structured errors) are never counted.
- `max_steps` becomes an optional TOML config field (`Config.max_steps:
  int | None = None`, serialized only when set), validated `> 0` in `create_app`
  (mirroring `context_window`), and threaded into `AgentRunner` through
  `runner_factory`. Precedence (CLI > env > config > defaults) is unchanged; no
  new CLI/env flags.

**Tech Stack:** Python 3.11+, pydantic, pytest (asyncio auto mode), ruff.

**Spec:** `docs/superpowers/specs/2026-09-02-coding-agent-unbounded-max-steps.md`
(authoritative; committed `adbe4f2`).

## Global Constraints

- Every commit ends with the trailer line
  `Co-Authored-By: Claude <noreply@anthropic.com>` (second `-m` below).
- TDD per task: write the failing test first, watch it fail, implement the
  minimum, watch it pass. The suite is green at every commit boundary.
- `TurnOutcome.reason` enum and the reducer are unchanged (`max_steps` and
  `progress_loop` keep their meaning downstream; `progress_loop` broadens from
  "3 consecutive identical calls" to ">= 3 unchanged executed calls within the
  last 8").
- New tests are fast and deterministic: scripted fake providers and duck-typed
  executors only, no network, no real tools.
- Out of scope (do NOT build): a per-turn token budget and any
  "stop after K steps with no file mutation" heuristic.
- All commands assume the dev environment (`uv run pytest`, `uv run ruff`), as
  in the safe-ops plan.

## Sequencing Hazards

`src/coding_agent/runtime/runner.py` is shared by Tasks 1, 3, and 4, and the
three edits are intentionally disjoint (loop head vs. post-loop tail vs. the
detector block inside the wave loop), so each task is independently committable
with a green suite. Ordering rules:

- Task 2 MUST run after Task 1: passing `max_steps=None` to the runner requires
  the Task 1 signature change first (the old `int` signature would `TypeError`).
- Task 4 MUST run after Tasks 1 and 3: it builds on the Task 1 `step_source`
  loop-head line and the Task 3 capped-return tail, and its new tests already
  assert the Task 3 notice on the `max_steps` path.
- Determinism guard: never script an infinite distinct-call provider with a
  finite `ScriptedProvider` expecting a cap to stop it — the unbounded default
  would then depend on a provider list exhausting (`provider_error`). Tests that
  need bounded termination pass an explicit small `max_steps`; tests that must
  run past 20 prove it with a scripted concluding text.

## File Map

- `src/coding_agent/runtime/runner.py` (modify): `max_steps: int | None = None`;
  loop head from `itertools.count` when `None`; warning notice before the
  `max_steps` return; windowed repetition detector + signature/fingerprint
  helpers. Touched by Tasks 1, 3, 4 (ordered to keep the diff disjoint).
- `src/coding_agent/config/config.py` (modify): `Config.max_steps: int | None =
  None`; serialize when set.
- `src/coding_agent/app.py` (modify): validate `max_steps > 0` when set; pass
  `cfg.max_steps` into `AgentRunner` in `runner_factory`.
- `src/coding_agent/tui/reducer.py`, `src/coding_agent/runtime/models.py`
  (unchanged).
- `tests/test_runner.py` (modify): fix helper default; add the unbounded / cap
  notice / detector property tests.
- `tests/test_policy_runner_hardening.py` (modify): fix helper default.
- `tests/test_w4_agent_robustness.py` (modify): fix helper default.
- `tests/test_w6_configuration.py` (modify): config field + app wiring tests.
- `README.md` (modify): document the optional `max_steps` TOML key.

---

### Task 1: Make the runner unbounded by default (`max_steps: int | None = None`)

**Files:** Modify `tests/test_runner.py` and `src/coding_agent/runtime/runner.py`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_runner.py`, append:

```python
@pytest.mark.asyncio
async def test_unbounded_run_is_not_capped_at_old_default(tmp_path):
    # 21 distinct single-call waves then a final answer. With the old hidden
    # default of 20 this stops at max_steps; unbounded it completes.
    waves = [
        tool_response(json.dumps({"path": f"file{i}.py"})) for i in range(21)
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
```

(`json` is already imported in the module.)

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_runner.py -q`
Expected: `test_unbounded_run_is_not_capped_at_old_default` FAILS — with the
current `max_steps: int = 20` the run returns `reason == "max_steps"` at
`steps == 20` (or, after passing `None` into the current int signature, raises
`TypeError` inside `range(1, None + 1)`). `test_int_cap_still_bounded` PASSES
(the guard the int path must keep).

- [ ] **Step 3: Implement the unbounded default**

In `src/coding_agent/runtime/runner.py`:

Edit A — add `import itertools` to the imports (keep alphabetical with the stdlib
block: after `import json`, before `import time`).

Edit B — change the signature default:

```python
        max_steps: int | None = None,
```

Edit C — replace the loop head (currently `last_signatures` is declared first,
then `for step in range(1, self.max_steps + 1):`):

```python
        last_signatures: deque[tuple[str, str]] = deque(maxlen=3)
        step_source = (
            itertools.count(1)
            if self.max_steps is None
            else range(1, self.max_steps + 1)
        )
        for step in step_source:
```

No other body change in this task: the consecutive-3 detector and every
`steps=step`/`steps=step - 1`/final `steps=self.max_steps` return stay exactly
as today. `itertools.count` never terminates, so the code after the loop is
reachable only when the int `range` is exhausted — the unbounded path exits only
via the real return reasons.

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_runner.py tests/test_w1_realtime_feedback.py tests/test_w3_parallelism_undo.py tests/test_tui_visual_refresh.py tests/test_w4_agent_robustness.py -q`
Expected: all PASS. These five modules construct `AgentRunner` without an
explicit `max_steps`; after this change they run unbounded, and each scripted
provider concludes, so nothing hangs.

- [ ] **Step 5: Commit**

```bash
git add tests/test_runner.py src/coding_agent/runtime/runner.py
git commit -m "Make AgentRunner unbounded by default with optional int cap" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Fix test helpers that implicitly relied on the old default of 20

**Files:** Modify `tests/test_runner.py`, `tests/test_policy_runner_hardening.py`,
and `tests/test_w4_agent_robustness.py` (tests only — no source change).

Audit result: no runner test drives a provider that loops until the old 20 cap.
Every caller needing bounded termination already passes an explicit small int
(`test_runner.py::test_max_steps_stops_repeating_calls` passes `max_steps=2`;
`test_w4_agent_robustness.py::test_loop_detection_resets_on_differing_arguments`
passes `max_steps=3`). The three helper defaults `max_steps=20` test nothing
about the cap, so they move to `max_steps=None` so the unbounded path is
exercised and a future regression there is caught.

- [ ] **Step 1: Change the helper defaults**

- `tests/test_runner.py` line ~53: `def make_runner(tmp_path, provider, *, max_steps=20):` → `*, max_steps=None`.
- `tests/test_policy_runner_hardening.py` line ~214: `def make_runner(tmp_path, provider, *, max_steps=20):` → `*, max_steps=None`.
- `tests/test_w4_agent_robustness.py` line ~205: in `_make_runner`, `max_steps=20` → `max_steps=None`.

- [ ] **Step 2: Run and verify**

Run: `uv run pytest tests/test_runner.py tests/test_policy_runner_hardening.py tests/test_w4_agent_robustness.py -q`
Expected: all PASS. The W4 loop tests still terminate via the detector
(`test_loop_detection` trips at wave 3) or the explicit cap (`..._resets_...`,
`max_steps=3`), so the None default never risks an unbounded hang.

- [ ] **Step 3: Commit**

```bash
git add tests/test_runner.py tests/test_policy_runner_hardening.py tests/test_w4_agent_robustness.py
git commit -m "Stop test helpers defaulting to the removed 20-step cap" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Emit a warning notice before the capped `max_steps` return

**Files:** Modify `tests/test_runner.py` and `src/coding_agent/runtime/runner.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runner.py`:

```python
@pytest.mark.asyncio
async def test_max_steps_cap_emits_warning_notice_before_returning(tmp_path):
    # A provider that never concludes, capped small: the outcome must be
    # preceded by a warning notice (no silent exit, session 09f88d4d).
    provider = ScriptedProvider([tool_response(), tool_response()])
    runner, _, events, _ = make_runner(tmp_path, provider, max_steps=2)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "max_steps"
    notices = [event for event in events if event.type == "notice"]
    assert notices
    last = notices[-1]
    assert last.payload["level"] == "warning"
    assert "max_steps" in last.payload["message"]
    # The notice is the last event emitted before the outcome.
    assert events.index(last) == len(events) - 1
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_runner.py -q`
Expected: `test_max_steps_cap_emits_warning_notice_before_returning` FAILS — no
notice is currently emitted on the `max_steps` path.

- [ ] **Step 3: Implement the notice**

In `src/coding_agent/runtime/runner.py`, in `run_turn`, immediately before the
final post-loop return (reachable only when the int `range` is exhausted):

```python
        await self._emit(
            "notice",
            run_id,
            turn_id,
            level="warning",
            message="reached the max_steps limit without a final answer",
        )
        return TurnOutcome(
            reason="max_steps",
            final_text=final_text,
            steps=self.max_steps,
            usage=usage,
        )
```

This mirrors the provider-idle notice path. No reducer change is needed.

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_runner.py tests/test_w4_agent_robustness.py -q`
Expected: all PASS — the new notice test passes, and
`test_loop_detection_resets_on_differing_arguments` (which reaches
`reason == "max_steps"` under the cap) still passes with the added notice.

- [ ] **Step 5: Commit**

```bash
git add tests/test_runner.py src/coding_agent/runtime/runner.py
git commit -m "Emit warning notice before capped max_steps return" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Windowed repetition detector over executed calls with result fingerprints

**Files:** Modify `src/coding_agent/runtime/runner.py` and `tests/test_runner.py`.

Replaces the consecutive-3 detector with a sliding-window (W = 8) repetition
detector. Signatures cover executed calls only and include the result
fingerprint, so identical arguments whose result changed are distinct.

- [ ] **Step 1: Write the failing tests**

In `tests/test_runner.py`, add a stateful executor and the four property tests:

```python
class VersionedExecutor:
    """Returns distinct content per call, as if an external write changed the
    underlying file between identical-argument reads."""

    def __init__(self):
        self.calls = []
        self._n = 0

    async def execute(self, call, **kwargs):
        self.calls.append(call)
        self._n += 1
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=f"generation-{self._n}",
        )
```

```python
@pytest.mark.asyncio
async def test_distinct_pure_read_exploration_never_trips(tmp_path):
    # 20 distinct reads (pure exploration, zero writes) must never trip.
    waves = [
        tool_response(json.dumps({"path": f"src/mod{i}.py"})) for i in range(20)
    ]
    provider = ScriptedProvider([*waves, text_response("done")])
    runner, _, events, _ = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert not any(
        event.type == "notice"
        and "repeated tool call" in (event.payload.get("message") or "")
        for event in events
    )


@pytest.mark.asyncio
async def test_interleaved_identical_result_repeats_trip(tmp_path):
    # A,B,A,B,A with identical arguments AND identical results each time: the
    # repeated A (non-consecutive) must trip the detector.
    a = json.dumps({"path": "main.py"})
    b = json.dumps({"path": "other.py"})
    provider = ScriptedProvider([tool_response(a), tool_response(b),
                                 tool_response(a), tool_response(b),
                                 tool_response(a)])
    runner, _, events, _ = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "progress_loop"
    assert any(
        event.type == "notice"
        and event.payload.get("level") == "warning"
        and "repeated tool call" in (event.payload.get("message") or "")
        for event in events
    )


@pytest.mark.asyncio
async def test_identical_result_repeats_trip_consecutively(tmp_path):
    provider = ScriptedProvider(
        [tool_response(json.dumps({"path": "main.py"})) for _ in range(3)]
    )
    runner, _, events, _ = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "progress_loop"


@pytest.mark.asyncio
async def test_identical_args_with_changed_content_does_not_trip(tmp_path):
    # Re-reading the SAME path with identical arguments, but the executor's
    # content changes between reads (a write landed), is progress, not a loop.
    read = tool_response(json.dumps({"path": "data.txt"}))
    provider = ScriptedProvider([read, read, read, text_response("done")])
    runner, _, events, _ = make_runner(
        tmp_path, provider, executor=VersionedExecutor()
    )
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert not any(
        event.type == "notice"
        and "repeated tool call" in (event.payload.get("message") or "")
        for event in events
    )


@pytest.mark.asyncio
async def test_calls_rejected_before_execution_are_never_counted(tmp_path):
    # Repeated invalid-argument calls never execute, so they must never enter
    # the repetition window: the turn survives to the scripted final text.
    invalid = tool_response("[]")
    provider = ScriptedProvider([invalid, invalid, invalid, invalid,
                                 text_response("recovered")])
    runner, _, events, executor = make_runner(tmp_path, provider)
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )
    assert outcome.reason == "completed"
    assert executor.calls == []
    assert not any(
        event.type == "notice"
        and "repeated tool call" in (event.payload.get("message") or "")
        for event in events
    )
```

`make_runner` currently builds a `RecordingExecutor` internally; extend it with
an optional `executor` keyword (`def make_runner(tmp_path, provider, *,
max_steps=None, executor=None):` then `executor = executor or RecordingExecutor()`
before construction). All existing callers keep the default executor.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_runner.py tests/test_w4_agent_robustness.py -q`
Expected: four of the five new tests FAIL for the intended reasons under the
old consecutive detector and the old `(tool, args)` identity:
`test_interleaved_identical_result_repeats_trip` — A,B,A,B,A has no 3-run, so
it never trips and the exhausted `ScriptedProvider` ends in
`provider_error` instead of `progress_loop`;
`test_identical_result_repeats_trip_consecutively` and
`test_identical_args_with_changed_content_does_not_trip` — the old identity
ignores the result, so changed content still looks identical and trips
(`progress_loop`/premature stop rather than `completed`);
`test_calls_rejected_before_execution_are_never_counted` — the old detector
counts the repeated invalid-argument calls (they look identical) and trips,
instead of reaching the final text.
`test_distinct_pure_read_exploration_never_trips` PASSES (distinct args never
tripped even the old detector) and pins the must-not-misfire property.

- [ ] **Step 3: Implement the windowed detector**

In `src/coding_agent/runtime/runner.py`:

Edit A — imports: add `import hashlib`, and change
`from collections import OrderedDict, deque` to
`from collections import Counter, OrderedDict, deque`.

Edit B — add module constants and helpers near the top (after `_MUTATION_TOOLS`):

```python
_REPETITION_WINDOW = 8
_REPETITION_THRESHOLD = 3
_RESULT_FINGERPRINT_CAP = 4096


def _result_fingerprint(result: ToolResult) -> str:
    """Hash an executed call's persisted content and error.

    The hashed input is capped per field so huge outputs stay bounded; a
    content change (e.g. a file re-written between identical reads) must yield
    a different fingerprint.
    """
    content = (result.content or "")[:_RESULT_FINGERPRINT_CAP]
    error = (result.error or "")[:_RESULT_FINGERPRINT_CAP]
    return hashlib.sha256(
        f"{content}\x00{error}".encode("utf-8")
    ).hexdigest()


def _tool_call_signature(call: ToolCall, result: ToolResult) -> str:
    """Deterministic identity of one executed call and its observed result."""
    arguments = json.dumps(call.arguments, sort_keys=True)
    return hashlib.sha256(
        f"{call.name}\x00{arguments}\x00{_result_fingerprint(result)}".encode(
            "utf-8"
        )
    ).hexdigest()
```

Edit C — in `run_turn`, replace the detector state/loop-head lines added in Task
1:

```python
        last_signatures: deque[tuple[str, str]] = deque(maxlen=3)
        step_source = (
            itertools.count(1)
            if self.max_steps is None
            else range(1, self.max_steps + 1)
        )
```

with a window held across steps and waves:

```python
        signature_window: deque[str] = deque(maxlen=_REPETITION_WINDOW)
        step_source = (
            itertools.count(1)
            if self.max_steps is None
            else range(1, self.max_steps + 1)
        )
```

Edit D — replace the post-wave extend/check inside the wave loop:

```python
                last_signatures.extend(
                    (call.name, json.dumps(call.arguments, sort_keys=True))
                    for call in wave
                )
                if len(last_signatures) == 3 and len(set(last_signatures)) == 1:
                    await self._emit(
                        "notice",
                        run_id,
                        turn_id,
                        level="warning",
                        message="repeated tool call without progress",
                    )
                    return TurnOutcome(reason="progress_loop", steps=step, usage=usage)
```

with executed-call signature insertion and a window count:

```python
                for call, (_, result) in zip(wave, wave_results):
                    # Calls rejected before execution (invalid arguments) never
                    # count toward repetition; only genuinely executed calls do.
                    if call.id in invalid_by_id:
                        continue
                    signature_window.append(_tool_call_signature(call, result))
                if any(
                    count >= _REPETITION_THRESHOLD
                    for count in Counter(signature_window).values()
                ):
                    await self._emit(
                        "notice",
                        run_id,
                        turn_id,
                        level="warning",
                        message="repeated tool call without progress",
                    )
                    return TurnOutcome(reason="progress_loop", steps=step, usage=usage)
```

`invalid_by_id` and `wave_results` are already in scope at that point, aligned
with `wave` by `_run_call` ordering. `Counter` counts the whole bounded window
after each wave insertion, which yields the required semantics: distinct
signatures never repeat; identical signature (args and result) >= 3 times in the
last 8 trips — including interleaved `A,B,A,B,A` and multi-call waves; identical
args with changed content fingerprint as distinct.

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_runner.py tests/test_w4_agent_robustness.py -q`
Expected: all PASS. Existing W4 loop tests keep passing under the windowed rule:
`test_loop_detection` (3 identical consecutive writes → trip),
`test_loop_detection_resets_on_differing_arguments` (alternating write paths,
cap 3 → `max_steps`, no trip), and `test_max_steps_stops_repeating_calls` (2
identical reads under cap 2 → no trip, `max_steps` with notice).

- [ ] **Step 5: Commit**

```bash
git add tests/test_runner.py src/coding_agent/runtime/runner.py
git commit -m "Detect repetition over executed calls with result fingerprints" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Add the optional `max_steps` TOML config field

**Files:** Modify `src/coding_agent/config/config.py` and
`tests/test_w6_configuration.py`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_w6_configuration.py`, append to the config-model block:

```python
def test_load_reads_max_steps_from_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("model = 'm'\nmax_steps = 7\n")
    loaded = load_config(user_path=path)
    assert loaded.max_steps == 7


def test_max_steps_defaults_to_none_unbounded():
    assert Config().max_steps is None


def test_save_writes_max_steps_only_when_set(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, Config(model="m", api_key="k", max_steps=7))
    text = path.read_text(encoding="utf-8")
    assert "max_steps = 7" in text
    assert load_config(user_path=path).max_steps == 7

    unset = tmp_path / "unset.toml"
    save_config(unset, Config(model="m"))
    assert "max_steps" not in unset.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_w6_configuration.py -q`
Expected: FAILS — `Config` has no `max_steps` field (constructing one raises;
`load_config` silently drops the TOML key because it is not in
`Config.model_fields`). Because `Config` uses `extra="forbid"`, the field must
exist before any test can set it.

- [ ] **Step 3: Implement the field**

In `src/coding_agent/config/config.py`:

Edit A — add the field to `Config` (after `context_window`):

```python
    max_steps: int | None = None
```

`load_config` already forwards any `Config.model_fields` key, so a
`max_steps = N` line in the user or workspace TOML is picked up unchanged.

Edit B — in `_toml_lines`, emit only when set (mirror `context_window`):

```python
    if config.max_steps:
        lines.append(f"max_steps = {config.max_steps}")
```

Edit C — update the module docstring's settings list to include "optional
``max_steps``".

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_w6_configuration.py tests/test_config*.py -q`
Expected: all PASS, including the roundtrip tests.

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/config/config.py tests/test_w6_configuration.py
git commit -m "Add optional max_steps TOML config field" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Validate and wire `max_steps` through `create_app`, and document it

**Files:** Modify `src/coding_agent/app.py`, `tests/test_w6_configuration.py`,
and `README.md`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_w6_configuration.py` (the wiring test follows the W6
precedent of reading resolved settings off the app; the runner is already built
when `create_app` constructs the `AgentRuntime`, so no extra async is needed):

```python
def test_create_app_rejects_non_positive_max_steps(tmp_path, monkeypatch):
    _clear_all_env(monkeypatch)
    cfg = Config(model="m", api_key="k", max_steps=0)
    with pytest.raises(ConfigurationError):
        create_app(workspace=str(tmp_path), config=cfg)


def test_create_app_threads_config_max_steps_into_the_runner(tmp_path, monkeypatch):
    _clear_all_env(monkeypatch)
    app = create_app(
        workspace=str(tmp_path),
        config=Config(model="m", api_key="k", max_steps=5),
    )
    assert app.runtime._runner.max_steps == 5


def test_create_app_default_max_steps_is_unbounded_none(tmp_path, monkeypatch):
    _clear_all_env(monkeypatch)
    app = create_app(
        workspace=str(tmp_path),
        config=Config(model="m", api_key="k"),
    )
    assert app.runtime._runner.max_steps is None
```

(`app.runtime._runner` is private, matching the repo's existing private-API test
precedent such as `_retryable`/`_ApprovalBroker`.) Note `Config(max_steps=0)`
constructs fine after Task 5; the rejection must come from `create_app`.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_w6_configuration.py -q`
Expected: all three FAIL — `create_app` neither validates `max_steps` nor
threads it into `AgentRunner` (the runner still takes its own default `None`
from Task 1, so `_runner.max_steps` is `None` even when the config says 5, and
`max_steps=0` is silently accepted).

- [ ] **Step 3: Implement validation and wiring**

In `src/coding_agent/app.py`, in `create_app`, after `cfg = _load_config()` and
next to the existing `context_window` resolution:

```python
    if cfg.max_steps is not None and cfg.max_steps <= 0:
        raise ConfigurationError("max_steps must be greater than zero")
```

(An absent/`None` value — the fully environment-configured launch path where
`_load_config()` returns an empty `Config()` — passes through as unbounded.)

In `runner_factory`, pass the resolved value into `AgentRunner`:

```python
        return AgentRunner(
            provider=llm_provider,
            registry=registry,
            executor=executor,
            context_policy=context_policy,
            store=store,
            event_sink=lambda event: _discard_event(event),
            system_prompt=system_prompt,
            model=store.header.model,
            context_window=store.header.context_window,
            permission_mode=permission_mode,
            max_steps=cfg.max_steps,
        )
```

No new CLI/env surface and no `resolve_config` overlay: precedence is
unchanged.

In `README.md`, under the "Running locally" paragraph (after the environment
variable sentence), add one line documenting the optional TOML key:

```
An optional workspace `.coding-agent.toml` (or user `config.toml`) may set
`max_steps = N` to cap steps per turn; when absent the agent runs unbounded.
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_w6_configuration.py tests/test_integration_flow.py tests/test_bootstrap.py -q`
Expected: all PASS (wiring and end-to-end config still compose).

- [ ] **Step 5: Commit**

```bash
git add src/coding_agent/app.py tests/test_w6_configuration.py README.md
git commit -m "Validate and wire max_steps from config into the runner" \
  -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Full-suite and lint verification

**Files:** none (verification only; fix only what a task above missed).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all PASS. If only a timing-sensitive concurrency test flakes under a
loaded machine, re-run it alone to confirm; do not change it.

- [ ] **Step 2: Run the linters**

Run: `uv run ruff check src tests`
Run: `uv run ruff format --check src tests`
Expected: both clean.

- [ ] **Step 3: Manual property smoke (fast, deterministic)**

Run and confirm PASS (an unbounded run that stays alive past 20 distinct steps,
a small-cap run that returns `max_steps` with a notice, an interleaved repeat
that returns `progress_loop`, and a changed-content re-read that completes):

```python
import asyncio, json
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.runtime.models import LLMEvent, Message, ToolResult
from coding_agent.runtime.runner import AgentRunner
from coding_agent.session.store import SessionStore
from coding_agent.tools.registry import ToolRegistry


def _read_wave(path):  # one tool-call response
    return [
        LLMEvent(type="tool_call_start", tool_call_id="c", tool_name="read_file"),
        LLMEvent(type="tool_call_delta", tool_call_id="c", arguments_delta=json.dumps({"path": path})),
        LLMEvent(type="response_end", finish_reason="tool_calls"),
    ]


def _text(text="done"):  # one concluding response
    return [LLMEvent(type="text_delta", text=text)]


class _P:
    """ScriptedProvider equivalent: one stream() call yields one response."""
    def __init__(self, responses):
        self.responses = list(responses)
    async def stream(self, messages, tools, *, model, signal):
        for event in self.responses.pop(0):
            yield event


class _E:
    def __init__(self, versions=False):
        self.n = 0
        self.versions = versions
    async def execute(self, call, **kwargs):
        self.n += 1
        content = "x" if not self.versions else f"gen-{self.n}"
        return ToolResult(tool_call_id=call.id, tool_name=call.name, ok=True, content=content)


async def _run(responses, *, cap=None, versions=False):
    with TemporaryDirectory() as td:
        store = SessionStore.create(Path(td) / "s", workspace=td, model="m", context_window=1000)
        store.append_new("user_message",
            {"message": Message(role="user", content="go")}, run_id="r", turn_id="t")
        events = []
        async def sink(e): events.append(e)
        runner = AgentRunner(provider=_P(responses), registry=ToolRegistry(), executor=_E(versions),
            context_policy=TruncatePolicy(1000), store=store, event_sink=sink,
            system_prompt=Message(role="system", content="s"), model="m", context_window=1000,
            permission_mode="full", max_steps=cap)
        outcome = await runner.run_turn("go", run_id="r", turn_id="t", signal=asyncio.Event())
        return outcome, events

async def main():
    o, _ = await _run([_read_wave(f"f{i}.py") for i in range(25)] + [_text()])  # 25 distinct reads
    assert o.reason == "completed" and o.steps > 20  # unbounded, not capped at 20
    o, ev = await _run([_read_wave("a.txt"), _read_wave("a.txt")], cap=2)
    assert o.reason == "max_steps" and any(e.type == "notice" for e in ev)
    o, _ = await _run([_read_wave(p) for p in ("m", "o", "m", "o", "m")])  # interleaved A,B,A,B,A
    assert o.reason == "progress_loop"
    o, _ = await _run([_read_wave("d.txt")] * 3 + [_text()], versions=True)  # content changed
    assert o.reason == "completed"
    print("PASS")

asyncio.run(main())
```

- [ ] **Step 4: Commit (only if a fix was needed)**

If any step forced a source/test change not yet committed, commit it with a
descriptive message plus the standard trailer.

---

## Self-Review

- Unbounded by default: `max_steps: int | None = None`; loop iterates
  `itertools.count` when `None`, `range(1, cap + 1)` when int; internal
  `step`/`steps` accounting unchanged; `test_unbounded_run_is_not_capped_at_old_default`
  proves > 20 distinct steps complete. ✅ (Task 1)
- Never-silent cap: warning notice emitted as the last event before
  `reason == "max_steps"`; reducer already renders notices, no reducer change.
  ✅ (Task 3)
- Windowed detector: W = 8 window over executed-call sha256 signatures of
  `(tool, sorted-JSON args, result fingerprint)`; trips at >= 3 within window;
  invalid-arg rejected calls never counted. Properties tested: distinct
  pure-read exploration never trips; identical-result repeats trip consecutively
  AND interleaved `A,B,A,B,A`; identical args with changed content do not trip;
  repeated invalid-argument calls never enter the window. ✅ (Task 4)
- Config: `Config.max_steps: int | None = None`, serialized only when set,
  TOML-loaded via the existing `model_fields` forwarding; `create_app` rejects
  `max_steps <= 0` and passes `cfg.max_steps` into `AgentRunner`. ✅ (Tasks 5-6)
- Pre-existing tests reconciled: `test_runner`, `test_policy_runner_hardening`,
  and `test_w4_agent_robustness` helper defaults move to `None`; every caller
  needing a bound passes an explicit int; W4 loop tests keep passing under the
  windowed rule. ✅ (Task 2)
- Reason enum and reducer semantics unchanged; `progress_loop` broadened only.
  ✅
- Final gates: full `uv run pytest -q`, `ruff check`, `ruff format --check`, and
  a direct smoke over the four properties. ✅ (Task 7)
