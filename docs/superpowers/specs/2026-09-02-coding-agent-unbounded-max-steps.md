# Coding Agent Unbounded Run Loop and Windowed Repetition Detection

## Status

Agreed owner contract captured on 2026-09-02. This document amends two earlier
designs and is authoritative for the amended behavior:

- the per-turn step budget (`max_steps`) stated in the MVP design
  (`2026-08-30-coding-agent-mvp-design.md`), which fixed a default of 20;
- the W4 no-progress detector ("≥3 identical calls in a row") described in the
  feature roadmap design (`2026-09-01-coding-agent-feature-roadmap-design.md`).

The run-loop and detector changes land together because both touch
`AgentRunner.run_turn` and both exist to make the agent loop honest: unbounded
by default so genuine work is never silently cut off, and a windowed
repetition detector that flags only real no-progress stalls.

## 1. Motivation

### 1.1 Silent max-steps cut (session 09f88d4d)

In a real session the model consumed all 20 default steps with genuine tool
calls and never emitted a concluding prose answer. The runner returned
`TurnOutcome(reason="max_steps", ...)` with **no** `notice` event, so the TUI
stopped silently after the last tool row — no system row explained why the run
ended. Every other abnormal termination path (provider idle, repeated tool
call, provider error) already surfaces a notice; the capped-budget path was the
one silent exit. Two things are wrong here, not one:

1. A hidden numeric cap of 20 truncates legitimate long exploration that a
   model would otherwise conclude naturally.
2. Even when a cap is configured and reached, ending silently is unacceptable.

### 1.2 Exploration misfire risk in the current no-progress detector

The current detector (in `src/coding_agent/runtime/runner.py`) keeps
`last_signatures: deque(maxlen=3)` of `(tool_name, canonical arguments json)`
and returns `reason="progress_loop"` when the last 3 signatures are identical.
Identity is only `(tool, arguments)` — it ignores the tool *result*. That makes
it both too narrow and semantically wrong:

- It cannot see progress that is expressed through a *changing result*: two
  identical `read_file` calls either side of a write that changed the file look
  "the same" even though the second call observed new content (that is
  progress, not a loop).
- Consecutive-identical counting is a poor model of "no progress": a genuinely
  stalled agent usually re-issues the same call, but a legitimately productive
  agent can also repeat a pure read while it holds context. The owner wants a
  detector that keys on *executed calls plus their results* so broad pure-read
  exploration can never misfire, while genuinely unchanged repetition (even
  interleaved with other calls) is caught.

## 2. Run-loop contract

### 2.1 Unbounded by default

`AgentRunner.max_steps` becomes `int | None` with default `None` (= unlimited),
declared in `src/coding_agent/runtime/runner.py`. When `None`, the per-turn
loop in `AgentRunner.run_turn` is unbounded: it keeps stepping until the turn
concludes or exits for a real reason:

- `completed` — the model produced a response with no tool calls;
- `aborted` — the signal fired;
- `provider_error` / `session_error` / `provider_timeout`;
- `progress_loop` — the windowed repetition detector (§4) fired.

The hidden default of 20 no longer exists.

### 2.2 Optional int cap keeps today's semantics

When `max_steps` is an `int`, the loop stops when the step budget is exhausted
exactly as it does today, returning
`TurnOutcome(reason="max_steps", final_text=..., steps=<budget>, usage=...)`.
The step counter and the existing `steps` accounting on every other return path
are unchanged.

### 2.3 Never-silent cap

When a configured int cap IS hit, the runner MUST emit a
`RuntimeEvent(type="notice", level="warning", message=...)` immediately before
returning `TurnOutcome(reason="max_steps", ...)` — mirroring the sibling paths
that already emit notices (provider idle, repeated tool call). Suggested
message: `reached the max_steps limit without a final answer`.

No extra condition is required on this path: the `max_steps` return is only
reachable when the model has not concluded — the final response still requested
tool calls — so "the model never produced a final answer" holds by
construction.

The TUI reducer already renders `notice` as a system row
(`src/coding_agent/tui/reducer.py`, `notice` branch), so **no reducer change is
needed**. This fixes session 09f88d4d: the UI can no longer go silent at the
cap.

## 3. Configuration surface

`max_steps` becomes an optional TOML config value following how the other
settings are modeled in `src/coding_agent/config/config.py`:

- `Config.max_steps: int | None = None` (absent/`None` ⇒ unbounded). Because the
  model has `extra="forbid"`, `load_config` already forwards any
  `Config.model_fields` key from TOML, so a `max_steps = N` line in the user or
  workspace file is picked up automatically; `_toml_lines` gains an emit only
  when the value is set (a truthy/`None` guard like `context_window`, so an
  unset field is never written back).
- A configured value must be a positive integer; `max_steps <= 0` is a clear
  `ConfigurationError` (mirroring the `context_window` validation in
  `src/coding_agent/app.py`).
- Production wiring: the `runner_factory` closure in `create_app`
  (`src/coding_agent/app.py`) passes the resolved `config.max_steps` into
  `AgentRunner(max_steps=...)`. `None` (the common case, including a fully
  environment-configured launch that never touches a config file) yields the
  unbounded default.
- Precedence is unchanged (CLI > env > config > defaults). No new CLI or
  environment flags are added now; the config value is the only new surface.
  The interactive `SetupScreen` wizard is not required to gain a `max_steps`
  field (advanced setting, edit TOML directly).

## 4. Windowed repetition detector

Replace the consecutive-3-identical detector with a sliding-window repetition
detector over **executed** calls, also in `src/coding_agent/runtime/runner.py`.

### 4.1 Definition

- **Executed call**: a call whose tool actually ran (the result was produced by
  `executor.execute`). Calls rejected before execution (e.g. invalid-argument
  structured errors) never enter the window; only genuinely executed calls are
  counted.
- **Signature**: a deterministic hash over `(tool_name, canonical sorted-JSON
  arguments, result fingerprint)`.
  - canonical sorted-JSON arguments: `json.dumps(arguments, sort_keys=True)`;
  - result fingerprint: a hash of the call's persisted result `content` and
    `error`, with the hashed input capped at ~4096 chars per field.
  - Any stable hash (e.g. `hashlib.sha256` over a length-prefixed
    concatenation) is acceptable; the properties below must hold, not a
    specific hash function.
- **Window**: the last `W = 8` executed-call signatures, kept in a bounded
  deque shared across steps and waves.
- **Trip rule**: after each executed wave, insert that wave's signatures into
  the window; then, if any signature occurs `>= 3` times within the window,
  emit a `notice` (`level="warning"`,
  `message="repeated tool call without progress"`, matching today's wording) and
  return `TurnOutcome(reason="progress_loop", ...)`.

### 4.2 Properties that MUST hold (and be tested)

- **Must NOT trip** on broad pure-read/search exploration: many distinct reads,
  greps, and commands, zero writes, over dozens of steps (window holds only
  distinct signatures).
- **MUST trip** on re-reading the same unchanged file, or re-running the same
  unchanged grep/command, 3+ times within the window — identical arguments and
  identical result ⇒ identical signature.
- **Must NOT trip** on re-reading a file after a write changed its content:
  identical arguments but a different result fingerprint ⇒ different signature.
- **MUST trip** on interleaved repeats `A,B,A,B,A` (non-consecutive counting);
  each is counted anywhere in the window, not only as a run.

Consecutive triples remain a trip (a subset of the window rule), so the W4
"3 identical in a row" scenario keeps working. Because signatures are counted
per executed call, a single multi-call wave can also contribute several
identical signatures at once; the rule is uniform.

## 5. Out of scope

- A per-turn token/usage budget. The optional `max_steps` int is the only
  numeric knob for now.
- Any "stop after K steps with no file mutation" heuristic — that idea is
  rejected by the owner.
- Changes to the `TurnOutcome.reason` enum: `max_steps` and `progress_loop`
  keep their names and downstream reducer semantics. `progress_loop`'s meaning
  broadens from "3 consecutive identical calls" to "≥3 unchanged executed calls
  within the last 8".

## 6. Test expectations

### 6.1 Adjust existing tests

Run the full suite after the change. Any test that implicitly relied on the old
default of 20 to terminate a looping provider must now pass an explicit
`max_steps` (or its provider must be scripted to conclude). Audit the runner
test helpers (`tests/test_runner.py`, `tests/test_policy_runner_hardening.py`,
`tests/test_w4_agent_robustness.py`) that pass `max_steps=20` by default, and
the runner-construction helpers in `tests/test_w1_realtime_feedback.py`,
`tests/test_w3_parallelism_undo.py`, and `tests/test_tui_visual_refresh.py`
that pass no `max_steps` at all. A helper default that tests nothing about the
cap should move to `max_steps=None` so the unbounded path is exercised; tests
that need bounded termination pass an explicit small int. Existing assertions
on `reason == "progress_loop"` (e.g. W4 `test_loop_detection`) and on
`reason == "max_steps"` with an explicit cap (e.g. runner
`test_max_steps_stops_repeating_calls`, W4
`test_loop_detection_resets_on_differing_arguments`) must keep passing under
the windowed detector.

### 6.2 New tests

Fast and deterministic, fake providers only, no network:

1. **Default `None` does not stop a distinct-call provider.** A scripted
   provider returns many (> 20) distinct read/search tool calls then a final
   text; assert the turn completes with `reason == "completed"` and
   `steps > 20` — proving the loop was not capped at the old default.
2. **Configured int cap still returns `max_steps` and now emits a notice
   first.** A provider that never concludes, capped at a small int, yields
   `reason == "max_steps"`; assert a `notice` event with `level == "warning"`
   (message about reaching the limit) precedes the outcome — no silent exit.
3. **Exploration-like distinct pure-read sequence does not trip.**
   Many distinct reads/greps (dozens of waves) never emit the repeated-tool
   notice and terminate only by the scripted final text.
4. **Same-result repeats trip**, including interleaved `A,B,A,B,A`; assert
   `reason == "progress_loop"` and the `repeated tool call without progress`
   notice.
5. **Re-read after a content-changing write does not trip.** A fake filesystem
   write changes the file; a subsequent read of the same path with the same
   arguments returns new content ⇒ distinct signature ⇒ no `progress_loop` and
   the turn proceeds to a scripted completion.

## 7. Files and verification

Component changes:

| Component | Change |
|---|---|
| `runtime/runner.py` | `max_steps: int \| None = None`; unbounded loop when `None`; warning notice before the `max_steps` return; windowed repetition detector (W=8, ≥3 within window) over executed-call signatures with result fingerprints |
| `config/config.py` | `Config.max_steps: int \| None = None`; serialize when set |
| `app.py` | validate `max_steps > 0` when set; pass `config.max_steps` into `AgentRunner` in `runner_factory` |
| `runtime/models.py` | unchanged (reason enum already contains `max_steps`/`progress_loop`) |
| `tui/reducer.py` | unchanged (notice already renders as a system row) |
| `tests/*` | §6.1 adjustments + §6.2 new coverage |

Final gates before commit: `uv run pytest -q`, `uv run ruff check src tests`,
and `uv run ruff format --check src tests` all green.
