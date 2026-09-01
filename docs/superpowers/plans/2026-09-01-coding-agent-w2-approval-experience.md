# W2 — Approval Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Show a diff preview for write/edit approvals, let the user remember a
decision (once/turn/session/always), and feed deny feedback back to the model.

**Architecture:** A `DecisionMemory` (policy/memory.py) maps normalized
`(tool_name, args)` signatures to remembered allow/deny decisions scoped
once/turn/session/always. `ToolExecutor` consults memory before the approval
policy and records the chosen decision. `_ApprovalBroker.resolve` and
`AgentRuntime.resolve_approval` gain `remember` and `feedback` params; denies
return `approval denied: <reason>[; <feedback>]`. `ApprovalScreen` renders a
`difflib` unified diff and a remember selector + optional feedback input.
Approvals persist as a new `approval` session record type (used later by W5).

**Tech Stack:** Python 3.11+, Textual 8.2.8, difflib (stdlib), pydantic.

**Spec:** `docs/superpowers/specs/2026-09-01-coding-agent-feature-roadmap-design.md` §2.

## Global Constraints

- No secrets in repo; the `always` allowlist writes to the config dir with mode `0600`.
- `policy/memory.py` is pure and synchronous (filesystem aside).
- Existing approval tests stay green; additive behavior only.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/policy/memory.py` (new): `DecisionMemory`.
- `src/coding_agent/tools/executor.py`: memory lookup, decision recording, deny feedback.
- `src/coding_agent/runtime/runtime.py`: broker + runtime `resolve` params; approval record.
- `src/coding_agent/session/models.py`: `approval` record type.
- `src/coding_agent/tui/widgets.py`: `render_approval_diff`; `ApprovalScreen` remember+feedback.
- `src/coding_agent/tui/app.py`: wire new resolve params.
- `tests/test_w2_approval_experience.py` (new).

---

## Task 1: `DecisionMemory` module

**Files:** Create `src/coding_agent/policy/memory.py`; test `tests/test_w2_approval_experience.py`.

**Interfaces:**
- Produces:
  - `signature(tool_name: str, arguments: dict) -> tuple[str, str]`
  - `class DecisionMemory:`
    - `remember(signature, decision: Literal["allow","deny"], scope: Literal["once","turn","session","always"]) -> None`
    - `lookup(signature) -> Literal["allow","deny"] | None`
    - `clear_turn()`, `clear_session()`
    - `load_always(path: Path | None = None)`, `persist_always(path: Path | None = None)`
    - `always_path` property (default `<config_dir>/approvals.json`; config_dir from `~/.config/coding-agent` unless overridden).

- [ ] **Step 1: Failing tests**

```python
import json
import os
from pathlib import Path

from coding_agent.policy.memory import DecisionMemory, signature


def test_signature_normalizes_arguments():
    assert signature("write_file", {"path": "a", "content": "x"}) == (
        "write_file", '{"content": "x", "path": "a"}')
    assert signature("write_file", {"content": "x", "path": "a"}) == signature(
        "write_file", {"path": "a", "content": "x"})


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
```

- [ ] **Step 2: Run and verify failure** (`pytest tests/test_w2_approval_experience.py -q`)

- [ ] **Step 3: Implement** — plain module, `json` sorted keys for the signature; `always` persisted as `{signature_str: decision}`; writes with `os.open(path, O_CREAT|O_WRONLY, 0o600)` then `chmod`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Add DecisionMemory with scoped approval decision persistence`

---

## Task 2: Executor consults memory and records decisions

**Files:** Modify `src/coding_agent/tools/executor.py`; test `tests/test_w2_approval_experience.py`.

**Interfaces:**
- Consumes: `DecisionMemory`, `ApprovalBroker.request`.
- Produces:
  - `ToolExecutor(..., memory: DecisionMemory | None = None)`.
  - `ToolExecutor.execute(..., remember: Literal["once","turn","session","always"] = "once", feedback: str | None = None)`.
  - Behavior: if `memory.lookup(signature)` returns a decision, short-circuit without calling the broker; else after approval, `memory.remember(sig, decision, scope=remember)`.
  - Deny with feedback → `ToolResult(error="approval denied: <reason>[; <feedback>]")`.

- [ ] **Step 1: Failing tests**

```python
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
    return ToolCall(id="c1", name="write_file",
                    arguments={"path": "a.txt", "content": "new"})


async def test_remembered_allow_short_circuits_broker(tmp_path):
    registry = ToolRegistry(); registry.register(make_write_file_tool())
    broker = _RecordingBroker()
    memory = DecisionMemory()
    mem_sig = signature("write_file", {"path": "a.txt", "content": "new"})
    memory.remember(mem_sig, "allow", scope="session")
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker, memory=memory)
    result = await executor.execute(_write_tool_call(), run_id="r",
                                    workspace=tmp_path, permission_mode="default",
                                    signal=asyncio.Event())
    assert result.ok
    assert broker.requests == []  # no approval asked


async def test_deny_with_feedback_reaches_model(tmp_path):
    registry = ToolRegistry(); registry.register(make_write_file_tool())
    broker = _RecordingBroker(); broker.decision = "deny"
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker)
    result = await executor.execute(_write_tool_call(), run_id="r",
                                    workspace=tmp_path, permission_mode="default",
                                    signal=asyncio.Event(),
                                    remember="turn", feedback="use relative path")
    assert not result.ok
    assert "approval denied" in (result.error or "")
    assert "use relative path" in (result.error or "")


async def test_decision_recorded_on_resolve(tmp_path):
    registry = ToolRegistry(); registry.register(make_write_file_tool())
    broker = _RecordingBroker(); broker.decision = "approve"
    memory = DecisionMemory()
    executor = ToolExecutor(registry, DefaultApprovalPolicy(), broker, memory=memory)
    await executor.execute(_write_tool_call(), run_id="r",
                           workspace=tmp_path, permission_mode="default",
                           signal=asyncio.Event(), remember="turn")
    assert memory.lookup(signature("write_file", {"path": "a.txt", "content": "new"})) == "allow"
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement** — in `execute`, before `policy.decide`, compute `sig = signature(call.name, active_call.arguments)` and `remembered = memory.lookup(sig) if memory else None`. If `remembered == "allow"` skip approval; if `"deny"` return the deny error. After an `ask` resolves to approve, `memory.remember(sig, "allow", scope=remember)`. On deny, build the error string with `request.reason` and `feedback`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Consult DecisionMemory and record decisions in ToolExecutor`

---

## Task 3: Broker/runtime resolve params + approval record

**Files:** Modify `src/coding_agent/runtime/runtime.py`, `src/coding_agent/session/models.py`; test `tests/test_w2_approval_experience.py`.

**Interfaces:**
- Produces:
  - `_ApprovalBroker.resolve(request_id, decision, remember="once", feedback=None)`.
  - `AgentRuntime.resolve_approval(request_id, decision, remember="once", feedback=None)`.
  - New `SessionRecord` type `"approval"` appended with payload `{request_id, tool_name, decision, scope, feedback, tool_call_id}`.
  - `RuntimeEvent("approval_resolved")` payload gains `remember` and `feedback`.

- [ ] **Step 1: Failing tests**

```python
async def test_resolve_approval_accepts_remember_and_feedback():
    # Build a real AgentRuntime with a runner whose executor asks approval,
    # drive submit, then resolve with remember="turn", feedback="no".
    # Assert: outcome error contains feedback; store has an "approval" record.
    store = SessionStore.create(tmp_path, workspace=str(tmp_path), model="test",
                                context_window=10_000)
    # ... construct runtime with FakeProvider emitting a write_file tool call
    run_id = await runtime.submit("write a.txt")
    # wait for approval_requested event
    await runtime.resolve_approval(request_id, "deny", remember="turn", feedback="no")
    records = store.records()
    assert any(r.type == "approval" for r in records)
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `session/models.py`: add `"approval"` to `RecordType` literal.
  - `runtime.py` `_ApprovalBroker.resolve`: accept `remember`/`feedback`; include in `approval_resolved` payload; append an `approval` record via `self.store` (give the broker a store reference or return the decision payload to the runtime to record — record in `AgentRuntime.resolve_approval` after broker resolve).
  - `AgentRuntime.resolve_approval`: pass `remember`/`feedback` through; after resolve, `self.store.append_new("approval", {...})`.
  - Executor deny path uses the recorded feedback (see Task 2). Ensure the deny error flows from the broker decision + feedback.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Persist approval decisions and pass remember/feedback through the runtime`

---

## Task 4: Approval diff preview + remember/feedback UI

**Files:** Modify `src/coding_agent/tui/widgets.py`, `src/coding_agent/tui/app.py`; test `tests/test_w2_approval_experience.py`.

**Interfaces:**
- Consumes: `ApprovalScreen` existing; `_approval_decision` callback in `app.py`.
- Produces:
  - `render_approval_diff(request: ApprovalRequest) -> rich.text.Text` — colorized unified diff for `write_file`/`edit_file`, truncated to 40 hunk lines with `… (N more lines)`.
  - `ApprovalScreen` shows the diff region for mutate-file tools, a remember selector (`once|turn|session|always`), and an optional feedback `Input`.
  - App callback: `_approval_decision(decision, remember, feedback)` → `resolve_approval(request_id, decision, remember, feedback)`.

- [ ] **Step 1: Failing tests**

```python
def test_render_approval_diff_write(tmp_path):
    (tmp_path / "a.txt").write_text("old\n")
    from coding_agent.session.models import ApprovalRequest
    from coding_agent.tui.widgets import render_approval_diff
    req = ApprovalRequest(request_id="1", run_id="r", tool_call_id="c",
                          tool_name="write_file", risk_level="mutate_file",
                          arguments={"path": "a.txt", "content": "new\n"})
    text = render_approval_diff(req, workspace=tmp_path)
    assert "+new" in str(text)
    assert "-old" in str(text)


def test_render_approval_diff_missing_file():
    # no current file -> all added lines
    req = ApprovalRequest(request_id="1", run_id="r", tool_call_id="c",
                          tool_name="write_file", risk_level="mutate_file",
                          arguments={"path": "new.txt", "content": "hello\n"})
    text = render_approval_diff(req, workspace=tmp_path)
    assert "+hello" in str(text)


def test_approval_screen_has_remember_and_feedback():
    # build ApprovalScreen with a mutate_file request; assert compose contains
    # the diff Static, the remember selector, and the feedback Input.
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `widgets.py`: `render_approval_diff` — resolve path under the workspace, read current content (or `""`), compute proposed content (write → `arguments["content"]`; edit → apply `old_text→new_text` to current). Use `difflib.unified_diff(current.splitlines(keepends=True), proposed.splitlines(keepends=True), fromfile=path, tofile=path)`; build a `rich.text.Text`, green `+`, red `-`, dim line numbers; truncate after 40 lines.
  - `ApprovalScreen`: add `render_approval_diff` output in a bordered Static (only when `request.tool_name in {"write_file", "edit_file"}`), a `Select` for remember scope, and an `Input` for feedback. The existing Approve/Deny buttons call a new callback signature carrying `(decision, remember, feedback)`.
  - `app.py`: `_approval_decision(decision, remember, feedback)`; wire `runtime.resolve_approval(request_id, decision, remember, feedback)`.
  - Runner factory in `app.py`: construct `ToolExecutor(registry, approval_policy, broker, memory=DecisionMemory())`.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Show approval diffs with remember scope and deny feedback in the approval panel`

---

## Self-Review

- Spec §2 covered: memory scopes (T1), executor short-circuit + feedback (T2), runtime record (T3), diff + UI (T4). ✅
- Types consistent: `DecisionMemory.lookup/remember`, `execute(..., remember, feedback)`, `resolve_approval(..., remember, feedback)`. ✅
- W5's inbox depends on the `approval` record type added in T3. ✅
