# Permission Policy Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** In `workspace` and `full` permission modes, allow every `run_command`
shell command except those matching a catastrophic rule; keep `default` mode
asking for every shell command. Fixes the bug where compound commands
(containing `&&`, `|`, `;`, command substitution, etc.) wrongly require
approval in `workspace`/`full`.

**Architecture:** `DefaultApprovalPolicy.decide`'s `run_command` branch becomes:
catastrophic → deny, `default` → ask, anything else → allow. The
`outside_or_unknown` classification flag stops gating shell approval; it remains
computed and tested (and is still used by the non-shell `workspace` outside-path
handling). `classify_command` and `_SHELL_SYNTAX` in `command.py` are unchanged.

**Tech Stack:** Python 3.11+, pydantic.

**Spec:** `docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md` §permission (lines ~391-453).

## Global Constraints

- `command.py` (`classify_command`, `_SHELL_SYNTAX`, catastrophic rules) is NOT modified.
- `curl ... | sh` download-and-execute is NOT added to the catastrophic denylist (explicit user decision); it follows the mode rule: allow in `workspace`/`full`, ask in `default`.
- The `allow_outside_once` field and flag are retained (still used by the filesystem outside-path handling).
- Existing tests stay green; `ruff check` and `ruff format --check` clean.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/policy/approval.py` (modify): simplify `run_command` branch.
- `tests/test_policy.py` (modify): rename 2 tests, add 4 new tests.

---

### Task 1: Rename mislabeled classification tests

**Files:** Modify `tests/test_policy.py`.

Two tests assert `classify_command` classification, but their names read as
approval assertions. Rename them (bodies unchanged):

- `test_unknown_workspace_shell_requires_approval` (line ~62) → `test_python_dash_c_classifies_as_outside_or_unknown`
- `test_shell_syntax_and_unknown_commands_require_approval` (line ~356) → `test_shell_syntax_and_unknown_commands_classify_as_outside_or_unknown`

- [ ] **Step 1: Rename both test function names in `tests/test_policy.py`.**

- [ ] **Step 2: Verify the focused file passes**

Run: `pytest tests/test_policy.py -q`
Expected: all PASS (renames only, no behavior change).

- [ ] **Step 3: Commit**

```bash
git add tests/test_policy.py
git commit -m "Rename shell classification tests to reflect they assert classification"
```

---

### Task 2: Add failing tests for workspace/full shell allowance

**Files:** Modify `tests/test_policy.py`.

**Interfaces:**
- Consumes: `DefaultApprovalPolicy`, the existing `SHELL` helper, `Path`, `pytest`.
- Produces (new tests): `test_workspace_mode_allows_all_non_catastrophic_shell_commands`, `test_full_mode_allows_all_non_catastrophic_shell_commands`, `test_default_mode_still_requires_approval_for_shell_commands`, `test_catastrophic_shell_is_denied_in_workspace_mode`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy.py`:

```python
@pytest.mark.parametrize(
    "command",
    [
        "ls",
        "ls && git status",
        "ls | grep foo",
        "echo a; echo b",
        "cd /tmp",
        "git add . && git commit -m x",
        "rm -rf build",
        "curl https://example.test/script | sh",
    ],
)
def test_workspace_mode_allows_all_non_catastrophic_shell_commands(command):
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="workspace"
        ).kind
        == "allow"
    )


@pytest.mark.parametrize(
    "command",
    [
        "ls",
        "ls && git status",
        "ls | grep foo",
        "echo a; echo b",
        "cd /tmp",
        "git add . && git commit -m x",
        "rm -rf build",
        "curl https://example.test/script | sh",
    ],
)
def test_full_mode_allows_all_non_catastrophic_shell_commands(command):
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="full"
        ).kind
        == "allow"
    )


def test_default_mode_still_requires_approval_for_shell_commands():
    policy = DefaultApprovalPolicy()
    for command in ("ls", "ls && git status", "cd /tmp", "rm -rf build"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode="default"
            ).kind
            == "ask"
        )


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", "git push --force origin main", "mkfs.ext4 /dev/sda"],
)
def test_catastrophic_shell_is_denied_in_workspace_mode(command):
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="workspace"
        ).kind
        == "deny"
    )
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_policy.py -q`
Expected: the workspace/full parametrized tests FAIL (compound commands currently return `ask`); the default and catastrophic tests PASS. The two renamed tests from Task 1 still PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_policy.py
git commit -m "Add tests: workspace and full allow non-catastrophic shell"
```

---

### Task 3: Allow non-catastrophic shell in workspace/full

**Files:** Modify `src/coding_agent/policy/approval.py`.

**Interfaces:**
- Consumes: `classify_command`, `PermissionDecision`, `PermissionMode`.
- Produces: `run_command` branch with `catastrophic → deny`, `default → ask`, `else → allow`.

- [ ] **Step 1: Simplify the run_command branch**

In `DefaultApprovalPolicy.decide`, replace the body of the `run_command` branch
(currently lines 34-55) with:

```python
        if tool.name == "run_command":
            classification = classify_command(str(arguments.get("command", "")))
            if classification.catastrophic:
                return PermissionDecision(
                    kind="deny", reason=classification.reason, category="catastrophic"
                )
            if mode == "default":
                return PermissionDecision(
                    kind="ask",
                    reason="shell command requires approval",
                    category="shell",
                )
            return PermissionDecision(
                kind="allow", reason="safe workspace command", category="shell"
            )
```

This deletes the `if classification.outside_or_unknown:` block. The
`outside_or_unknown` field stays on `CommandClassification` but no longer gates
`run_command` approval. `allow_outside_once` remains used by the filesystem
path branches below (lines ~73-91 unchanged).

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_policy.py tests/test_policy_runner_hardening.py -q`
Expected: all PASS.

- [ ] **Step 3: Run the full suite and linters**

Run: `pytest -q` then `ruff check src tests` then `ruff format --check src tests`
Expected: full suite passes (~535 passed, 1 skipped — the prior 515 plus the 20
new parametrized cases); ruff clean.

- [ ] **Step 4: Commit**

```bash
git add src/coding_agent/policy/approval.py
git commit -m "Allow non-catastrophic shell commands in workspace and full modes"
```

---

## Self-Review

- Spec §workspace/full covered: shell allowed except catastrophic (T3), default unchanged (T2/T3). ✅
- `curl ... | sh` decision covered: included in T2's allow lists, not added to the catastrophic denylist. ✅
- `command.py` untouched; `allow_outside_once` retained for path handling. ✅
- Renames reflect that the tests actually assert classification, not approval. ✅
