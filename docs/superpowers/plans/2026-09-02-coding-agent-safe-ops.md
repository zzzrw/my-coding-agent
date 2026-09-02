# Safe Operations Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let an agent doing real work (clearing workspace/build dirs, cleaning a
user-level cache, stopping a stale dev server) proceed without being
hard-blocked or forced into shell gymnastics, while true system-destroying
commands stay denied in every permission mode. Implement the approved design in
`.scratch/design-brief.md` sections A (permission), B (delete tools), C (system
prompt); section D (specs) is already committed (`e16d7e8`); section E is the
final task here.

**Architecture:**
- `_rm_is_catastrophic` becomes scoped: an `rm` operand is catastrophic only when
  it can wipe a protected root (`/`, `~`, `$HOME`, `${HOME}`, `/home`, `/root`),
  a whole user home (`/home/<name>` / `/home/<name>/*`, `~/*`, `$HOME/*`,
  `${HOME}/*`), or anything under `/root/`. Home subpaths are NOT catastrophic
  and fall through to the mode rule. With destructive flags, an operand holding
  `..` or a `$VAR` other than `$HOME`/`${HOME}` stays catastrophic.
- The `/dev/` catastrophic rules get a `/dev/null` carve-out via a negative
  lookahead; real block devices keep matching.
- `git reset --hard` detection moves to a token analyzer (the existing
  global-option skipper is renamed to a shared `_git_subcommand_index`) so
  interleaved options (`-q`, `-C <dir>`, `-c k=v`, `--git-dir`, `--exec-path`)
  cannot bypass it.
- Two new workspace-bounded `mutate_file` tools `remove_file` and
  `clear_directory` in `tools/filesystem.py`, registered in `app.py`, resolved
  through `resolve_tool_path` exactly like `write_file`.
- The module-level `SYSTEM_PROMPT` constant becomes a `build_system_prompt(
  workspace, permission_mode)` builder so the model is told the safety
  boundaries, the active mode, and the workspace root up front.

**Tech Stack:** Python 3.11+, pydantic, pytest (asyncio auto mode), ruff.

**Spec:** `docs/superpowers/specs/2026-08-30-coding-agent-mvp-design.md`
§8 Tools and Execution and §9 Permission and Safety (already updated in commit
`e16d7e8`); authoritative decisions in `.scratch/design-brief.md` (sections A,
B, C, E).

## Global Constraints

- Every commit ends with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  (second `-m` in the commit commands below).
- Section A4 "still catastrophic" cases stay denied: whole-root/whole-home
  `rm`; `git push --force`; destructive `git reset --hard`; `mkfs`/`fdisk`/
  `shutdown`/`reboot`/`poweroff`; `dd`/redirection to real block devices;
  fork bomb. True system commands are unchanged.
- The token analyzers already in `classify_command` (`_rm_is_catastrophic`,
  `_git_push_is_catastrophic`, `_git_remote_is_catastrophic`,
  `_git_config_shell_alias_is_catastrophic`, `_nested_shell_is_catastrophic`,
  `_command_substitutions_are_catastrophic`) keep running BEFORE the
  `_CATASTROPHIC` regex loop; that ordering is unchanged.
- The contiguous `git\s+reset\s+--hard` regex is kept (it also covers
  quoted/verbatim occurrences) AND the new token analyzer is added.
- `rm -rf //` stays non-catastrophic (unchanged from today); trailing-slash
  equivalence is applied only to the protected-root/whole-home forms.
- The bare `/home` mount is included in the protected-root set (wiping it
  destroys every user home and prior behavior already denied it); home
  subpaths `/home/<name>/<anything deeper>` stay mode-governed.
- `remove_file`/`clear_directory` are NOT routed through `run_command`; they
  are `mutate_file` tools so the executor's `MutationJournal` snapshots them.
- Existing tests stay green at every task boundary; ruff clean at the end.
- Executor commands below assume an activated dev environment (`uv run pytest`
  or the project venv's `pytest`). The tracked repo does not commit `.scratch/`.

## File Map

- `src/coding_agent/policy/command.py` (modify): scoped `_rm_is_catastrophic`,
  `/dev/null` carve-outs, `git reset --hard` token analyzer.
- `src/coding_agent/tools/filesystem.py` (modify): add `remove_file` and
  `clear_directory` tools + factories.
- `src/coding_agent/app.py` (modify): register both tools in `_make_registry`;
  replace the `SYSTEM_PROMPT` constant with `build_system_prompt(...)`.
- `tests/test_policy.py` (modify): reconcile rm assertions, add `/dev/null`
  and `git reset --hard` tests.
- `tests/test_tools_filesystem.py` (modify): add delete-tool tests.
- `tests/test_app_system_prompt.py` (new): system-prompt guidance tests.

---

### Task 1: Narrow the `rm` catastrophic rule to protected roots and whole homes

**Files:** Modify `tests/test_policy.py` and `src/coding_agent/policy/command.py`.

Reconcile the OLD broad rm expectations (`rm -rf ~/projects`,
`rm -rf ${HOME}/cache`, and every `/home/...` operand were catastrophic) to the
new scoped semantics and implement them. Guard tests pin the still-denied
whole-home/root forms.

- [ ] **Step 1: Write the failing / reconciled tests**

In `tests/test_policy.py`:

Edit A — in `test_more_catastrophic_command_variants_are_always_denied`,
remove the `"rm -rf ~/projects"` entry and replace the whole parametrize list
with:

```python
@pytest.mark.parametrize(
    "command",
    [
        "fdisk /dev/sda",
        "/sbin/fdisk /dev/sda",
        "mkfs /dev/sda",
        "mkfs.ext4 /dev/sda",
        "echo bad >/dev/sda",
        "git clean --force -d",
        "git clean -xdf",
        ": () { : | : & }; :",
        "rm -rf /home/user",
        "rm -rf /root/.ssh",
        "rm -rf /root",
        "rm -rf /home",
        "rm -rf /home/user/*",
        "rm -rf /home/user/",
        "rm -rf ~/*",
        "rm -rf ~/",
        "rm -rf $HOME/*",
        "rm -rf ${HOME}/*",
        "rm -rf /home/me",
    ],
)
def test_more_catastrophic_command_variants_are_always_denied(command):
    assert classify_command(command).catastrophic is True
```

Edit B — in `test_executable_paths_variables_and_short_force_are_catastrophic`,
replace `"rm -rf ${HOME}/cache"` with `"rm -rf ${HOME}"` so the list is:

```python
@pytest.mark.parametrize(
    "command",
    [
        "/bin/rm -rf /",
        "rm -rf $HOME",
        "rm -rf ${HOME}",
        "git push -f origin main",
    ],
)
def test_executable_paths_variables_and_short_force_are_catastrophic(command):
    assert classify_command(command).catastrophic is True
```

Edit C — append a new test asserting the home-subpath allowances:

```python
@pytest.mark.parametrize(
    "command",
    [
        "rm -rf ~/projects",
        "rm -rf ${HOME}/cache",
        "rm -rf /home/me/.cache/foo",
        "rm -rf /home/me/.cache",
        "rm -rf ~/foo",
        "rm -rf $HOME/.cache/foo",
        "rm -rf /home/me/workspace/build/app.js",
        "rm -f /home/me/build/app.js",
    ],
)
def test_home_subpath_removals_are_not_catastrophic(command):
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is False
    for mode in ("workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "allow"
        )
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="default"
        ).kind
        == "ask"
    )
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_policy.py -q`
Expected: `test_home_subpath_removals_are_not_catastrophic` FAILS (every
subpath is currently catastrophic). The edited catastrophic tests still PASS
(all remaining entries are catastrophic today). Overall run FAILS.

- [ ] **Step 3: Implement the scoped `_rm_is_catastrophic`**

In `src/coding_agent/policy/command.py`, replace the whole `_rm_is_catastrophic`
function (currently lines ~32-57) with the helper set below. Add the module
constants and helpers above the function:

```python
_RM_PROTECTED_ROOT_OPERANDS = {"/", "~", "$HOME", "${HOME}", "/home", "/root"}
_RM_HOME_CONTENT_GLOBS = {"~/*", "$HOME/*", "${HOME}/*"}


def _rm_operand_wipes_protected_root(operand: str) -> bool:
    """True when an ``rm`` operand can wipe a protected root or a whole home.

    A single trailing slash is equivalent to none (``rm -rf ~/`` wipes the
    home directory, ``/home/me/`` wipes that user's home). The doubled-slash
    root form ``//`` is left untouched, preserving prior classification.
    """
    candidates = [operand]
    if len(operand) > 1 and operand.endswith("/") and not operand.startswith("//"):
        candidates.append(operand[:-1])
    for candidate in candidates:
        if candidate in _RM_PROTECTED_ROOT_OPERANDS:
            return True
        if candidate in _RM_HOME_CONTENT_GLOBS:
            return True
        if candidate.startswith("/root/") or re.fullmatch(
            r"/home/[^/]+(?:/|/\*)?", candidate
        ):
            return True
    return False


def _rm_references_unresolvable_variable(operand: str) -> bool:
    """True when ``operand`` references a ``$VAR`` other than ``$HOME``."""
    if "$" not in operand:
        return False
    remainder = operand.replace("$HOME", "").replace("${HOME}", "")
    return "$" in remainder


def _rm_is_catastrophic(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if Path(token).name != "rm":
            continue
        operands = [
            item
            for item in tokens[index + 1 :]
            if item == "--" or not item.startswith("-")
        ]
        if operands and operands[0] == "--":
            operands = operands[1:]
        destructive_flags = any(
            item in {"--recursive", "--force"}
            or (item.startswith("-") and ("r" in item or "f" in item))
            for item in tokens[index + 1 :]
        )
        if any(
            _rm_operand_wipes_protected_root(item)
            or (destructive_flags and ".." in item)
            or (destructive_flags and _rm_references_unresolvable_variable(item))
            for item in operands
        ):
            return True
    return False
```

No other `command.py` change in this task (`_CATASTROPHIC`, the git analyzers,
and `classify_command` are untouched).

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_policy.py -q`
Expected: all PASS, including `test_home_subpath_removals_are_not_catastrophic`
and the whole-home guard entries (`/home`, `/home/user/`, `~/`, `/root`,
`${HOME}`).

- [ ] **Step 5: Commit**

```bash
git add tests/test_policy.py src/coding_agent/policy/command.py
git commit -m "Narrow rm catastrophic rule to protected roots and whole homes" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Carve `/dev/null` out of the `/dev` redirection and `dd` rules

**Files:** Modify `tests/test_policy.py` and `src/coding_agent/policy/command.py`.

`> /dev/null`, `2>/dev/null`, `>/dev/null 2>&1`, and `dd ... of=/dev/null`
must no longer be catastrophic; redirection to real block devices (`/dev/sda`,
`/dev/nvme...`) and `dd` to a real device stay catastrophic.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy.py`:

```python
@pytest.mark.parametrize(
    "command",
    [
        "ls > /dev/null",
        "echo hi 2>/dev/null",
        "ls >/dev/null 2>&1",
        "dd if=/dev/zero of=/dev/null bs=1M count=1",
        "dd if=x of=/dev/null",
    ],
)
def test_dev_null_redirection_is_not_catastrophic(command):
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is False
    for mode in ("workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "allow"
        )


@pytest.mark.parametrize(
    "command",
    [
        "echo bad 2>/dev/sda",
        "echo bad >>/dev/nvme0n1",
        "dd if=x of=/dev/sda",
    ],
)
def test_dev_block_device_writes_remain_catastrophic(command):
    assert classify_command(command).catastrophic is True
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_policy.py -q`
Expected: `test_dev_null_redirection_is_not_catastrophic` FAILS (`> /dev/null`
is currently catastrophic). The block-device test PASSES.

- [ ] **Step 3: Implement the `/dev/null` carve-outs**

In `src/coding_agent/policy/command.py`, in `_CATASTROPHIC` (currently lines
~14-24) replace these two patterns:

```python
    r"\bdd\s+.*of=/dev/",
    r">{1,2}\s*/dev/",
```

with:

```python
    r"\bdd\s+.*\bof=/dev/(?!null(?:\s|$))",
    r">{1,2}\s*/dev/(?!null(?:\s|$))",
```

The negative lookahead `(?!null(?:\s|$))` exempts exactly `/dev/null` (and
`/dev/null` followed by end/whitespace); every other `/dev/...` target still
matches.

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_policy.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_policy.py src/coding_agent/policy/command.py
git commit -m "Allow redirection to dev null and dd of dev null" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Classify any `git reset` with a `--hard` flag as catastrophic

**Files:** Modify `tests/test_policy.py` and `src/coding_agent/policy/command.py`.

The contiguous regex `git\s+reset\s+--hard` lets `git reset -q --hard HEAD` and
`git -C . reset --hard` through. Add a token analyzer that reuses the
global-option skipper so any `git reset ... --hard` is catastrophic.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy.py`:

```python
@pytest.mark.parametrize(
    "command",
    [
        "git reset -q --hard HEAD",
        "git -C . reset --hard",
        "git -c x=y reset --hard HEAD",
        "git --git-dir=/tmp/g reset --hard",
        "git --exec-path /tmp reset --hard",
    ],
)
def test_git_reset_hard_with_interleaved_options_is_catastrophic(command):
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


@pytest.mark.parametrize(
    "command",
    [
        "git reset --soft HEAD",
        "git reset -q HEAD",
        "git reset --mixed",
        "git reset HEAD",
        "git -C . reset --soft",
    ],
)
def test_non_hard_git_reset_forms_are_not_catastrophic(command):
    assert classify_command(command).catastrophic is False
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_policy.py -q`
Expected: `test_git_reset_hard_with_interleaved_options_is_catastrophic` FAILS
(`git reset -q --hard HEAD` and `git -C . reset --hard` are currently
non-catastrophic). The negative test PASSES.

- [ ] **Step 3: Implement the reset token analyzer**

In `src/coding_agent/policy/command.py`:

Edit A — rename the shared global-option skipper (now used by push, remote, and
reset). Change the definition:

```python
def _git_push_subcommand_index(tokens: list[str], git_index: int) -> int | None:
```

to:

```python
def _git_subcommand_index(tokens: list[str], git_index: int) -> int | None:
```

and update its two existing call sites: in `_git_push_is_catastrophic`
(`push_index = _git_push_subcommand_index(tokens, index)`) and in
`_git_remote_is_catastrophic`
(`subcommand_index = _git_push_subcommand_index(tokens, index)`) to call
`_git_subcommand_index`.

Edit B — add, immediately after `_git_remote_is_catastrophic`:

```python
def _git_reset_arguments_are_catastrophic(arguments: list[str]) -> bool:
    """True when a ``git reset`` invocation carries the ``--hard`` flag.

    ``--hard`` may appear anywhere among the subcommand's options (``-q``,
    ``--quiet``, etc.). A bare ``--`` ends option parsing, so tokens after it
    are pathspec operands rather than flags.
    """
    for item in arguments:
        if item == "--":
            break
        if item == "--hard":
            return True
    return False


def _git_reset_is_catastrophic(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if Path(token).name != "git":
            continue
        subcommand_index = _git_subcommand_index(tokens, index)
        if subcommand_index is None or tokens[subcommand_index] != "reset":
            continue
        if _git_reset_arguments_are_catastrophic(tokens[subcommand_index + 1 :]):
            return True
    return False
```

Edit C — in `classify_command`, add the analyzer to the token OR-chain (which
currently checks `_rm_is_catastrophic`, `_git_push_is_catastrophic`,
`_git_remote_is_catastrophic`, `_git_config_shell_alias_is_catastrophic`,
`_nested_shell_is_catastrophic`, `_command_substitutions_are_catastrophic`):

```python
    if (
        _rm_is_catastrophic(tokens)
        or _git_push_is_catastrophic(tokens)
        or _git_remote_is_catastrophic(tokens)
        or _git_reset_is_catastrophic(tokens)
        or _git_config_shell_alias_is_catastrophic(tokens)
        or _nested_shell_is_catastrophic(tokens)
        or _command_substitutions_are_catastrophic(command)
    ):
```

The contiguous `git\s+reset\s+--hard` regex stays in `_CATASTROPHIC`.

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_policy.py tests/test_policy_runner_hardening.py -q`
Expected: all PASS (interleaved reset variants catastrophic; soft/mixed/plain
reset and the config-alias/remote/push suites unchanged).

- [ ] **Step 5: Commit**

```bash
git add tests/test_policy.py src/coding_agent/policy/command.py
git commit -m "Classify git reset hard across interleaved options as catastrophic" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add the workspace-bounded `remove_file` tool

**Files:** Modify `tests/test_tools_filesystem.py`,
`src/coding_agent/tools/filesystem.py`, and `src/coding_agent/app.py`.

`remove_file(path)` deletes a single file or an empty directory. It resolves
through `resolve_tool_path` (workspace-bounded, approval `ask` outside in
`default`/`workspace`, `allow` in `full`) and carries
`risk_level="mutate_file"` so the executor journals it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tools_filesystem.py`, extend the filesystem import:

```python
from coding_agent.tools.filesystem import (
    MAX_READ_CHARS,
    make_edit_file_tool,
    make_read_file_tool,
    make_remove_file_tool,
    make_write_file_tool,
)
```

Append:

```python
@pytest.mark.asyncio
async def test_remove_file_deletes_a_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("bye", encoding="utf-8")
    result = await make_remove_file_tool().execute(
        {"path": "a.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert "removed" in result.content
    assert not target.exists()


@pytest.mark.asyncio
async def test_remove_file_deletes_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = await make_remove_file_tool().execute(
        {"path": "empty"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert not empty.exists()


@pytest.mark.asyncio
async def test_remove_file_refuses_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside-remove.txt"
    outside.write_text("secret", encoding="utf-8")
    result = await make_remove_file_tool().execute(
        {"path": "../outside-remove.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="workspace"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "workspace" in (result.error or "")
    assert outside.exists()


@pytest.mark.asyncio
async def test_remove_file_leaves_non_empty_directory_untouched(tmp_path):
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "x.txt").write_text("x", encoding="utf-8")
    result = await make_remove_file_tool().execute(
        {"path": "keep"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert keep.exists()
    assert (keep / "x.txt").exists()
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_tools_filesystem.py -q`
Expected: collection FAILS — `make_remove_file_tool` does not exist yet.

- [ ] **Step 3: Implement the tool**

In `src/coding_agent/tools/filesystem.py`, add the args model next to the
other args models:

```python
class _RemoveFileArgs(BaseModel):
    path: str
```

and add the tool class after `_EditTool`:

```python
class _RemoveFileTool:
    args_model = _RemoveFileArgs
    schema = ToolSchema(
        name="remove_file",
        description="Delete a file or an empty directory",
        parameters=_RemoveFileArgs.model_json_schema(),
        risk_level="mutate_file",
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            path = resolve_tool_path(
                context.workspace,
                args.path,
                permission_mode=context.permission_mode,
                allow_outside_once=context.allow_outside_once,
            )
            if signal.is_set():
                return _result(self.schema.name, False, error="cancelled")
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
            return _result(self.schema.name, True, f"removed {path}")
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))
```

Add the factory next to the other factories:

```python
def make_remove_file_tool():
    return _RemoveFileTool()
```

- [ ] **Step 4: Register the tool**

In `src/coding_agent/app.py`:

Edit A — add `make_remove_file_tool` to the filesystem import tuple so it is
alphabetical:

```python
from coding_agent.tools.filesystem import (
    make_edit_file_tool,
    make_read_file_tool,
    make_remove_file_tool,
    make_write_file_tool,
)
```

Edit B — in `_make_registry`, register it between `edit_file` and `run_command`
(matching the spec's §8 tool order):

```python
    for tool in (
        make_read_file_tool(),
        make_list_files_tool(),
        make_grep_files_tool(),
        make_write_file_tool(),
        make_edit_file_tool(),
        make_remove_file_tool(),
        make_run_command_tool(),
    ):
```

- [ ] **Step 5: Run and verify pass**

Run: `pytest tests/test_tools_filesystem.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_tools_filesystem.py src/coding_agent/tools/filesystem.py src/coding_agent/app.py
git commit -m "Add workspace-bounded remove_file tool" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add the workspace-bounded `clear_directory` tool and verify both are registered

**Files:** Modify `tests/test_tools_filesystem.py`,
`src/coding_agent/tools/filesystem.py`, and `src/coding_agent/app.py`.

`clear_directory(path)` removes every entry under a directory while keeping the
directory itself (recursively, symlinks removed as links). Same
workspace-bounding and `risk_level="mutate_file"` as `remove_file`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tools_filesystem.py`, add `make_clear_directory_tool` to the
import (alphabetical, before `make_edit_file_tool`):

```python
from coding_agent.tools.filesystem import (
    MAX_READ_CHARS,
    make_clear_directory_tool,
    make_edit_file_tool,
    make_read_file_tool,
    make_remove_file_tool,
    make_write_file_tool,
)
```

Append:

```python
@pytest.mark.asyncio
async def test_clear_directory_removes_contents_and_keeps_directory(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.txt").write_text("a", encoding="utf-8")
    nested = cache / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")
    result = await make_clear_directory_tool().execute(
        {"path": "cache"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is True
    assert "cleared" in result.content
    assert cache.is_dir()
    assert list(cache.iterdir()) == []


@pytest.mark.asyncio
async def test_clear_directory_refuses_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside-cache"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    result = await make_clear_directory_tool().execute(
        {"path": "../outside-cache"},
        context=ToolContext(workspace=tmp_path, permission_mode="workspace"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "workspace" in (result.error or "")
    assert (outside / "keep.txt").exists()


@pytest.mark.asyncio
async def test_clear_directory_rejects_a_file_path(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    result = await make_clear_directory_tool().execute(
        {"path": "a.txt"},
        context=ToolContext(workspace=tmp_path, permission_mode="full"),
        signal=asyncio.Event(),
    )
    assert result.ok is False
    assert "not a directory" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "x"


def test_delete_tools_registered_with_app_registry():
    from coding_agent.app import _make_registry

    schemas = {schema.name: schema for schema in _make_registry().schemas()}
    for name in ("remove_file", "clear_directory"):
        assert name in schemas
        assert schemas[name].risk_level == "mutate_file"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_tools_filesystem.py -q`
Expected: collection FAILS — `make_clear_directory_tool` does not exist yet, so
the module import errors (the registration test is not reached).

- [ ] **Step 3: Implement the tool**

In `src/coding_agent/tools/filesystem.py`, add `import shutil` to the imports
and the args model next to the others:

```python
class _ClearDirectoryArgs(BaseModel):
    path: str
```

Add the tool class after `_RemoveFileTool`:

```python
class _ClearDirectoryTool:
    args_model = _ClearDirectoryArgs
    schema = ToolSchema(
        name="clear_directory",
        description="Remove all contents of a directory, keeping the directory",
        parameters=_ClearDirectoryArgs.model_json_schema(),
        risk_level="mutate_file",
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            path = resolve_tool_path(
                context.workspace,
                args.path,
                permission_mode=context.permission_mode,
                allow_outside_once=context.allow_outside_once,
            )
            if signal.is_set():
                return _result(self.schema.name, False, error="cancelled")
            if not path.is_dir():
                return _result(
                    self.schema.name, False, error=f"{path} is not a directory"
                )
            for child in path.iterdir():
                if child.is_symlink() or not child.is_dir():
                    child.unlink()
                else:
                    shutil.rmtree(child)
            return _result(self.schema.name, True, f"cleared {path}")
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))
```

Add the factory next to the others:

```python
def make_clear_directory_tool():
    return _ClearDirectoryTool()
```

- [ ] **Step 4: Register the tool**

In `src/coding_agent/app.py`, add `make_clear_directory_tool` to the filesystem
import (alphabetical, before `make_edit_file_tool`):

```python
from coding_agent.tools.filesystem import (
    make_clear_directory_tool,
    make_edit_file_tool,
    make_read_file_tool,
    make_remove_file_tool,
    make_write_file_tool,
)
```

and register it in `_make_registry` right after `remove_file`:

```python
    for tool in (
        make_read_file_tool(),
        make_list_files_tool(),
        make_grep_files_tool(),
        make_write_file_tool(),
        make_edit_file_tool(),
        make_remove_file_tool(),
        make_clear_directory_tool(),
        make_run_command_tool(),
    ):
```

- [ ] **Step 5: Run and verify pass**

Run: `pytest tests/test_tools_filesystem.py -q`
Expected: all PASS, including
`test_delete_tools_registered_with_app_registry`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_tools_filesystem.py src/coding_agent/tools/filesystem.py src/coding_agent/app.py
git commit -m "Add clear_directory tool and register delete tools" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Embed safety boundaries in the system prompt

**Files:** New `tests/test_app_system_prompt.py`; modify `src/coding_agent/app.py`.

Replace the static `SYSTEM_PROMPT` constant with a
`build_system_prompt(workspace, permission_mode)` builder that states the
hard-denied commands, the sanctioned delete tools, the `/dev/null` guidance,
the active mode, dev-server management, and the workspace root. Wire both the
runner factory and the runtime to the per-run prompt.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app_system_prompt.py`:

```python
from pathlib import Path

from coding_agent.app import build_system_prompt


def _content(workspace: Path, mode: str) -> str:
    return build_system_prompt(workspace, mode).content  # type: ignore[arg-type]


def test_system_prompt_lists_hard_denied_commands(tmp_path):
    content = _content(tmp_path, "default")
    for fragment in (
        "/root",
        "/home/<user>",
        "git push --force",
        "git reset --hard",
        "git clean -f",
        "mkfs",
        "fdisk",
        "shutdown",
        "reboot",
        "poweroff",
    ):
        assert fragment in content


def test_system_prompt_recommends_the_sanctioned_delete_tools(tmp_path):
    content = _content(tmp_path, "default")
    assert "remove_file" in content
    assert "clear_directory" in content


def test_system_prompt_discourages_dev_null_redirection(tmp_path):
    content = _content(tmp_path, "default")
    assert "> /dev/null" in content
    assert "2>&1" in content


def test_system_prompt_states_the_active_permission_mode(tmp_path):
    content = _content(tmp_path, "workspace")
    assert '"workspace"' in content


def test_system_prompt_names_the_workspace_root(tmp_path):
    content = _content(tmp_path, "full")
    assert str(tmp_path) in content


def test_system_prompt_guides_dev_server_management(tmp_path):
    content = _content(tmp_path, "default")
    assert "pgrep" in content
    assert "pkill" in content
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_app_system_prompt.py -q`
Expected: collection FAILS — `coding_agent.app` has no `build_system_prompt`.

- [ ] **Step 3: Implement the builder and rewire `create_app`**

In `src/coding_agent/app.py`:

Edit A — replace the module-level constant:

```python
SYSTEM_PROMPT = Message(
    role="system",
    content=(
        "You are coding-agent, an engineering assistant operating in the user's "
        "workspace. Inspect relevant files before changing them, use the provided "
        "tools for workspace operations, verify changes when practical, and give a "
        "concise final response."
    ),
)
```

with a builder function (kept at the same module location so it can be imported
by tests):

```python
def build_system_prompt(workspace: Path, permission_mode: PermissionMode) -> Message:
    """Return the system prompt embedding the safety boundaries for a run."""
    return Message(
        role="system",
        content=(
            "You are coding-agent, an engineering assistant operating in the user's "
            "workspace. Inspect relevant files before changing them, use the provided "
            "tools for workspace operations, verify changes when practical, and give a "
            "concise final response.\n\n"
            "Permission boundaries\n"
            "- Hard-denied in every mode (never attempt and do not work around): rm "
            "of /, ~, $HOME, /root, or a whole /home/<user>; git push --force; git "
            "reset --hard; git clean -f; mkfs/fdisk/shutdown/reboot/poweroff.\n"
            "- For deletions prefer the workspace-bounded remove_file and "
            "clear_directory tools, and prefer relative in-workspace paths.\n"
            '- Do not silence command output with "> /dev/null" (blocked); if you '
            "must combine stderr, pipe with 2>&1.\n"
            f'- The active permission mode is "{permission_mode}": in "default" '
            "every shell command requires user approval; in \"workspace\"/\"full\" "
            "only the hard-denied commands above are rejected.\n"
            '- A dev server started with "&" persists across tool calls; inspect it '
            "with pgrep and stop it with pkill or kill.\n"
            f"- Workspace root: {workspace}"
        ),
    )
```

Edit B — inside `create_app`, compute the prompt once after the journal is
created (currently `journal = MutationJournal()`):

```python
    system_prompt = build_system_prompt(resolved_workspace, permission_mode)
```

Edit C — replace the two `system_prompt=SYSTEM_PROMPT,` references (in
`runner_factory` and in the `AgentRuntime` construction) with
`system_prompt=system_prompt,`.

No other module references `SYSTEM_PROMPT` (verified by grep), so the constant
can go away entirely.

- [ ] **Step 4: Run and verify pass**

Run: `pytest tests/test_app_system_prompt.py -q`
Expected: all PASS.

Run: `pytest tests/test_cli_provider_hardening.py tests/test_integration_flow.py tests/test_w2_approval_experience.py -q`
Expected: all PASS (create_app still wires a system prompt end-to-end).

- [ ] **Step 5: Commit**

```bash
git add tests/test_app_system_prompt.py src/coding_agent/app.py
git commit -m "Embed permission and tool boundaries in the system prompt" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Verify the full suite, linters, and the safety expectations

**Files:** none (verification only; fix only what a task above missed).

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all PASS. Baseline before this round is 571 passed / 1 skipped; the
new parametrized cases add to that. One timing-sensitive concurrency test
(`test_distinct_path_slow_tools_run_in_parallel_before_same_path_third`) can
flake under a loaded machine — if only it fails, re-run it alone to confirm it
passes; do not change it.

- [ ] **Step 2: Run the linters**

Run: `ruff check src tests`
Run: `ruff format --check src tests`
Expected: both clean.

- [ ] **Step 3: Verify the real `classify_command` outcomes**

Run this and confirm every line prints PASS:

```python
from coding_agent.policy.command import classify_command

DENIED = [
    "rm -rf /",
    "rm -rf /home/user",
    "rm -rf /root/.ssh",
    "rm -rf ~/*",
    "rm -rf $HOME",
    'rm -rf "$PWD/../sibling"',
    'rm -rf "$ROOT"',
    "git push --force origin main",
    "git reset -q --hard HEAD",
    "git -C . reset --hard",
    "git clean -xdf",
    "mkfs.ext4 /dev/sda",
    "echo bad >/dev/sda",
    "dd if=x of=/dev/sda",
]
ALLOWED = [
    "rm -rf ~/projects",
    "rm -rf ${HOME}/cache",
    "rm -rf /home/me/.cache/foo",
    "rm -f /home/me/build/app.js",
    "ls > /dev/null",
    "echo hi 2>/dev/null",
    "ls >/dev/null 2>&1",
    "dd if=/dev/zero of=/dev/null bs=1M count=1",
]
for command in DENIED:
    assert classify_command(command).catastrophic is True, command
for command in ALLOWED:
    assert classify_command(command).catastrophic is False, command
print("PASS")
```

- [ ] **Step 4: Commit (only if a fix was needed)**

If any step above forced a source/test change not yet committed, commit it
with a descriptive message plus the standard trailer.

---

## Self-Review

- A1 covered: `rm` scoped to protected roots/whole homes; home subpaths
  mode-governed; guard tests pin `/home`, `/home/user/`, `~/`, `/root`,
  `${HOME}`, `~/*`, `$HOME/*`, `${HOME}/*`, `/home/me`. ✅ (Task 1)
- A2 covered: `/dev/null` carve-out for redirection and `dd`; real block
  devices still catastrophic. ✅ (Task 2)
- A3 covered: token analyzer + shared `_git_subcommand_index` catches
  `-q`/`-C`/`-c`/`--git-dir`/`--exec-path` variants; soft/mixed/plain reset and
  `--`-prefixed pathspecs stay allowed; the contiguous regex remains. ✅ (Task 3)
- A4 reconciled: every listed expectation is asserted against the real
  `classify_command` in Task 1-3 tests and re-verified in Task 7. ✅
- B covered: `remove_file`/`clear_directory` workspace-bounded via
  `resolve_tool_path`, `risk_level="mutate_file"`, registered in `_make_registry`
  (asserted by a test), not routed through `run_command`. ✅ (Tasks 4-5)
- C covered: `build_system_prompt` embeds hard-denied list, delete-tool
  guidance, `/dev/null` guidance, active mode, dev-server management, workspace
  root; wired into both the runner factory and the runtime. ✅ (Task 6)
- E covered: full `pytest`, `ruff check`, `ruff format --check`, and a direct
  `classify_command` pass over the still-denied and newly-allowed sets. ✅
  (Task 7)
- No safety regression: the four still-denied categories plus true system
  commands remain denied in every mode (guards + Task 7 Step 3). ✅
