# Coding Agent MVP Design

## Status

Design approved through brainstorming and passed an independent adversarial
review after resolving interface and permission-boundary findings.

## 1. Goal

Build a Python coding agent for the assignment in `ASSIGNMENT_NOTICE.md`.
The product is an interactive terminal UI (TUI) that can hold a multi-turn
conversation, inspect and modify a local workspace, run commands, show live
agent/tool state, request approval for side effects, and resume persisted
sessions.

The implementation must own the important agent logic: message history,
context handling, tool definitions and local execution, model-output parsing,
termination conditions, and error handling. It may use a model vendor client
library and a TUI library, but it must not use an Agent framework or Agent SDK.

The primary architectural references are:

- Pi: thin agent loop, harness separation, streaming events, session/context
  boundaries, and interactive product layering.
- `my-pi-agent`: Python/Pydantic implementation, tool registry, JSONL session,
  context handling, path checks, and Never-Throw tool results.
- Deer-Flow: Textual TUI, runtime bridge, immutable view state, and pure
  reducer.
- oh-my-cli/Codex: approval policy and dangerous-command boundaries.

DeepSeek Harness is an architectural reference only for capability boundaries
and persistent-versus-live event separation. Its Cordis runtime, full plugin
tree, profile/bundle system, and scope machinery are out of MVP scope.

## 2. MVP Scope

### Required

- Python implementation in one package.
- Interactive Textual TUI.
- Multi-turn conversation in one active session.
- One active run at a time.
- OpenAI-compatible Chat Completions provider behind an `LLMProvider`
  interface.
- Async streaming from provider to runtime and live assistant text rendering.
- Eight local tools:
  - `read_file`
  - `list_files`
  - `grep_files`
  - `write_file`
  - `edit_file`
  - `run_command`
  - `remove_file`
  - `clear_directory`
- Pydantic runtime models and boundary validation.
- JSONL session persistence and resume.
- A replaceable `ContextPolicy`; MVP implementation is deterministic
  turn-aligned truncation.
- Three permission modes: `default`, `workspace`, and `full`.
- Mode-aware path policy, shell timeout, output limits, and a non-bypassable
  catastrophic-command denylist.
- Runtime events consumed by a TUI reducer.
- Local slash commands and session selector.
- Fake-provider unit tests and temporary-workspace integration tests.

### Explicitly out of MVP

- Web, desktop, remote, ACP, or SDK server entrypoints.
- OS-level sandboxing, network isolation, or credential isolation.
- Full multi-provider protocol support.
- Parallel tool execution (the tool contract retains `is_parallel_safe`).
- Steering/follow-up queues and multiple concurrent turns.
- Session tree navigation (`rewind`, `fork`) and multi-lane sessions.
- Dynamic plugins, skills, MCP, and subagents in the core MVP. Interfaces must
  not prevent later addition; MCP is the first planned post-MVP increment and
  Subagent S1 (foreground delegation) follows it.
- Theme system, mouse interaction, Vim mode, complex overlays, or media
  rendering.

## 3. Architecture

```text
Textual TUI
  -> AgentRuntime (public session-scoped runtime)
       -> AgentRunner (turn/step agent loop)
            -> LLMProvider
            -> ContextPolicy
            -> ToolExecutor -> ToolRegistry
            -> ApprovalPolicy
            -> SessionStore
            -> RuntimeEvent/EventSink
  -> TuiReducer -> TuiState -> Textual widgets
```

### Ownership rules

- TUI renders state and sends commands. It never calls a tool, provider, or
  session writer directly.
- `AgentRuntime` owns the current session, active run, cancellation state,
  approval waiters, current policy, and event subscribers. It is the single
  run coordinator: it appends `turn_start` and `user_message`, then schedules
  `AgentRunner`; the runner appends the ordered assistant/tool/turn records
  through the same `SessionStore` instance. No other component writes JSONL.
- `AgentRunner` owns the model/tool loop and turn termination.
- `LLMProvider` owns provider request/stream conversion only.
- `ToolRegistry` owns tool discovery, schemas, and lookup.
- `ToolExecutor` is the only path for argument validation, permission,
  timeout, cancellation, error conversion, hooks, and output truncation.
- `ContextPolicy` creates a model-facing view without mutating the session.
- `SessionStore` is the only persistence implementation and serializes all
  appends through one controlled append method/lock. During a run,
  `AgentRuntime` is the only coordinator allowed to invoke it directly or
  through `AgentRunner`; append order is the run's execution order and there is
  never a second active writer.
- `RuntimeEvent` is live UI/control data; `SessionRecord` is durable history.

## 4. Package Layout

```text
src/coding_agent/
├── app.py
├── runtime/
│   ├── runtime.py       # AgentRuntime public facade
│   ├── runner.py        # AgentRunner turn/step loop
│   ├── models.py        # runtime outcomes and status
│   └── events.py        # RuntimeEvent and EventSink
├── llm/
│   ├── protocol.py      # LLMProvider and LLMEvent contracts
│   └── openai_compatible.py
├── tools/
│   ├── models.py        # Tool, ToolSchema, ToolResult
│   ├── registry.py
│   ├── executor.py
│   ├── filesystem.py    # read/write/edit
│   ├── search.py        # list/grep
│   └── shell.py
├── session/
│   ├── models.py        # SessionRecord and projection models
│   └── store.py
├── context/
│   ├── policy.py        # ContextPolicy and ContextView
│   └── truncate.py
├── policy/
│   ├── approval.py
│   └── command.py
└── tui/
    ├── app.py
    ├── state.py
    ├── reducer.py
    └── widgets.py
```

## 5. Data Models

All durable and boundary models use Pydantic. Provider-specific wire shapes
are converted at the provider boundary and do not leak into the runtime.

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    ok: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

`ToolCall.arguments` is an internal dictionary. Chat Completions function
arguments arrive as a JSON string, are parsed by the provider/runtime boundary,
and are never executed before parsing and schema validation succeed.

```python
class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    risk_level: Literal["read", "mutate_file", "mutate_shell", "dangerous"]
    is_parallel_safe: bool = False
```

The runtime receives one configured system prompt (or a synchronous/async
provider that returns it). It is injected as the first message of every
`ContextView`; it is not reconstructed from user text. A resume uses the
current application prompt configuration, while the session keeps only the
conversation facts and a prompt version/hash in optional metadata.

```python
class SessionSummary(BaseModel):
    id: str
    workspace: str
    created_at: datetime
    updated_at: datetime
    title: str
    last_status: str

class ContextView(BaseModel):
    messages: list[Message]
    used_tokens: int
    context_window: int
    estimated: bool
    compacted: bool
    removed_turns: int = 0
    overflow: bool = False

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

When a provider does not report a reasoning level or context window, the
runtime exposes `None` and the TUI renders `-` (or an explicitly configured
fallback window). It must not invent a precise provider value.

## 6. Provider and Streaming

For MVP, `CancellationToken` is an `asyncio.Event` passed by the runtime;
`set()` requests cancellation and `is_set()` is checked at await/output
boundaries. A later implementation may replace this alias with a richer token
without changing provider ownership.

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

The normalized provider events are:

```text
text_delta
tool_call_start
tool_call_delta
tool_call_end
response_end
error
```

`delta` is an incremental response fragment. Text fragments are appended to
the current assistant buffer. Tool-call argument fragments are concatenated by
`tool_call_id` and parsed only after the response ends. A partial or
`finish_reason=length` tool call is never executed.

Streaming rules:

- `LLMEvent` is provider-to-runtime data.
- `RuntimeEvent(assistant_delta)` is runtime-to-TUI live data.
- Only the complete assistant message is persisted as a normal model message.
- Partial text may remain visible in the TUI after an error, but it is not used
  as a complete resumable model message.

## 7. Agent Runtime and Lifecycle

```python
class AgentRuntime:
    async def submit(self, prompt: str) -> str: ...
    async def new_session(self) -> str: ...
    async def list_sessions(self) -> list[SessionSummary]: ...
    async def abort(self, run_id: str) -> None: ...
    async def resolve_approval(
        self, request_id: str, decision: Literal["approve", "deny"]
    ) -> None: ...
    async def resume(self, session_id: str) -> None: ...
    async def compact(self) -> None: ...
    async def set_permission(
        self, mode: Literal["default", "workspace", "full"]
    ) -> None: ...
    def subscribe(self, sink: EventSink) -> Callable[[], None]: ...
```

The constructor creates a new session when no session is supplied. `new_session`
closes the current idle view, creates a fresh session, resets the visible
transcript and context policy, and starts with `workspace` permission.
`list_sessions` returns newest-updated sessions first. A session selector uses
this method; `resume` accepts a full id or a unique id prefix and rejects an
ambiguous prefix.

`submit()` creates an `asyncio.Task` and returns a `run_id` without blocking the
caller; it rejects a second active run. A turn consists of one or more steps;
each step is one model response plus its tool calls/results.

```text
submit
  -> run_started
  -> turn_start + user_message
  -> repeat:
       ContextPolicy.prepare
       LLMProvider.stream
       assistant aggregation
       no tool calls -> completed
       otherwise sequential tool execution and result append
  -> turn_end
  -> run_finished (with outcome=completed|aborted|max_steps) or run_error
```

The runner persists a `tool_call` audit record before invoking a tool. If that
append fails, the tool is not invoked. It persists the corresponding
`tool_result` before requesting the next model step; a result-append failure
ends the run with `session_error` rather than risking an unrecorded side
effect.

Termination reasons:

```text
completed
max_steps
aborted
provider_error
session_error
```

`max_steps` is now an optional per-turn step cap that defaults to *unbounded*
(`None`): the run loop keeps stepping until the model concludes, the run is
aborted/fails, or (with a configured int cap) the budget is exhausted. When a
configured cap is reached the runner first emits a warning `notice` and then
returns `reason="max_steps"` — the cap is never silent. This amends the earlier
"default 20" wording; the authoritative contract is the superseding spec
`2026-09-02-coding-agent-unbounded-max-steps.md`.

## 8. Tools and Execution

Tool contracts:

```text
read_file(path, start_line=1, end_line=None)
list_files(path=".", recursive=False, max_entries=200)
grep_files(pattern, path=".", include=None, max_results=100)
write_file(path, content)
edit_file(path, old_text, new_text)
remove_file(path)
clear_directory(path)
run_command(command, timeout_seconds=120)
load_skill(skill)
```

Rules:

- File paths are resolved once by `ToolExecutor` with the active permission
  mode. `default` and `workspace` classify an outside path as requiring a
  one-time approval; an approved call may use that outside path, while a
  denied call fails. `full` permits an absolute or outside path without this
  prompt, subject to the catastrophic-command policy and ordinary OS
  permissions.
- `read_file` uses 1-based inclusive line numbers.
- `list_files` and `grep_files` skip `.git`, `node_modules`, virtualenvs,
  build outputs, and cache directories by default.
- `edit_file` succeeds only when `old_text` occurs exactly once.
- `write_file` and `edit_file` use same-directory temp file + flush/fsync +
  atomic replace.
- `remove_file` deletes a single file or empty directory; `clear_directory`
  removes a directory's contents while keeping the directory itself. Both
  return a short confirmation and carry `risk_level="mutate_file"`. Their paths
  resolve through the same permission-aware path policy and approval rules as
  `write_file` and `edit_file`, so an outside-workspace path requires the same
  one-time approval in `default`/`workspace` and is allowed in `full`. They are
  the sanctioned delete tools for in-workspace cleanup instead of raw `rm`, and
  being `mutate_file` calls they are mutation-journaled like other file
  mutations, so a removed file's prior content remains restorable (a
  `clear_directory` snapshot is best-effort).
- `run_command` uses the workspace as cwd, captures stdout/stderr, enforces a
  timeout, and applies output limits.
- Tool errors become `ToolResult(ok=False)` and are returned to the model;
  ordinary tool failures do not crash the AgentRunner.
- MVP executes calls sequentially. `is_parallel_safe` remains in the schema
  for a later read-only parallel executor.
- `load_skill(skill)` (skills feature, 2026-09-02) returns the Markdown body of
  a discovered skill by effective name from the two skills roots (workspace then
  user-global, first wins); an unknown or unsafe name returns `ok=False` with
  `unknown skill: <name>`. Content comes only from a resolved `SKILL.md` under
  those roots and is bounded (truncated past ~16,000 chars with a note). It is a
  read tool with no `path` argument, so it is allowed in every permission mode
  without approval. Full semantics in
  `2026-09-02-coding-agent-skills-support.md`.

## 9. Permission and Safety

`ApprovalPolicy` returns `allow`, `ask`, or `deny`.

```python
class ApprovalRequest(BaseModel):
    request_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: str
    reason: str
    status: Literal["pending", "approved", "denied", "cancelled"] = "pending"
```

### `default`

```text
read/list/grep: allow
write/edit: ask
shell: ask
outside workspace: ask; approve grants this call only
catastrophic command: deny
```

### `workspace`

```text
workspace read/write/edit: allow
shell (including compound and unknown syntax): allow
outside explicit path: ask; approve grants this call only
catastrophic command: deny
```

Shell classification is conservative but informational: `&&`, `|`, `;`,
absolute paths, `..`, `cd`, `git -C`, redirection, command substitution,
unknown scripts, and inline file-writing code are flagged `outside-or-unknown`,
but `workspace` runs them without approval — only the catastrophic denylist
gates shell here. `default` still asks for every shell command. This is an
application policy, not an OS sandbox; shell still runs with the user's process
permissions.

### `full`

```text
ordinary tools: allow without approval
workspace-outside paths: allow
shell (including compound and unknown syntax): allow without approval
catastrophic command: deny, non-bypassable
```

The workspace boundary is therefore mode-aware: `default`/`workspace` enforce
containment by default and require explicit per-call approval to override it;
`full` intentionally bypasses the prompt. This is an application policy, not an
OS sandbox.

Entering `full` requires a visible high-risk confirmation. Permission changes
are local commands, take effect on the next tool call, and append a
`policy_changed` record. Resume always starts in `default`; historical policy
records remain auditable.

Initial catastrophic patterns are deterministic regular-expression/token rules,
including:

```text
\b(mkfs|fdisk|shutdown|reboot|poweroff)\b
git\s+push\b.*--force(?:-with-lease)?\b
git\s+reset\s+--hard\b
git\s+clean\s+-[^\n]*f[^\n]*d
\bdd\b[^\n]*\bof=/dev/
>\s*/dev/(sd|hd|nvme|vd)[a-z0-9]*
:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;
```

The `rm` rule is workspace-and-root scoped: an operand is catastrophic (denied
in every mode) only when it can wipe a protected root or a whole home
directory — exactly `/`, `~`, `$HOME`, `${HOME}`, or `/root`; a whole user home
(`/home/<name>` or `/home/<name>/*` where `<name>` is a single path component)
or its contents via a glob (`~/*`, `$HOME/*`, `${HOME}/*`); or anything under
`/root/` (e.g. `/root/.ssh`). With destructive flags (`-r`/`-f`), an operand
containing `..` or an unresolvable `$VAR` other than `$HOME`/`${HOME}` is also
catastrophic. Home subpaths — `/home/<name>/<anything deeper>`, `~/<sub>`,
`$HOME/<sub>`, `${HOME}/<sub>` — are NOT catastrophic: mode governs them
(`default` asks, `workspace`/`full` allow). `git reset --hard` stays
catastrophic even when global options (`-q`, `-C <dir>`, `-c key=val`,
`--git-dir`) separate `git` from `--hard`; a token analyzer skips the options.

Redirection to `/dev/null` is allowed: `> /dev/null`, `2>/dev/null`,
`> /dev/null 2>&1`, and `dd ... of=/dev/null` are not catastrophic. The
`\bdd\b[^\n]*\bof=/dev/` and `>\s*/dev/(sd|hd|nvme|vd)[a-z0-9]*` rules above
express the intent for real block devices only (`/dev/sd*`, `/dev/hd*`,
`/dev/nvme*`, `/dev/vd*`).

For in-workspace cleanup the model should use the workspace-bounded
`remove_file` and `clear_directory` delete tools (§8) rather than raw `rm`;
`rm` of a permitted home/workspace subpath remains available where the active
mode allows it.

Unknown or unclassifiable shell syntax is `ask` in `default`; it is `allow`
in `workspace` and `full` unless it matches a catastrophic rule.
Download-and-execute pipelines such as `curl ... | sh` are not catastrophic;
they follow the same mode rule: allow in `workspace`/`full`, ask in `default`.
The denylist is a minimum safety net, has known false negatives, and is not a
complete shell parser or OS sandbox.

Approval flow:

```text
ToolExecutor
  -> ApprovalRequest(pending)
  -> RuntimeEvent(approval_requested)
  -> TUI modal
  -> resolve_approval()
  -> execute or error ToolResult
```

The MVP cancellation token is an `asyncio.Event`: `set()` requests
cancellation, and providers/tools check `is_set()` at await and output
boundaries. `AgentRuntime.abort()` closes the provider async iterator when
possible, marks pending approvals cancelled, and asks an active shell process
to terminate its process group (SIGTERM, then SIGKILL after a short grace
period). Cleanup is awaited before the run is reported finished. A tool that
already produced a side effect is recorded as a normal result or cancellation
error; it is never silently retried.

`resolve_approval()` accepts only a currently pending request belonging to the
active run. Unknown, already-resolved, or cancelled request ids return a
structured runtime error; late decisions after abort are ignored and do not
restart a run. The MVP creates at most one pending approval at a time because
tool calls execute sequentially. Non-TTY approval defaults to deny.

## 10. Session and Context

Session files are JSONL. The first implementation is linear but reserves tree
fields:

```python
class SessionRecord(BaseModel):
    id: str
    seq: int
    timestamp: datetime
    type: Literal[
        "turn_start", "user_message", "assistant_message", "tool_call",
        "tool_result", "turn_end", "compaction", "policy_changed",
        "run_error", "interrupted",
    ]
    parent_id: str | None = None
    run_id: str | None = None
    turn_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
```

Every new record points to the previous active record. Future `rewind` moves
the active leaf without deleting records; future `fork` appends a new child
with the selected record as parent.

Model context uses a projection that retains boundary information:

```python
class SessionMessage(BaseModel):
    record_id: str
    turn_id: str
    seq: int
    message: Message
```

The model projection includes `user_message`, complete `assistant_message`,
and `tool_result` records. The separate `tool_call` record is an audit record
for the call already embedded in the assistant message; it is not projected as
another model message. This prevents one call from being sent to the provider
twice while retaining an explicit execution audit trail.

Payload contracts are:

```text
user_message:
    {"message": Message}
assistant_message:
    {"message": Message, "complete": true}
tool_call:
    {"tool_call": ToolCall, "source_assistant_record_id": str}
tool_result:
    {"result": ToolResult}
```

Projection validates that every projected `tool_result.tool_call_id` refers to
one tool call in the preceding assistant message, that a call has at most one
result, and that an incomplete assistant/tool pair is excluded from resumable
context rather than replayed. The independent `tool_call` audit record is
never projected.

The configured system prompt is prepended by `AgentRunner` after the history
projection. It is not assigned a `turn_id` and is not included in turn
grouping; the context policy always preserves it as the first message. Its body
includes a deterministic `## Available skills` catalog (skills feature,
2026-09-02): one `- name: description` line per discovered skill or nothing when
none are installed; full SKILL.md bodies are never auto-injected. See
`2026-09-02-coding-agent-skills-support.md`.

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
```

MVP `TruncatePolicy`:

```text
under budget -> copy full history
over budget  -> retain system prompt
             -> remove oldest complete turns
             -> retain newest complete turns
             -> insert system compact marker
             -> leave SessionStore unchanged
```

`force=True` uses the same turn-aligned algorithm even when the current view is
under budget. If there is no removable complete turn, it returns an unchanged
view and reports `compacted=False`; otherwise it reports `compacted=True`.
The current active turn is never removed. If a single current turn alone is
larger than the budget, tool outputs are bounded first and the policy keeps the
entire current turn as the minimum executable context, returning
`overflow=True`. The runner may make one request with that view; a provider
context-overflow response becomes `provider_error` and is not retried with the
same unchanged view.

Tool output is bounded before context preparation. Provider usage is preferred
for statusline accounting; local `serialized_chars / 4` is the fallback and is
marked as estimated. `/compact` uses the same deterministic policy with
`force=True`, writes a `compaction` record containing strategy and boundary
metadata, and never deletes the original transcript. The record payload
contains `strategy`, `removed_turn_ids`, `retained_turn_ids`,
`tokens_before`, `tokens_after`, and `forced`; no message content is replaced
in the durable log.

Resume restores workspace, model, and projected messages, but starts with
`workspace` permission and no active process, approval, or tool execution.
An open final turn is marked interrupted and is not replayed.

## 11. Runtime Events and TUI

Runtime events:

```text
run_started
run_finished
run_error
user_message
assistant_started
assistant_delta
assistant_finished
tool_started
tool_finished
approval_requested
approval_resolved
context_updated
session_loaded
policy_changed
notice
```

The event envelope and the rendered state are Pydantic models:

```python
class RuntimeEvent(BaseModel):
    event_id: str
    timestamp: datetime
    type: Literal[
        "run_started", "run_finished", "run_error", "user_message",
        "assistant_started", "assistant_delta", "assistant_finished",
        "tool_started", "tool_finished", "approval_requested",
        "approval_resolved", "context_updated", "session_loaded",
        "policy_changed", "notice",
    ]
    run_id: str | None = None
    turn_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

class EventSink(Protocol):
    async def __call__(self, event: RuntimeEvent) -> None: ...

class TranscriptItem(BaseModel):
    kind: Literal["user", "assistant", "tool", "system"]
    item_id: str
    text: str = ""
    pending: bool = False
    started_at: float | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_status: Literal["running", "success", "error", "cancelled"] | None = None

class TuiState(BaseModel):
    session_id: str | None = None
    workspace: str
    git_branch: str | None = None
    model: str
    reasoning: str | None = None
    context_used: int = 0
    context_window: int = 0
    context_estimated: bool = False
    policy: Literal["default", "workspace", "full"] = "default"
    status: Literal[
        "idle", "running", "waiting_approval", "error", "aborted"
    ] = "idle"
    active_run_id: str | None = None
    active_turn_id: str | None = None
    transcript: list[TranscriptItem] = Field(default_factory=list)
    active_tool_call_id: str | None = None
    pending_approval: ApprovalRequest | None = None
    input_text: str = ""
```

Event payload rules are fixed for the reducer: `assistant_delta` carries a
`message_id` and text; `tool_started`/`tool_finished` carry a
`tool_call_id`; `approval_requested` carries a complete `ApprovalRequest`;
`approval_resolved` carries its id, decision, and resulting status;
`context_updated` carries `used_tokens`, `context_window`, and `estimated`;
`run_finished` carries an outcome (`completed`, `aborted`, or `max_steps`).
The reducer merges assistant deltas by message id and updates tool rows by
tool-call id; it never appends a new row for every text delta.

The remaining payloads have these minimum fields:

```text
run_started:       {session_id, model, policy}
run_error:         {code, message, recoverable}
user_message:      {message_id, text}
assistant_started: {message_id}
assistant_finished:{message_id, usage?, finish_reason?}
tool_started:      {tool_call_id, tool_name, arguments}
tool_finished:     {tool_call_id, tool_name, ok, content, error?}
session_loaded:    {session_id, workspace, model}
policy_changed:    {policy, previous_policy}
notice:            {level, message}
```

The TUI keeps one Pydantic `TuiState` snapshot and applies a pure reducer:

```text
RuntimeEvent -> reduce(TuiState, event) -> new TuiState -> render widgets
```

Layout:

```text
Scrollable Transcript
Input TextArea
Bottom Statusline
```

Required widgets:

- transcript with user, assistant, tool, and system rows;
- assistant streaming updates by message id;
- pending assistant rows (no first token yet) render an animated
  `thinking…` placeholder (spinner frame + elapsed seconds);
- tool running/success/error/cancelled status;
- approval modal;
- `/session` selector: a modal option list with a live search box (matching
  session id, title, or workspace), an autofocused list so ↑/↓ + Enter work
  without a click, and a keybinding hint (`type to search · ↑↓ select ·
  Enter switch · Esc cancel`);
- slash command palette shown while input starts with `/`;
- statusline.

Statusline minimum fields:

```text
model
reasoning level
context used / remaining
context window when known
input/output usage when known
workspace
git branch or -
session short id
permission policy
runtime status
```

On narrow terminals, low-priority fields are hidden instead of wrapping the
statusline. The TUI does not require a theme system for MVP.

LLM-wait indicator: `assistant_started` opens an assistant row with
`pending=True` and a `started_at` anchor. Until the first `assistant_delta`
arrives, that row renders an animated placeholder using the current
`SPINNER_FRAMES` frame, the label `thinking…`, and an elapsed counter that
increments each second, e.g. `⠹ thinking… (3s)`. The first `assistant_delta`
clears `pending` and the placeholder is replaced by streamed text; an
`assistant_finished` without any delta also clears it. The placeholder lives
in the transcript row so the user sees that a reply is pending without
watching the statusline; the statusline spinner keeps running as today.

Exit confirmation: `ctrl+c` never exits on the first press. While idle, the
first `ctrl+c` clears any composer draft (when present) and arms an exit
confirmation with a `Press ctrl+c again to exit` notice; a second `ctrl+c`
performs the shutdown. Editing the composer draft, or starting a new run,
disarms the confirmation. While a run is active, `ctrl+c` aborts it (or
dismisses a pending approval); shutdown still waits for the abort to settle.

History backtracking (fork): while idle with an empty composer, pressing
`esc` twice within 800 ms opens a rewind picker listing the user-authored
messages of the current session. Up/down selects a message; `enter` forks the
session at that message: `AgentRuntime.fork_at(message_id)` creates a new
session whose persisted records end at the selected user message, swaps the
runtime onto it, and returns the prompt text. The TUI refills the composer
with that prompt (nothing auto-submits); editing and resubmitting continues
from the fork point. The original session is untouched. A fork is rejected
while a run is active or an approval is pending; the selection is re-validated
against the persisted records and, on failure, the prompt is refilled with an
error notice.

Transcript rendering is incremental: `TranscriptView.render_state` keeps one
widget per transcript row keyed by item id. On each state snapshot it updates
only the rows whose content changed — replacing the row's renderable in place
and refreshing that one widget — and mounts only newly appended rows; settled
rows are never re-created. A materially changed row set (e.g. a session
switch) falls back to a full re-render. Tool rows render compact labels: a
bounded command header that never embeds write/edit payloads (`content`,
`old_text`, `new_text` are omitted), a collapsed first-line preview, and a
truncated body only when the row is expanded.

Commands are local and do not enter model history:

```text
/help
/new
/session
/resume <id>
/compact
/context
/permission
/permission default
/permission workspace
/permission full
/skills
/clear
/quit
```

## 12. Testing and Demonstration

### Automated tests

- Fake provider stream aggregation and tool-call parsing.
- Normal completion, multiple steps, `max_steps`, provider failure, abort,
  and incomplete tool-call rejection.
- Tool schema errors, unknown tools, unique edit matching, path traversal,
  shell timeout, output truncation, and Never-Throw behavior.
- All three permission modes, approval allow/deny/cancel, catastrophic deny,
  and policy switching.
- JSONL append/load/resume, malformed final line, interrupted turn, and
  permission reset on resume.
- Context under budget, over budget, complete-turn truncation, tool/result
  pairing, compact record, and provider-usage fallback.
- RuntimeEvent reducer transitions for streaming assistant/tool rows,
  approvals, errors, session load, and statusline state.
- Textual Pilot tests for input submission, Ctrl-C, command palette,
  approval modal, and session selector.
- Temporary workspace integration tests for all built-in tools.

### Manual video acceptance

Demonstrate a real small project, such as a dependency-light web page or game:

```text
create or inspect project
  -> read/list/grep
  -> write/edit files
  -> run test or browser-independent verification command
  -> observe an error or requested change
  -> submit a follow-up turn
  -> Agent updates files and verifies again
```

The video should show the TUI transcript, live assistant/tool state, statusline,
approval flow where relevant, file changes, and final result. Real model calls
are for manual acceptance only; automated tests use Fake Provider.

## 13. Assignment Compliance Checklist

Non-code delivery must also satisfy the assignment notice:

- Create a new public Git repository after the assignment was issued.
- Preserve the complete Git history; do not squash or rewrite pushed history.
- Provide `README.txt` with the repository URL, run instructions, feature
  summary, and other notes within 1,000 Chinese characters.
- Keep API keys in environment variables or uncommitted local configuration;
  never commit them or include them in the README/video.
- Record a real task demonstration in an MP4 video of at most two minutes and
  200 MB.
- Prepare an English project introduction of at most one minute.
- Preserve double-blind anonymity: do not reveal name, undergraduate school,
  or other identifying information in repository materials, video, or
  interview.
- Submit only the required named ZIP containing `README.txt` and the video by
  the deadline in the assignment notice.

## 14. Post-MVP Roadmap

1. MCP stdio extension: `.mcp.json`, `initialize`, `tools/list`, `tools/call`,
   external tools adapted into ToolRegistry.
2. Subagent S1: foreground `task` tool, isolated child Session, filtered tools,
   reused AgentRunner, final summary returned to parent.
3. Skill loader from `.agents/skills/*/SKILL.md`. Built on 2026-09-02 as a
   two-root (workspace + user-global) mechanism under
   `.coding-agent/skills/*/SKILL.md`; see
   `2026-09-02-coding-agent-skills-support.md`.
4. Read-only parallel tool execution using `is_parallel_safe`.
5. Pi-style LLM SummaryPolicy and externalized tool results.
6. Session tree operations (`rewind`, `fork`) and branch summaries.
7. Background tasks, richer TUI panels, and stronger sandbox backends.
