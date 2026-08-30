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
- Six local tools:
  - `read_file`
  - `list_files`
  - `grep_files`
  - `write_file`
  - `edit_file`
  - `run_command`
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
transcript and context policy, and starts with `default` permission.
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

Default `max_steps` is 20 and is configurable.

## 8. Tools and Execution

Tool contracts:

```text
read_file(path, start_line=1, end_line=None)
list_files(path=".", recursive=False, max_entries=200)
grep_files(pattern, path=".", include=None, max_results=100)
write_file(path, content)
edit_file(path, old_text, new_text)
run_command(command, timeout_seconds=120)
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
- `run_command` uses the workspace as cwd, captures stdout/stderr, enforces a
  timeout, and applies output limits.
- Tool errors become `ToolResult(ok=False)` and are returned to the model;
  ordinary tool failures do not crash the AgentRunner.
- MVP executes calls sequentially. `is_parallel_safe` remains in the schema
  for a later read-only parallel executor.

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
simple workspace-safe shell: allow
outside explicit path: ask; approve grants this call only
outside-or-unknown shell: ask
catastrophic command: deny
```

Shell classification is conservative. Absolute paths, `..`, `cd`, `git -C`,
external redirection, command substitution, unknown scripts, and inline
file-writing code are `outside-or-unknown`; `default` and `workspace` ask for
one-time approval before executing them. This is an application policy, not an
OS sandbox; shell still runs with the user's process permissions.

### `full`

```text
ordinary tools: allow without approval
workspace-outside paths: allow
ordinary shell: allow without approval
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
rm\s+-rf\s+(/|/\*|~)
\b(mkfs|fdisk|shutdown|reboot|poweroff)\b
git\s+push\b.*--force(?:-with-lease)?\b
git\s+reset\s+--hard\b
git\s+clean\s+-[^\n]*f[^\n]*d
\bdd\b[^\n]*\bof=/dev/
>\s*/dev/(sd|hd|nvme|vd)[a-z0-9]*
:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;
```

Unknown or unclassifiable shell syntax is `ask` in `default` and
`workspace`; it is `allow` in `full` unless it matches a catastrophic rule.
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
grouping; the context policy always preserves it as the first message.

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
`default` permission and no active process, approval, or tool execution.
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
- tool running/success/error/cancelled status;
- approval modal;
- `/session` selector using a modal option list;
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
3. Skill loader from `.agents/skills/*/SKILL.md`.
4. Read-only parallel tool execution using `is_parallel_safe`.
5. Pi-style LLM SummaryPolicy and externalized tool results.
6. Session tree operations (`rewind`, `fork`) and branch summaries.
7. Background tasks, richer TUI panels, and stronger sandbox backends.
