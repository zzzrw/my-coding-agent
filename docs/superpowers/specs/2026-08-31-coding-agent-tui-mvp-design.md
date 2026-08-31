# Coding Agent TUI MVP Design

## Status

Approved during brainstorming on 2026-08-31. This design covers Tasks 13-15 of the MVP implementation plan.

## 1. Goal and Scope

Complete the runnable interactive MVP on top of the existing runtime, provider,
tools, session, policy, context, runner, and immutable TUI reducer layers.

The deliverable includes:

- A Textual shell with a single transcript as the primary surface.
- A composer and a fixed, compact bottom statusline.
- Runtime event delivery through a bounded asynchronous bridge and reducer.
- Approval, abort, and runtime-error interactions.
- Local slash commands and a session selector.
- CLI configuration and dependency wiring for the `coding-agent` executable.
- Fake-provider integration coverage and assignment-facing run documentation.

The implementation does not add features outside the existing MVP spec. In
particular, it does not add themes, mouse workflows, parallel tool execution,
steering queues, session trees, plugins, MCP, or remote entrypoints.

## 2. User Experience

The layout is exactly three vertically stacked regions:

```text
VerticalScroll#transcript
TextArea#composer
Static#statusline
```

The transcript occupies the main view and contains user, assistant, tool, and
system rows. There is no persistent information sidebar. Workspace directory,
git branch, model, reasoning level, permission mode, context, usage, session
short id, and runtime status are rendered in the bottom statusline. The
statusline remains one row; low-priority fields are hidden at narrow widths
rather than wrapped.

The composer submits on Enter only while the runtime is idle. Ctrl-C aborts an
active run, denies/cancels a pending approval, or clears idle input according
to the current state. A submitted slash command is handled locally and is
never added to model history.

Approval is a focused modal containing the tool name, arguments, risk category,
and reason, with Approve and Deny actions. The modal calls only
`runtime.resolve_approval()` and does not execute tools or read terminal input
directly.

## 3. Architecture

```text
Textual widgets
  -> CodingAgentApp command handlers
       -> AgentRuntime public API

AgentRuntime RuntimeEvent
  -> bounded asyncio queue owned by TUI bridge
       -> Textual event-loop worker
            -> reduce(TuiState, RuntimeEvent)
                 -> transcript/statusline rendering
```

`CodingAgentApp` owns the displayed `TuiState`, Textual widgets, command
routing, and bridge lifecycle. It subscribes to the runtime when mounted and
unsubscribes when unmounted or exited. The TUI never calls a provider, tool,
or session store directly.

The runtime event sink places events into a bounded queue. Assistant text
deltas may be coalesced by message id when the queue is full; lifecycle,
tool, approval, policy, and error events must remain observable. The bridge
applies events on the Textual event loop, one event at a time, by producing a
new state through the pure reducer. Runtime tasks never call widgets directly.
A bridge or rendering failure becomes a visible system/error row and must not
escape as an uncaught widget exception.

The existing reducer remains the source of truth for event-to-state mapping.
Widgets render state and do not independently infer run lifecycle, approval,
or tool status.

## 4. Components

### 4.1 `tui/app.py`

`CodingAgentApp` accepts an injected `AgentRuntime` and initial `TuiState` (or
constructs the initial state from runtime metadata). It composes the exact
three-region layout, starts and stops the runtime bridge, handles composer
submission, Ctrl-C, local commands, and approval actions, and refreshes child
widgets after each reduced state.

Async runtime calls run in Textual workers/tasks so the UI event loop remains
responsive. A second prompt is rejected while a run is active and produces a
local notice.

### 4.2 `tui/widgets.py`

Widgets provide focused rendering only:

- Transcript container and row rendering for all `TranscriptItem` kinds.
- Composer setup and input affordances.
- Compact statusline formatting with field-priority truncation.
- Approval modal and session selector modal.

Widget code receives state or explicit values and does not own persistence or
agent behavior.

### 4.3 `tui/commands.py`

A small parser returns a command name and argument list for slash-prefixed
input. The command registry supports exactly:

```text
/help
/new
/session
/resume <id-or-unique-prefix>
/compact
/context
/permission
/permission default
/permission workspace
/permission full
/clear
/quit
```

Commands either call the corresponding runtime API or publish a TUI notice.
Unknown commands remain local and produce an error notice. `/clear` clears only
the visible transcript. `/new` creates a persisted session and resets visible
state through the existing `session_loaded` event. `/resume` accepts a full id
or unique prefix; ambiguous and missing matches become local notices.

### 4.4 Top-level `app.py`

The application factory resolves the workspace, model, base URL, credential
environment variable, context window, and session directory. It constructs
the OpenAI-compatible provider, six built-in tools, registry, policy,
executor, context policy, session store, runtime, and Textual app.

`main()` parses `--workspace`, `--model`, `--base-url`, `--session-dir`, and
`--context-window`, then starts Textual. `--help` works without credentials.
Missing model or credential errors are reported only on the actual run path and
never include secret values.

## 5. Data Flow

### Prompt flow

1. Composer receives a prompt.
2. If it starts with `/`, the local command registry handles it.
3. Otherwise the app calls `runtime.submit(prompt)` from an async worker.
4. Runtime events enter the bridge queue.
5. The bridge reduces each event into a new immutable `TuiState`.
6. Transcript and statusline widgets render the new snapshot.

### Approval flow

1. Runtime publishes `approval_requested`.
2. Reducer stores the validated `ApprovalRequest` and sets
   `waiting_approval`.
3. The app presents the modal.
4. Approve or Deny calls `runtime.resolve_approval(request_id, decision)`.
5. Runtime publishes resolution; reducer closes the pending modal state.

### Session flow

`/session` requests summaries from `runtime.list_sessions()`, displays newest
first with short id, updated time, workspace, and bounded title, then calls
`runtime.resume()` for the selected session. `/new` calls `new_session()`.
Both operations are rejected while a run is active and report the failure as a
notice.

## 6. Error and Safety Behavior

- Runtime exceptions are converted to visible system/error rows.
- Approval modal actions are the only UI path to resolve an approval.
- Ctrl-C sets runtime cancellation and cancels pending approval through the
  runtime API; it does not execute or retry a tool.
- Command text is never persisted as a user prompt.
- Missing configuration output is redacted and does not print API keys.
- Queue saturation may combine assistant deltas, but must preserve lifecycle,
  tool, approval, policy, and error events.
- No UI operation bypasses `ToolExecutor` or the existing permission policy.

## 7. Testing and Verification

Task 13 tests use Textual Pilot with an injected fake runtime to verify:

- The exact transcript, composer, and statusline widgets exist.
- Consecutive assistant deltas render as one visible assistant message while a
  terminal lifecycle event remains applied.
- Enter submits an idle prompt through the runtime.
- Ctrl-C calls abort during a run.
- Approval actions call only `resolve_approval()`.

Task 14 tests verify command parsing, local handling, session selector resume,
unknown commands, ambiguous prefixes, statusline field formatting, and
permission/context controls.

Task 15 adds a Fake Provider integration test that writes a small file and runs
an allowed verification command in a temporary workspace. Final verification
runs:

```text
pytest -q
ruff check src tests
ruff format --check src tests
python -m coding_agent.app --help
```

The implementation is complete only when these checks pass, the CLI help path
works without credentials, and the real CLI can reach the three-region TUI
shell with a configured provider.
