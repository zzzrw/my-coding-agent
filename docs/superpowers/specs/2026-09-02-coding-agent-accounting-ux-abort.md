# Post-Roadmap Correctness & UX Amendments (2026-09-02)

Consolidated, dated amendments to the MVP spec (`2026-08-30-coding-agent-mvp-design.md`)
and its roadmap follow-ups. Each item supersedes/extends earlier text; tests are named so a
reader can verify behavior.

## Context / token accounting (supersedes MVP context & statusline text)

- **Provider usage is the meter.** `AgentRunner.run_turn` is seeded with the previous full
  request's `Usage` (persisted across turns, restored on resume); after `assistant_finished`
  a fresh `context_updated` is re-emitted with the final total so the meter updates without
  waiting for the next request. `runtime` derives `context_used` from the outcome usage rather
  than keeping a stale step value.
- **Displayed "used" is the full-request total** (`total_tokens` when > 0; if an endpoint omits
  it, fall back to `input+output`, else to an estimate). Items appended after the measured
  response are estimated only via `_after_last_assistant` (messages strictly after the newest
  assistant response), so the prior response's own output is never double-counted and the meter
  grows monotonically.
- **Estimates are a fallback/planning tool only** and measure the true wire payload: the same
  messages *and* tool/skill schemas the provider receives, CJK/emoji ≈ 1 token per codepoint,
  ceiling division.
- **Compaction headroom:** auto-compaction fires when the estimate exceeds
  `window − max(12 000, window * 0.05)` (an explicit budget stays authoritative; tiny windows
  fall back to the full window and never drop the current turn).
- **Statusline shows only `ctx used/window (configured|estimated)`** — the middle "remaining"
  number was removed.
- Verification: `tests/test_ctx_meter.py`, `tests/test_context.py`, `tests/test_runtime.py`.

## Aborted/interrupted turns keep their completed tool work

`SessionStore.project_messages` used to drop every record of a turn closed with reason
`aborted`/`interrupted`; the next turn then forgot the writes/observations made right before
the abort (session `15fad35e`). Now an interrupted turn is skipped only when it produced no
assistant/tool content (a lone user message is not replayed); completed assistant/tool groups
project into later requests, and the per-group dangling cleaner still excises a partial final
step. Read-side only; no writes change. Verification: `tests/test_session_hardening.py`.

## Permissions & config

- **Default permission mode is `workspace`** (new sessions, resume, and fork all start in it);
  `default`/`full` remain selectable via `/permission`.
- **Permission changes are confirmed**: a successful change emits a `[notice] permission mode
  changed to <mode>` row.
- **Reply language**: `Config.language` (default `zh`; `en`/… accepted). The system prompt
  opens with `Respond in <Language>.`; `create_app` threads it from the config file or an
  explicit argument.
- Supersedes MVP/config text that said the initial mode is `default` or that `max_steps`
  defaults to 20: the run loop is **unbounded by default** with an optional int cap (see
  `2026-09-02-coding-agent-unbounded-max-steps`).

## TUI behavior

- **Transcript follows the newest row for local commands too**: a row appended while the user
  is at the bottom stays visible even when an overlay opens before the scheduled refresh runs
  (re-scroll after layout settles). Verification: `tests/test_scroll_follow.py`.
- **Drafting indicator**: while a step is generating a tool call with no prose yet, the pending
  row shows `drafting <tool> · N chars` (ephemeral UI only, never persisted). Never shows raw
  tool-call arguments.
- **Full tool-call arguments never reach the user**: `/inbox` rows, resumed tool labels,
  approval header, and plan rows all cap/omit payload keys (`content`/`old_text`/`new_text`),
  bounded ~160 chars. The approval diff preview (a capped change preview) is retained.
- **System prompt** was rewritten with senior-engineer guidance (How you work / Final response /
  `load_skill` hook); the permission boundaries text is preserved.
