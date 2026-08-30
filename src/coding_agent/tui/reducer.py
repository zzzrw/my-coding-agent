from __future__ import annotations

from typing import Any

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.session.models import ApprovalRequest
from coding_agent.tui.state import TranscriptItem, TuiState


def reduce(state: TuiState, event: RuntimeEvent) -> TuiState:
    """Apply one runtime event without mutating the input snapshot."""
    payload = event.payload
    updates: dict[str, Any] = {}
    transcript = [row.model_copy() for row in state.transcript]

    if event.type == "run_started":
        updates.update(
            status="running",
            active_run_id=event.run_id,
            active_turn_id=event.turn_id,
            session_id=_optional_str(payload.get("session_id"), state.session_id),
            model=_optional_str(payload.get("model"), state.model),
            policy=_policy(payload.get("policy"), state.policy),
            pending_approval=None,
            active_tool_call_id=None,
        )

    elif event.type == "run_finished":
        outcome = payload.get("outcome", {})
        reason = outcome if isinstance(outcome, str) else None
        if isinstance(outcome, dict):
            reason = outcome.get("reason")
        status = _terminal_status(reason)
        updates.update(
            status=status,
            active_run_id=None,
            active_turn_id=None,
            active_tool_call_id=None,
            pending_approval=None,
        )

    elif event.type == "run_error":
        updates.update(
            status="error",
            active_run_id=None,
            active_turn_id=None,
            active_tool_call_id=None,
            pending_approval=None,
        )
        _append_system(transcript, payload.get("message", "runtime error"))

    elif event.type == "user_message":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id:
            _append_or_update(
                transcript,
                TranscriptItem(
                    kind="user",
                    item_id=message_id,
                    text=_text(payload.get("text")),
                ),
            )

    elif event.type == "assistant_started":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id:
            _append_or_update(
                transcript,
                TranscriptItem(kind="assistant", item_id=message_id),
            )

    elif event.type == "assistant_delta":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id:
            index = _find_assistant(transcript, message_id)
            if index is None:
                transcript.append(
                    TranscriptItem(
                        kind="assistant",
                        item_id=message_id,
                        text=_text(payload.get("text")),
                    )
                )
            else:
                row = transcript[index]
                transcript[index] = row.model_copy(
                    update={"text": row.text + _text(payload.get("text"))}
                )

    elif event.type == "assistant_finished":
        message_id = _non_empty_str(payload.get("message_id"))
        if message_id is not None and _find_assistant(transcript, message_id) is None:
            transcript.append(TranscriptItem(kind="assistant", item_id=message_id))

    elif event.type == "tool_started":
        call_id = _non_empty_str(payload.get("tool_call_id"))
        if call_id:
            row = TranscriptItem(
                kind="tool",
                item_id=call_id,
                tool_call_id=call_id,
                tool_name=_optional_str(payload.get("tool_name"), None),
                text=_arguments_text(payload.get("arguments")),
                tool_status="running",
            )
            _append_or_update(transcript, row, tool_call_id=call_id)
            updates.update(active_tool_call_id=call_id, status="running")

    elif event.type == "tool_finished":
        call_id = _non_empty_str(payload.get("tool_call_id"))
        if call_id:
            ok = bool(payload.get("ok", False))
            error = _text(payload.get("error"))
            text = _text(payload.get("content"))
            status = _tool_status(payload.get("status"), ok, error, text)
            index = _find_tool(transcript, call_id)
            replacement = TranscriptItem(
                kind="tool",
                item_id=call_id,
                tool_call_id=call_id,
                tool_name=_optional_str(payload.get("tool_name"), None),
                text=text or error,
                tool_status=status,
            )
            if index is None:
                transcript.append(replacement)
            else:
                old = transcript[index]
                transcript[index] = old.model_copy(
                    update={
                        "tool_name": replacement.tool_name or old.tool_name,
                        "text": replacement.text,
                        "tool_status": status,
                    }
                )
            if (
                state.active_tool_call_id == call_id
                or updates.get("active_tool_call_id") == call_id
            ):
                updates["active_tool_call_id"] = None

    elif event.type == "approval_requested":
        request = _approval(payload.get("request"))
        if request is not None:
            updates.update(pending_approval=request, status="waiting_approval")

    elif event.type == "approval_resolved":
        request_id = _non_empty_str(payload.get("request_id"))
        if (
            state.pending_approval is not None
            and request_id == state.pending_approval.request_id
        ):
            updates["pending_approval"] = None
            if payload.get("status") == "cancelled":
                updates["status"] = "aborted"
            elif payload.get("status") in {"approved", "denied"}:
                updates["status"] = "running"

    elif event.type == "status_changed":
        status = payload.get("status")
        if status in {"idle", "running", "waiting_approval", "error", "aborted"}:
            updates["status"] = status

    elif event.type == "context_updated":
        updates.update(
            context_used=_int(payload.get("used_tokens"), state.context_used),
            context_window=_int(payload.get("context_window"), state.context_window),
            context_estimated=bool(payload.get("estimated", state.context_estimated)),
        )

    elif event.type == "session_loaded":
        updates.update(
            session_id=_optional_str(payload.get("session_id"), state.session_id),
            workspace=_optional_str(payload.get("workspace"), state.workspace),
            model=_optional_str(payload.get("model"), state.model),
            policy="default",
            status="idle",
            active_run_id=None,
            active_turn_id=None,
            active_tool_call_id=None,
            pending_approval=None,
            transcript=[],
        )

    elif event.type == "policy_changed":
        updates["policy"] = _policy(payload.get("policy"), state.policy)

    elif event.type == "notice":
        _append_system(transcript, payload.get("message", ""))

    if event.type != "session_loaded":
        updates["transcript"] = transcript
    return state.model_copy(update=updates)


def _append_or_update(
    transcript: list[TranscriptItem],
    item: TranscriptItem,
    *,
    tool_call_id: str | None = None,
) -> None:
    index = (
        _find_tool(transcript, tool_call_id)
        if tool_call_id
        else _find_assistant(transcript, item.item_id)
    )
    if index is None:
        transcript.append(item)
    else:
        transcript[index] = transcript[index].model_copy(
            update=item.model_dump(exclude_unset=True)
        )


def _append_system(transcript: list[TranscriptItem], message: object) -> None:
    transcript.append(
        TranscriptItem(
            kind="system",
            item_id=f"system-{len(transcript)}",
            text=_text(message),
        )
    )


def _find_assistant(transcript: list[TranscriptItem], message_id: str) -> int | None:
    return next(
        (
            index
            for index, row in enumerate(transcript)
            if row.kind == "assistant" and row.item_id == message_id
        ),
        None,
    )


def _find_tool(transcript: list[TranscriptItem], call_id: str) -> int | None:
    return next(
        (
            index
            for index, row in enumerate(transcript)
            if row.kind == "tool" and row.tool_call_id == call_id
        ),
        None,
    )


def _approval(value: object) -> ApprovalRequest | None:
    if isinstance(value, ApprovalRequest):
        return value
    if isinstance(value, dict):
        try:
            return ApprovalRequest.model_validate(value)
        except (TypeError, ValueError):
            return None
    return None


def _terminal_status(reason: object) -> str:
    if reason == "aborted":
        return "aborted"
    if reason in {"provider_error", "session_error"}:
        return "error"
    return "idle"


def _tool_status(value: object, ok: bool, error: str, text: str) -> str:
    if value in {"running", "success", "error", "cancelled"}:
        return value
    if ok:
        return "success"
    return _tool_error_status(error, text)


def _tool_error_status(error: str, text: str) -> str:
    combined = f"{error} {text}".lower()
    return "cancelled" if "cancel" in combined else "error"


def _arguments_text(value: object) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _text(value: object) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_str(value: object, fallback: str | None) -> str | None:
    value = _non_empty_str(value)
    return value if value is not None else fallback


def _int(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _policy(value: object, fallback: str) -> str:
    return value if value in {"default", "workspace", "full"} else fallback
