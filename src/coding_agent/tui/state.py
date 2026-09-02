from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from coding_agent.session.models import ApprovalRequest


class TranscriptItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["user", "assistant", "tool", "system", "local_command"]
    item_id: str
    text: str = ""
    pending: bool = False
    started_at: float | None = None
    timestamp: datetime | None = None
    level: Literal["notice", "error"] | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_status: Literal["running", "success", "error", "cancelled"] | None = None
    command: str | None = None
    elapsed_seconds: float | None = None
    truncated: bool | None = None
    exit_code: int | None = None
    retries: int | None = None
    expanded: bool = False


class TuiState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str | None = None
    workspace: str
    git_branch: str | None = None
    model: str
    reasoning: str | None = None
    context_used: int = 0
    context_window: int = 0
    context_estimated: bool = False
    policy: Literal["default", "workspace", "full"] = "workspace"
    status: Literal["idle", "running", "waiting_approval", "error", "aborted"] = "idle"
    active_run_id: str | None = None
    active_turn_id: str | None = None
    transcript: list[TranscriptItem] = Field(default_factory=list)
    active_tool_call_id: str | None = None
    pending_approval: ApprovalRequest | None = None
    input_text: str = ""
    compacting: bool = False
    run_started_at: float | None = None
    spinner_frame: int = 0


def initial_state(
    workspace: str,
    model: str,
    *,
    session_id: str | None = None,
    context_window: int = 0,
    policy: Literal["default", "workspace", "full"] = "workspace",
) -> TuiState:
    return TuiState(
        session_id=session_id,
        workspace=workspace,
        model=model,
        context_window=context_window,
        policy=policy,
    )
