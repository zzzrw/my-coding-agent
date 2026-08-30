from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from coding_agent.runtime.models import Message


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionHeader(StrictModel):
    kind: Literal["header"] = "header"
    schema_version: int = 1
    session_id: str
    workspace: str
    model: str
    title: str = "New session"
    created_at: datetime
    updated_at: datetime
    context_window: int


RecordType = Literal[
    "turn_start",
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "turn_end",
    "compaction",
    "context_updated",
    "run_start",
    "run_end",
    "error",
]


class SessionRecord(StrictModel):
    id: str
    seq: int
    timestamp: datetime
    type: RecordType
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None
    run_id: str | None = None
    turn_id: str | None = None


class SessionMessage(StrictModel):
    record_id: str
    turn_id: str | None = None
    seq: int
    message: Message


class SessionSummary(StrictModel):
    id: str
    workspace: str
    created_at: datetime
    updated_at: datetime
    title: str
    last_status: str


class ContextView(StrictModel):
    messages: list[Message]
    used_tokens: int
    context_window: int
    estimated: bool
    compacted: bool
    removed_turns: int = 0
    overflow: bool = False


class ApprovalRequest(StrictModel):
    request_id: str
    tool_call_id: str
    tool_name: str
    risk_level: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
