from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

RuntimeEventType = Literal[
    "run_started",
    "run_finished",
    "run_error",
    "turn_started",
    "turn_finished",
    "user_message",
    "assistant_started",
    "assistant_delta",
    "assistant_finished",
    "assistant_message",
    "tool_started",
    "tool_finished",
    "approval_requested",
    "approval_resolved",
    "status_changed",
    "context_updated",
    "session_loaded",
    "policy_changed",
    "notice",
]


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: RuntimeEventType
    run_id: str | None = None
    turn_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


EventSink = Callable[[RuntimeEvent], Awaitable[None]]
