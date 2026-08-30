from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(StrictModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class Usage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMEvent(StrictModel):
    type: Literal[
        "text_delta",
        "tool_call_start",
        "tool_call_delta",
        "tool_call_end",
        "response_end",
        "error",
    ]
    text: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    error: str | None = None


class RuntimeStatus(StrictModel):
    status: Literal["idle", "running", "waiting_approval", "error", "aborted"] = "idle"
    run_id: str | None = None
    turn_id: str | None = None
    usage: Usage | None = None
    context_used: int = 0
    context_window: int | None = None
    context_estimated: bool = False


class TurnOutcome(StrictModel):
    reason: Literal[
        "completed", "max_steps", "aborted", "provider_error", "session_error"
    ]
    final_text: str = ""
    steps: int = 0
    usage: Usage | None = None
