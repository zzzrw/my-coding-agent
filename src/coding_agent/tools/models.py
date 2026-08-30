from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]
    risk_level: Literal["read", "mutate_file", "mutate_shell", "dangerous"]
    is_parallel_safe: bool = False


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    ok: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
