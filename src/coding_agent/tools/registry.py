import asyncio
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from .models import ToolResult, ToolSchema

PermissionMode = Literal["default", "workspace", "full"]


class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Path
    permission_mode: PermissionMode
    allow_outside_once: bool = False


class Tool(Protocol):
    schema: ToolSchema
    args_model: type[BaseModel]

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ToolContext,
        signal: asyncio.Event,
    ) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: OrderedDict[str, Tool] = OrderedDict()

    def register(self, tool: Tool) -> None:
        name = tool.schema.name
        if name in self._tools:
            raise ValueError(f"duplicate tool name: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[ToolSchema]:
        return [tool.schema.model_copy(deep=True) for tool in self._tools.values()]
