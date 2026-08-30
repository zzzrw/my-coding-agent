import asyncio

import pytest
from pydantic import BaseModel

from coding_agent.tools.models import ToolResult, ToolSchema
from coding_agent.tools.registry import ToolContext, ToolRegistry


class _Args(BaseModel):
    value: str = ""


class _FakeTool:
    args_model = _Args

    def __init__(self, name: str):
        self.schema = ToolSchema(
            name=name,
            description=name,
            parameters=_Args.model_json_schema(),
            risk_level="read",
        )

    async def execute(
        self, arguments, *, context: ToolContext, signal: asyncio.Event
    ) -> ToolResult:
        return ToolResult(
            tool_call_id="", tool_name=self.schema.name, ok=True, content="ok"
        )


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry()
    registry.register(_FakeTool("read_file"))
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(_FakeTool("read_file"))


def test_registry_schema_order_is_registration_order_and_lookup():
    registry = ToolRegistry()
    first = _FakeTool("read_file")
    second = _FakeTool("write_file")
    registry.register(first)
    registry.register(second)
    assert [schema.name for schema in registry.schemas()] == ["read_file", "write_file"]
    assert registry.get("read_file") is first
    assert registry.get("missing") is None
