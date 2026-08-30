import asyncio
from collections.abc import AsyncIterator, Iterable
from copy import deepcopy

from coding_agent.runtime.models import LLMEvent, Message, ToolCall
from coding_agent.tools.models import ToolResult, ToolSchema


class FakeProvider:
    def __init__(self, events: Iterable[LLMEvent]):
        self.events = list(events)
        self.requests: list[tuple[list[Message], list[ToolSchema], str]] = []

    async def stream(
        self, messages, tools, *, model, signal: asyncio.Event
    ) -> AsyncIterator[LLMEvent]:
        self.requests.append((deepcopy(messages), deepcopy(tools), model))
        for event in self.events:
            if signal.is_set():
                return
            yield event


class RepeatingToolProvider(FakeProvider):
    def __init__(self, tool_name: str):
        super().__init__(
            [
                LLMEvent(
                    type="tool_call_start", tool_call_id="call-1", tool_name=tool_name
                ),
                LLMEvent(
                    type="tool_call_delta", tool_call_id="call-1", arguments_delta="{}"
                ),
                LLMEvent(type="response_end", finish_reason="tool_calls"),
            ]
        )


class BlockingFakeProvider(FakeProvider):
    def __init__(self):
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, messages, tools, *, model, signal):
        self.started.set()
        await self.release.wait()
        if not signal.is_set():
            yield LLMEvent(type="response_end", finish_reason="stop")


class FakeTool:
    def __init__(self, name: str):
        self.name = name
        self.schema = ToolSchema(
            name=name,
            description=name,
            parameters={"type": "object"},
            risk_level="read",
        )

    async def run(self, arguments):
        return ToolResult(tool_call_id="", tool_name=self.name, ok=True, content="ok")


def assistant_with_tool(name: str, arguments: dict) -> Message:
    return Message(
        role="assistant",
        tool_calls=[ToolCall(id="call-1", name=name, arguments=arguments)],
    )


def assistant_text(text: str) -> Message:
    return Message(role="assistant", content=text)
