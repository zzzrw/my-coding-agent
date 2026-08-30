import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from coding_agent.runtime.models import LLMEvent, Message
from coding_agent.tools.models import ToolSchema


class LLMProvider(Protocol):
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        model: str,
        signal: asyncio.Event,
    ) -> AsyncIterator[LLMEvent]: ...
