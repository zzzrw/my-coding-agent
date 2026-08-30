import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from coding_agent.runtime.models import LLMEvent, Message, Usage
from coding_agent.tools.models import ToolSchema


class ChatChunkParser:
    def __init__(self) -> None:
        self._calls: dict[int, tuple[str | None, str | None]] = {}

    def parse(self, chunk: Any) -> LLMEvent:
        return self.parse_many(chunk)[0]

    def parse_many(self, chunk: Any) -> list[LLMEvent]:
        if isinstance(chunk, str):
            if chunk == "[DONE]":
                return [LLMEvent(type="response_end")]
            chunk = json.loads(chunk)
        choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        events: list[LLMEvent] = []
        if delta.get("content") is not None:
            events.append(LLMEvent(type="text_delta", text=delta["content"]))
        tool_calls = delta.get("tool_calls") or []
        if tool_calls:
            for call in tool_calls:
                index = call.get("index", 0)
                fn = call.get("function") or {}
                call_id = call.get("id") or self._calls.get(index, (None, None))[0]
                name = fn.get("name") or self._calls.get(index, (None, None))[1]
                prior = self._calls.get(index)
                self._calls[index] = (call_id, name)
                if (
                    call_id
                    and not fn.get("arguments")
                    and (prior is None or (call.get("id") and call_id != prior[0]))
                ):
                    events.append(
                        LLMEvent(
                            type="tool_call_start", tool_call_id=call_id, tool_name=name
                        )
                    )
                args = fn.get("arguments")
                if args:
                    events.append(
                        LLMEvent(
                            type="tool_call_delta",
                            tool_call_id=call_id,
                            tool_name=name,
                            arguments_delta=args,
                        )
                    )
        reason = choice.get("finish_reason")
        if reason:
            if reason == "tool_calls":
                events.append(LLMEvent(type="tool_call_end", finish_reason=reason))
            else:
                usage = chunk.get("usage")
                parsed_usage = None
                if usage:
                    parsed_usage = Usage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    )
                events.append(
                    LLMEvent(
                        type="response_end", finish_reason=reason, usage=parsed_usage
                    )
                )
            return events
        usage = chunk.get("usage") if isinstance(chunk, dict) else None
        if usage:
            events.append(
                LLMEvent(
                    type="response_end",
                    usage=Usage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    ),
                )
            )
        return events or [LLMEvent(type="response_end")]


class OpenAICompatibleProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        )
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
        )
        self.client = client or AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        model: str,
        signal: asyncio.Event,
    ) -> AsyncIterator[LLMEvent]:
        wire_messages = []
        for message in messages:
            item = {"role": message.role}
            if message.role == "assistant" and message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
                if message.content is not None:
                    item["content"] = message.content
            elif message.role == "tool":
                item.update(
                    {
                        "content": message.content or "",
                        "tool_call_id": message.tool_call_id,
                    }
                )
                if message.name:
                    item["name"] = message.name
            else:
                item["content"] = message.content
            wire_messages.append(item)
        wire_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        parser = ChatChunkParser()
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=wire_messages,
                tools=wire_tools or None,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in response:
                if signal.is_set():
                    return
                if chunk == "[DONE]":
                    yield LLMEvent(type="response_end")
                    return
                parsed_events = parser.parse_many(
                    chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                )
                for event in parsed_events:
                    if (
                        event.type == "response_end"
                        and not event.finish_reason
                        and not event.usage
                    ):
                        continue
                    yield event
        except Exception as exc:  # noqa: BLE001 - provider failures become normalized events
            yield LLMEvent(type="error", error=str(exc))
