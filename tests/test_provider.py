import asyncio

import pytest
from fakes import FakeProvider

from coding_agent.llm.openai_compatible import ChatChunkParser
from coding_agent.runtime.models import LLMEvent


def test_text_chunk_becomes_text_delta():
    assert ChatChunkParser().parse(
        {"choices": [{"delta": {"content": "hello"}}]}
    ) == LLMEvent(type="text_delta", text="hello")


def test_tool_argument_chunks_preserve_call_identity():
    parser = ChatChunkParser()
    event = parser.parse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "read_file", "arguments": '{"pa'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert event.type == "tool_call_delta"
    event = parser.parse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'th"}'}}]
                    }
                }
            ]
        }
    )
    assert event.tool_call_id == "call-1" and event.arguments_delta == 'th"}'


def test_tool_start_and_end_events_are_explicit():
    parser = ChatChunkParser()
    start = parser.parse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "function": {"name": "read_file"}}
                        ]
                    }
                }
            ]
        }
    )
    end = parser.parse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    assert start.type == "tool_call_start" and end.type == "tool_call_end"


@pytest.mark.asyncio
async def test_fake_provider_yields_events_without_network():
    provider = FakeProvider(
        [
            LLMEvent(type="text_delta", text="done"),
            LLMEvent(type="response_end", finish_reason="stop"),
        ]
    )
    events = [
        event
        async for event in provider.stream([], [], model="fake", signal=asyncio.Event())
    ]
    assert [event.type for event in events] == ["text_delta", "response_end"]
