import asyncio
from types import SimpleNamespace

import pytest
from fakes import FakeProvider

from coding_agent.llm.openai_compatible import ChatChunkParser, OpenAICompatibleProvider
from coding_agent.runtime.models import LLMEvent, Message, ToolCall


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


def test_content_chunk_with_finish_reason_preserves_response_end():
    events = ChatChunkParser().parse_many(
        {
            "choices": [
                {
                    "delta": {"content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }
    )
    assert [event.type for event in events] == ["text_delta", "response_end"]
    assert events[-1].usage.total_tokens == 3


def test_parser_keeps_multiple_tool_calls_in_one_chunk():
    events = ChatChunkParser().parse_many(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "function": {"name": "read_file"}},
                            {
                                "index": 1,
                                "id": "c2",
                                "function": {"name": "list_files"},
                            },
                        ]
                    }
                }
            ]
        }
    )
    assert [event.tool_call_id for event in events] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_provider_serializes_post_tool_messages():
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return []

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider = OpenAICompatibleProvider(api_key="test", client=client)
    messages = [
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})
            ],
        ),
        Message(role="tool", tool_call_id="c1", name="read_file", content="contents"),
    ]
    [
        event
        async for event in provider.stream(
            messages, [], model="fake", signal=asyncio.Event()
        )
    ]
    assert captured["messages"] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "contents",
            "tool_call_id": "c1",
            "name": "read_file",
        },
    ]


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
