import json
from collections import OrderedDict

from coding_agent.runtime.models import Message, Usage
from coding_agent.session.models import ContextView, SessionMessage
from coding_agent.tools.models import ToolSchema

_ASCII_CHARS_PER_TOKEN = 4


def _text_tokens(text: str) -> int:
    """Estimate the tokens in ``text``.

    ASCII runs roughly four characters per token; every non-ASCII codepoint
    (CJK, emoji, accented text...) costs about one token of its own. Ceiling
    division keeps short ASCII payloads from estimating to zero, and counting
    codepoints directly (rather than JSON ``\\u`` escapes) keeps CJK/emoji at
    roughly a token each instead of inflating them.
    """
    ascii_chars = 0
    tokens = 0
    for char in text:
        if ord(char) < 128:
            ascii_chars += 1
        else:
            tokens += 1
    tokens += (ascii_chars + _ASCII_CHARS_PER_TOKEN - 1) // _ASCII_CHARS_PER_TOKEN
    return tokens


def _wire_message(message: Message) -> dict:
    """Mirror the exact per-role payload the provider assembles for a message.

    Kept in lockstep with ``OpenAICompatibleProvider.stream`` so the estimation
    measures the same shape that actually rides on the wire request.
    """
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
    return item


def _wire_tool(schema: ToolSchema) -> dict:
    """Mirror the exact tool/skill schema object the provider receives."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


class TruncatePolicy:
    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget

    @staticmethod
    def estimate_tokens(
        messages: list[Message], *, tools: list[ToolSchema] | None = None
    ) -> int:
        """Estimate the wire payload tokens for ``messages`` and ``tools``.

        Pure and deterministic: this is the single estimator shared by the live
        meter fallback and the planning used on replay, so stored compaction
        records never depend on which path produced them. It mirrors the request
        the provider receives (the same per-role messages AND tool/skill
        schemas) and counts CJK/emoji-aware with ceiling division, so it is not
        lower than what a real tool-schema-bearing request needs.
        """
        payload: dict = {"messages": [_wire_message(message) for message in messages]}
        if tools:
            payload["tools"] = [_wire_tool(schema) for schema in tools]
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return _text_tokens(serialized)

    def prepare(
        self,
        history: list[SessionMessage],
        *,
        system_prompt: Message,
        context_window: int,
        usage: Usage | None,
        force: bool = False,
        tools: list[ToolSchema] | None = None,
    ) -> ContextView:
        limit = min(self.budget or context_window, context_window)
        groups: OrderedDict[str, list[Message]] = OrderedDict()
        for item in history:
            key = item.turn_id or f"record:{item.record_id}"
            groups.setdefault(key, []).append(item.message.model_copy(deep=True))

        keys = list(groups)
        retained_keys = list(keys)
        base_messages = [system_prompt.model_copy(deep=True)] + [
            message for key in retained_keys for message in groups[key]
        ]
        estimate = self.estimate_tokens(base_messages, tools=tools)
        provider_total = (
            usage.total_tokens if usage and usage.total_tokens > 0 else None
        )
        provider_input = (
            usage.input_tokens if usage and usage.input_tokens > 0 else None
        )
        must_compact = force and len(retained_keys) > 1

        removed = 0
        while len(retained_keys) > 1 and (
            must_compact or self.estimate_tokens(base_messages, tools=tools) > limit
        ):
            retained_keys.pop(0)
            removed += 1
            must_compact = False
            marker = Message(
                role="system",
                content=f"[context compacted: {removed} earlier turn(s) removed]",
            )
            base_messages = [system_prompt.model_copy(deep=True), marker] + [
                message for key in retained_keys for message in groups[key]
            ]

        compacted = removed > 0
        if compacted:
            used_tokens = self.estimate_tokens(base_messages, tools=tools)
        elif provider_total is not None:
            # The meter shows the full request total. If messages were appended
            # since that response (a new turn's prompt, a tool result, ...) its
            # input_tokens never included them, so add a small estimate of the
            # items that arrived afterwards. Clamping keeps a purely-estimated
            # context from ever reporting less than the authoritative total.
            appended = estimate - provider_input if provider_input is not None else 0
            used_tokens = provider_total + max(0, appended)
        else:
            used_tokens = estimate
        overflow = used_tokens > limit and len(retained_keys) <= 1
        return ContextView(
            messages=base_messages,
            used_tokens=used_tokens,
            context_window=context_window,
            estimated=provider_total is None or compacted,
            compacted=compacted,
            removed_turns=removed,
            overflow=overflow,
        )
