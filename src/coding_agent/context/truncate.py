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


def _after_last_assistant(messages: list[Message]) -> list[Message]:
    """Return the messages strictly after the newest assistant response.

    The usage handed to :meth:`TruncatePolicy.prepare` always belongs to the
    request that produced the newest assistant message in ``messages``, so
    whatever sits after that message arrived after the measurement and is the
    only part a provider total cannot already account for. When there is no
    assistant message yet, nothing after a measured response exists to count.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            return messages[index + 1 :]
    return []


_HEADROOM_MIN = 12_000
_HEADROOM_FRACTION = 0.05


class TruncatePolicy:
    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget

    @staticmethod
    def _compaction_threshold(
        budget: int | None, context_window: int
    ) -> tuple[int, int]:
        """Return ``(capacity, threshold)`` used by :meth:`prepare`.

        ``capacity`` is the hard ceiling an overflow is reported against. The
        auto-compaction ``threshold`` reserves headroom under the operating
        window (``max(12000, window*0.05)``) so compaction fires before the next
        request can overflow -- a new prompt or tool result can push a request
        past the previous response's usage. When a budget was configured it is
        authoritative and no headroom is reserved; when the window is too small
        to reserve any meaningful headroom the threshold stays at the full
        window, so compaction never races below the current turn.
        """
        capacity = min(budget or context_window, context_window)
        threshold = capacity
        if budget is None:
            reserve = max(_HEADROOM_MIN, int(context_window * _HEADROOM_FRACTION))
            if reserve < context_window:
                threshold = context_window - reserve
        return capacity, threshold

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
        capacity, threshold = self._compaction_threshold(self.budget, context_window)
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
        provider_total = usage.authoritative_total() if usage is not None else None
        must_compact = force and len(retained_keys) > 1

        removed = 0
        while len(retained_keys) > 1 and (
            must_compact or self.estimate_tokens(base_messages, tools=tools) > threshold
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
            # The meter shows the full request total, which already includes the
            # last measured response's own output tokens (its output_tokens).
            # Only messages appended strictly after that response -- a newer
            # turn's prompt, tool results appended since the last request -- are
            # outside it. Estimate just that suffix: re-estimating the response
            # text or earlier history would double-count it and make the meter
            # oscillate (up at each step start, down when the next measured
            # total re-emits).
            appended_suffix = _after_last_assistant(base_messages)
            appended = self.estimate_tokens(appended_suffix) if appended_suffix else 0
            used_tokens = provider_total + appended
        else:
            used_tokens = estimate
        overflow = used_tokens > capacity and len(retained_keys) <= 1
        return ContextView(
            messages=base_messages,
            used_tokens=used_tokens,
            context_window=context_window,
            estimated=provider_total is None or compacted,
            compacted=compacted,
            removed_turns=removed,
            overflow=overflow,
        )
