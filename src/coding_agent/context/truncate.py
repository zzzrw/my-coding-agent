import json
from collections import OrderedDict

from coding_agent.runtime.models import Message, Usage
from coding_agent.session.models import ContextView, SessionMessage


class TruncatePolicy:
    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget

    @staticmethod
    def estimate_tokens(messages: list[Message]) -> int:
        serialized = json.dumps(
            [message.model_dump(mode="json") for message in messages],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (len(serialized) + 3) // 4

    def prepare(
        self,
        history: list[SessionMessage],
        *,
        system_prompt: Message,
        context_window: int,
        usage: Usage | None,
        force: bool = False,
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
        estimated_tokens = self.estimate_tokens(base_messages)
        provider_tokens = (
            usage.input_tokens if usage and usage.input_tokens > 0 else None
        )
        accounting_tokens = provider_tokens or estimated_tokens
        must_compact = force and len(retained_keys) > 1

        removed = 0
        while len(retained_keys) > 1 and (
            must_compact or self.estimate_tokens(base_messages) > limit
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
        used_tokens = (
            self.estimate_tokens(base_messages) if compacted else accounting_tokens
        )
        overflow = used_tokens > limit and len(retained_keys) <= 1
        return ContextView(
            messages=base_messages,
            used_tokens=used_tokens,
            context_window=context_window,
            estimated=provider_tokens is None or compacted,
            compacted=compacted,
            removed_turns=removed,
            overflow=overflow,
        )
