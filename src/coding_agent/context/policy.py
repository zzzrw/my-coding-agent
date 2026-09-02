from typing import Protocol

from coding_agent.runtime.models import Message, Usage
from coding_agent.session.models import ContextView, SessionMessage
from coding_agent.tools.models import ToolSchema


class ContextPolicy(Protocol):
    def prepare(
        self,
        history: list[SessionMessage],
        *,
        system_prompt: Message,
        context_window: int,
        usage: Usage | None,
        force: bool = False,
        tools: list[ToolSchema] | None = None,
    ) -> ContextView: ...
