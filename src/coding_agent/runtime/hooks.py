from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from coding_agent.runtime.models import ToolCall
from coding_agent.tools.models import ToolResult

BeforeToolHook = Callable[[ToolCall], Awaitable[ToolCall | ToolResult | None]]
AfterToolHook = Callable[[ToolCall, ToolResult], Awaitable[ToolResult | None]]
ErrorHook = Callable[[Exception], Awaitable[None]]
BeforeModelHook = Callable[[], Awaitable[None]]


@dataclass
class HookSet:
    before_model: list[BeforeModelHook] = field(default_factory=list)
    before_tool: list[BeforeToolHook] = field(default_factory=list)
    after_tool: list[AfterToolHook] = field(default_factory=list)
    on_error: list[ErrorHook] = field(default_factory=list)
