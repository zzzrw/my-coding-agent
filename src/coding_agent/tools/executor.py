import asyncio
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ValidationError

from coding_agent.policy.approval import ApprovalPolicy, PermissionMode
from coding_agent.runtime.hooks import HookSet
from coding_agent.runtime.models import ToolCall
from coding_agent.session.models import ApprovalRequest

from .models import ToolResult
from .registry import ToolContext, ToolRegistry

MAX_TOOL_OUTPUT_CHARS = 20_000


class ApprovalBroker(Protocol):
    async def request(
        self, request: ApprovalRequest
    ) -> Literal["approve", "deny", "cancelled"]: ...

    def cancel_all(self) -> None: ...


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: ApprovalPolicy,
        broker: ApprovalBroker,
        hooks: HookSet | None = None,
        default_timeout_seconds: float = 120.0,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.broker = broker
        self.hooks = hooks or HookSet()
        self.default_timeout_seconds = default_timeout_seconds

    async def execute(
        self,
        call: ToolCall,
        *,
        run_id: str,
        workspace: Path,
        permission_mode: PermissionMode,
        signal: asyncio.Event,
        output_sink=None,
    ) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None:
            return self._error(call, f"unknown tool: {call.name}")
        try:
            validated = tool.args_model.model_validate(call.arguments).model_dump()
        except ValidationError as exc:
            return self._error(call, f"invalid arguments: {exc}")

        active_call = call.model_copy(update={"arguments": validated})
        try:
            for hook in self.hooks.before_tool:
                replacement = await hook(active_call)
                if isinstance(replacement, ToolResult):
                    return self._normalize(call, replacement)
                if isinstance(replacement, ToolCall):
                    active_call = replacement

            decision = self.policy.decide(
                tool.schema,
                active_call.arguments,
                workspace=workspace,
                mode=permission_mode,
            )
            if decision.kind == "deny":
                return self._error(call, decision.reason)

            outside_once = False
            if decision.kind == "ask":
                request = ApprovalRequest(
                    request_id=uuid.uuid4().hex,
                    run_id=run_id,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=active_call.arguments,
                    risk_level=tool.schema.risk_level,
                    reason=decision.reason,
                )
                answer = await self.broker.request(request)
                if answer != "approve":
                    return self._error(
                        call,
                        "approval cancelled"
                        if answer == "cancelled"
                        else "approval denied",
                        cancelled=answer == "cancelled",
                    )
                outside_once = decision.allow_outside_once

            context = ToolContext(
                workspace=workspace,
                permission_mode=permission_mode,
                allow_outside_once=outside_once,
                on_output=output_sink,
            )
            result = await self._run_tool(
                tool.execute(active_call.arguments, context=context, signal=signal),
                call,
                signal,
            )
            for hook in self.hooks.after_tool:
                replacement = await hook(active_call, result)
                if replacement is not None:
                    result = replacement
            return self._normalize(call, result)
        except Exception as exc:  # noqa: BLE001
            for hook in self.hooks.on_error:
                with suppress(Exception):
                    await hook(exc)
            return self._error(call, str(exc))

    async def _run_tool(self, awaitable, call, signal) -> ToolResult:
        task = asyncio.create_task(awaitable)
        deadline = asyncio.get_running_loop().time() + self.default_timeout_seconds
        while not task.done():
            if signal.is_set():
                try:
                    return await asyncio.wait_for(asyncio.shield(task), 1.0)
                except TimeoutError:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    return self._error(call, "cancelled", cancelled=True)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                return self._error(call, "tool timed out", timed_out=True)
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task), min(0.05, remaining)
                )
            except TimeoutError:
                continue
        return task.result()

    @staticmethod
    def _error(call: ToolCall, error: str, **metadata) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=False,
            content="",
            error=error,
            metadata=metadata,
        )

    @staticmethod
    def _normalize(call: ToolCall, result: ToolResult) -> ToolResult:
        metadata = dict(result.metadata)
        content = result.content
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            content = content[:MAX_TOOL_OUTPUT_CHARS]
            metadata["truncated"] = True
        return result.model_copy(
            update={
                "tool_call_id": call.id,
                "tool_name": call.name,
                "content": content,
                "metadata": metadata,
            }
        )
