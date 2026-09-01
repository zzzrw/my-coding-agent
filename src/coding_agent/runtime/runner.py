import asyncio
import json
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path

from coding_agent.context.policy import ContextPolicy
from coding_agent.llm.protocol import LLMProvider
from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import Message, ToolCall, TurnOutcome, Usage
from coding_agent.session.store import SessionStore
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.models import ToolResult
from coding_agent.tools.registry import PermissionMode, ToolRegistry

EventSink = Callable[[RuntimeEvent], Awaitable[None]]

_PARALLEL_SAFE_TOOLS = frozenset({"read_file", "list_files", "grep_files"})
_MUTATION_TOOLS = frozenset({"write_file", "edit_file"})


def call_schema_parallel_safe(name: str) -> bool:
    """Return whether a tool's calls may run concurrently with other calls."""
    return name in _PARALLEL_SAFE_TOOLS


def partition_waves(
    calls: list[ToolCall],
    workspace: Path | None = None,
) -> list[list[ToolCall]]:
    """Split parsed calls into ordered waves that can run concurrently.

    A call joins the current wave iff it is parallel-safe or its mutation key
    (resolved path for ``write_file``/``edit_file``) is not already used in the
    wave. Calls that are neither parallel-safe nor key-distinct start a new
    wave. The flattened result preserves input call order.
    """
    waves: list[list[ToolCall]] = []
    used_keys: set[str] = set()
    base = workspace or Path.cwd()
    for call in calls:
        key = _mutation_key(call, base)
        can_join = call_schema_parallel_safe(call.name) or (
            key is not None and key not in used_keys
        )
        if can_join and waves and _compatible(waves[-1], call, key, base):
            waves[-1].append(call)
        else:
            waves.append([call])
            used_keys = set()
        if key is not None:
            used_keys.add(key)
    return waves


def _mutation_key(call: ToolCall, workspace: Path) -> str | None:
    """Return the resolved-path mutation key for a call, or None when it has none."""
    if call.name not in _MUTATION_TOOLS:
        return None
    path = call.arguments.get("path")
    if not isinstance(path, str) or not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return str(candidate.resolve())


def _compatible(
    wave: list[ToolCall],
    call: ToolCall,
    key: str | None,
    workspace: Path,
) -> bool:
    """Return whether ``call`` may join ``wave`` without creating a conflict.

    A parallel-safe read joins any wave. A mutating call joins only waves that
    already hold a mutation, so a reads-only wave never absorbs a write while
    distinct-path mutations still batch; same-path collisions are excluded by
    ``can_join`` via ``used_keys``.
    """
    if call_schema_parallel_safe(call.name):
        return True
    return key is not None and any(
        _mutation_key(other, workspace) is not None for other in wave
    )


class AgentRunner:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        executor: ToolExecutor,
        context_policy: ContextPolicy,
        store: SessionStore,
        event_sink: EventSink,
        system_prompt: Message,
        model: str,
        context_window: int,
        permission_mode: PermissionMode = "default",
        max_steps: int = 20,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.executor = executor
        self.context_policy = context_policy
        self.store = store
        self.event_sink = event_sink
        self.system_prompt = system_prompt
        self.model = model
        self.context_window = context_window
        self.permission_mode = permission_mode
        self.max_steps = max_steps

    async def run_turn(
        self,
        prompt: str,
        *,
        run_id: str,
        turn_id: str,
        signal: asyncio.Event,
    ) -> TurnOutcome:
        del prompt
        usage: Usage | None = None
        final_text = ""
        for step in range(1, self.max_steps + 1):
            if signal.is_set():
                return TurnOutcome(reason="aborted", steps=step - 1, usage=usage)
            try:
                view = self.context_policy.prepare(
                    self.store.project_messages(include_open_turn=True),
                    system_prompt=self.system_prompt,
                    context_window=self.context_window,
                    usage=usage,
                )
            except Exception:  # noqa: BLE001 - persistence boundary normalizes errors
                return TurnOutcome(reason="session_error", steps=step - 1, usage=usage)
            await self._emit(
                "context_updated",
                run_id,
                turn_id,
                used_tokens=view.used_tokens,
                context_window=view.context_window,
                estimated=view.estimated,
            )

            message_id = uuid.uuid4().hex
            await self._emit(
                "assistant_started", run_id, turn_id, message_id=message_id
            )
            text_parts: list[str] = []
            calls: OrderedDict[str, dict[str, str]] = OrderedDict()
            finish_reason: str | None = None
            provider_error: str | None = None
            try:
                async for event in self.provider.stream(
                    view.messages,
                    self.registry.schemas(),
                    model=self.model,
                    signal=signal,
                ):
                    if signal.is_set():
                        return TurnOutcome(
                            reason="aborted", steps=step - 1, usage=usage
                        )
                    if event.type == "text_delta":
                        text = event.text or ""
                        text_parts.append(text)
                        await self._emit(
                            "assistant_delta",
                            run_id,
                            turn_id,
                            message_id=message_id,
                            text=text,
                        )
                    elif event.type in {"tool_call_start", "tool_call_delta"}:
                        call_id = event.tool_call_id or f"call-{len(calls)}"
                        current = calls.setdefault(
                            call_id, {"name": event.tool_name or "", "arguments": ""}
                        )
                        if event.tool_name:
                            current["name"] = event.tool_name
                        if event.arguments_delta:
                            current["arguments"] += event.arguments_delta
                    elif event.type in {"tool_call_end", "response_end"}:
                        if event.finish_reason:
                            finish_reason = event.finish_reason
                        usage = event.usage or usage
                    elif event.type == "error":
                        provider_error = event.error or "provider error"
                        break
            except Exception as exc:  # noqa: BLE001
                provider_error = str(exc)

            if provider_error:
                await self._emit(
                    "run_error",
                    run_id,
                    turn_id,
                    code="provider_error",
                    message=provider_error,
                    recoverable=True,
                )
                return TurnOutcome(reason="provider_error", steps=step, usage=usage)

            final_text = "".join(text_parts)
            parsed_calls: list[ToolCall] = []
            invalid_results: list[ToolResult] = []
            for call_id, raw in calls.items():
                try:
                    arguments = json.loads(raw["arguments"] or "{}")
                    if not isinstance(arguments, dict):
                        raise TypeError("tool arguments must be an object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    arguments = {}
                    invalid_results.append(
                        ToolResult(
                            tool_call_id=call_id,
                            tool_name=raw["name"],
                            ok=False,
                            content=f"invalid tool arguments: {exc}",
                            error=f"invalid tool arguments: {exc}",
                        )
                    )
                parsed_calls.append(
                    ToolCall(id=call_id, name=raw["name"], arguments=arguments)
                )

            if calls and finish_reason != "tool_calls":
                await self._emit(
                    "notice",
                    run_id,
                    turn_id,
                    level="error",
                    message="truncated tool call was not executed",
                )
                return TurnOutcome(
                    reason="completed",
                    final_text=final_text,
                    steps=step,
                    usage=usage,
                )

            assistant = Message(
                role="assistant", content=final_text or None, tool_calls=parsed_calls
            )
            try:
                assistant_record = self.store.append_new(
                    "assistant_message",
                    {"message": assistant, "complete": True},
                    run_id=run_id,
                    turn_id=turn_id,
                )
            except Exception:  # noqa: BLE001 - persistence boundary normalizes errors
                return TurnOutcome(reason="session_error", steps=step, usage=usage)
            await self._emit(
                "assistant_finished",
                run_id,
                turn_id,
                message_id=message_id,
                usage=usage.model_dump() if usage else None,
                finish_reason=finish_reason,
            )

            if not parsed_calls:
                return TurnOutcome(
                    reason="completed",
                    final_text=final_text,
                    steps=step,
                    usage=usage,
                )

            if len(parsed_calls) >= 2:
                await self._emit(
                    "plan_preview",
                    run_id,
                    turn_id,
                    tool_calls=[
                        {"name": call.name, "arguments": call.arguments}
                        for call in parsed_calls
                    ],
                )

            invalid_by_id = {result.tool_call_id: result for result in invalid_results}

            async def _run_call(
                call: ToolCall,
                *,
                assistant_record=assistant_record,
                invalid_by_id=invalid_by_id,
            ) -> tuple[str, ToolResult]:
                self.store.append_new(
                    "tool_call",
                    {
                        "tool_call": call,
                        "source_assistant_record_id": assistant_record.id,
                    },
                    run_id=run_id,
                    turn_id=turn_id,
                )
                await self._emit(
                    "tool_started",
                    run_id,
                    turn_id,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                )
                result = invalid_by_id.get(call.id)
                if result is None:
                    result = await self.executor.execute(
                        call,
                        run_id=run_id,
                        workspace=Path(self.store.header.workspace),
                        permission_mode=self.permission_mode,
                        signal=signal,
                        output_sink=await self._tool_output_sink(
                            run_id, turn_id, call.id
                        ),
                    )
                self.store.append_new(
                    "tool_result",
                    {"result": result},
                    run_id=run_id,
                    turn_id=turn_id,
                )
                await self._emit(
                    "tool_finished",
                    run_id,
                    turn_id,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=result.ok,
                    content=result.content,
                    error=result.error,
                    metadata=result.metadata,
                )
                return call.id, result

            for wave in partition_waves(
                parsed_calls, workspace=Path(self.store.header.workspace)
            ):
                wave_results = await asyncio.gather(
                    *(_run_call(call) for call in wave),
                    return_exceptions=True,
                )
                if any(isinstance(item, Exception) for item in wave_results):
                    return TurnOutcome(reason="session_error", steps=step, usage=usage)
        return TurnOutcome(
            reason="max_steps",
            final_text=final_text,
            steps=self.max_steps,
            usage=usage,
        )

    async def _tool_output_sink(self, run_id: str, turn_id: str, call_id: str):
        async def sink(text: str) -> None:
            await self._emit(
                "tool_output_delta",
                run_id,
                turn_id,
                tool_call_id=call_id,
                text=text,
            )

        return sink

    async def _emit(self, event_type, run_id, turn_id, **payload) -> None:
        await self.event_sink(
            RuntimeEvent(
                type=event_type,
                run_id=run_id,
                turn_id=turn_id,
                payload=payload,
            )
        )
