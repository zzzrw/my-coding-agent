from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from coding_agent.tools.models import ToolSchema

from .command import classify_command

PermissionMode = Literal["default", "workspace", "full"]
DecisionKind = Literal["allow", "ask", "deny"]


class PermissionDecision(BaseModel):
    kind: DecisionKind
    reason: str
    category: str
    allow_outside_once: bool = False


class ApprovalPolicy(Protocol):
    def decide(
        self,
        tool: ToolSchema,
        arguments: dict[str, Any],
        *,
        workspace: Path,
        mode: PermissionMode,
    ) -> PermissionDecision: ...


class DefaultApprovalPolicy:
    def decide(self, tool, arguments, *, workspace, mode):
        if tool.name == "run_command":
            classification = classify_command(str(arguments.get("command", "")))
            if classification.catastrophic:
                return PermissionDecision(
                    kind="deny", reason=classification.reason, category="catastrophic"
                )
            if mode == "default":
                return PermissionDecision(
                    kind="ask",
                    reason="shell command requires approval",
                    category="shell",
                )
            if classification.outside_or_unknown:
                return PermissionDecision(
                    kind="ask",
                    reason=classification.reason,
                    category="shell",
                    allow_outside_once=True,
                )
            return PermissionDecision(
                kind="allow", reason="safe workspace command", category="shell"
            )
        mutating = tool.risk_level in {"mutate_file", "mutate_shell", "dangerous"}
        path = arguments.get("path")
        outside = False
        if path:
            try:
                resolved = (
                    (workspace / str(path)).expanduser().resolve()
                    if not Path(str(path)).is_absolute()
                    else Path(str(path)).expanduser().resolve()
                )
                outside = not (
                    resolved == workspace.resolve()
                    or workspace.resolve() in resolved.parents
                )
            except OSError:
                outside = True
        if mode == "default":
            if outside:
                return PermissionDecision(
                    kind="ask",
                    reason="path is outside workspace",
                    category="outside_path",
                    allow_outside_once=True,
                )
            return PermissionDecision(
                kind="ask" if mutating else "allow",
                reason="mutation requires approval" if mutating else "read-only tool",
                category="mutation" if mutating else "read",
            )
        if mode == "workspace" and outside:
            return PermissionDecision(
                kind="ask",
                reason="path is outside workspace",
                category="outside_path",
                allow_outside_once=True,
            )
        return PermissionDecision(
            kind="allow", reason="allowed by permission mode", category="tool"
        )
