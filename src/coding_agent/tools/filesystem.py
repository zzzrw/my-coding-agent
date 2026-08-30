import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import ToolResult, ToolSchema
from .registry import PermissionMode


def resolve_tool_path(
    workspace: Path,
    user_path: str,
    *,
    permission_mode: PermissionMode,
    allow_outside_once: bool,
) -> Path:
    path = Path(user_path).expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    root = workspace.resolve()
    inside = resolved == root or root in resolved.parents
    if not inside and permission_mode != "full" and not allow_outside_once:
        raise PermissionError("path is outside the workspace")
    return resolved


def _result(
    name: str, ok: bool, content: str = "", error: str | None = None, **metadata: Any
) -> ToolResult:
    return ToolResult(
        tool_call_id="",
        tool_name=name,
        ok=ok,
        content=content,
        error=error,
        metadata=metadata,
    )


class _ReadArgs(BaseModel):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class _WriteArgs(BaseModel):
    path: str
    content: str


class _EditArgs(BaseModel):
    path: str
    old_text: str
    new_text: str


class _ReadTool:
    args_model = _ReadArgs
    schema = ToolSchema(
        name="read_file",
        description="Read a text file",
        parameters=_ReadArgs.model_json_schema(),
        risk_level="read",
        is_parallel_safe=True,
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            path = resolve_tool_path(
                context.workspace,
                args.path,
                permission_mode=context.permission_mode,
                allow_outside_once=context.allow_outside_once,
            )
            if signal.is_set():
                return _result(self.schema.name, False, error="cancelled")
            lines = path.read_text(encoding="utf-8").splitlines()
            end = args.end_line or len(lines)
            if end < args.start_line:
                return _result(self.schema.name, True, "")
            return _result(
                self.schema.name, True, "\n".join(lines[args.start_line - 1 : end])
            )
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class _WriteTool:
    args_model = _WriteArgs
    schema = ToolSchema(
        name="write_file",
        description="Write a text file",
        parameters=_WriteArgs.model_json_schema(),
        risk_level="mutate_file",
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            path = resolve_tool_path(
                context.workspace,
                args.path,
                permission_mode=context.permission_mode,
                allow_outside_once=context.allow_outside_once,
            )
            if signal.is_set():
                return _result(self.schema.name, False, error="cancelled")
            _atomic_write(path, args.content)
            return _result(self.schema.name, True, f"wrote {path}")
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


class _EditTool:
    args_model = _EditArgs
    schema = ToolSchema(
        name="edit_file",
        description="Replace exact text in a file",
        parameters=_EditArgs.model_json_schema(),
        risk_level="mutate_file",
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            path = resolve_tool_path(
                context.workspace,
                args.path,
                permission_mode=context.permission_mode,
                allow_outside_once=context.allow_outside_once,
            )
            text = path.read_text(encoding="utf-8")
            count = text.count(args.old_text)
            if count != 1:
                return _result(
                    self.schema.name, False, error="old_text must match exactly once"
                )
            if signal.is_set():
                return _result(self.schema.name, False, error="cancelled")
            _atomic_write(path, text.replace(args.old_text, args.new_text, 1))
            return _result(self.schema.name, True, f"edited {path}")
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


def make_read_file_tool():
    return _ReadTool()


def make_write_file_tool():
    return _WriteTool()


def make_edit_file_tool():
    return _EditTool()
