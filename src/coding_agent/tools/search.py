from pathlib import Path

from pydantic import BaseModel, Field

from .filesystem import _result, resolve_tool_path
from .models import ToolSchema

SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    "target",
}


class _ListArgs(BaseModel):
    path: str = "."
    max_entries: int = Field(default=200, ge=1, le=2000)


class _GrepArgs(BaseModel):
    pattern: str
    path: str = "."
    max_results: int = Field(default=100, ge=1, le=1000)


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_symlink():
            continue
        yield path


class _ListTool:
    args_model = _ListArgs
    schema = ToolSchema(
        name="list_files",
        description="List workspace files",
        parameters=_ListArgs.model_json_schema(),
        risk_level="read",
        is_parallel_safe=True,
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            root = resolve_tool_path(
                context.workspace,
                args.path,
                permission_mode=context.permission_mode,
                allow_outside_once=context.allow_outside_once,
            )
            entries = [
                str(p.relative_to(context.workspace))
                for p in _files(root)
                if p.is_file()
            ][: args.max_entries]
            return _result(
                self.schema.name,
                True,
                "\n".join(entries),
                truncated=len(entries) >= args.max_entries,
            )
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


class _GrepTool:
    args_model = _GrepArgs
    schema = ToolSchema(
        name="grep_files",
        description="Search text files",
        parameters=_GrepArgs.model_json_schema(),
        risk_level="read",
        is_parallel_safe=True,
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            root = resolve_tool_path(
                context.workspace,
                args.path,
                permission_mode=context.permission_mode,
                allow_outside_once=context.allow_outside_once,
            )
            matches = []
            for path in _files(root):
                if signal.is_set():
                    return _result(self.schema.name, False, error="cancelled")
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                for n, line in enumerate(text.splitlines(), 1):
                    if args.pattern in line:
                        matches.append(
                            f"{path.relative_to(context.workspace)}:{n}:{line}"
                        )
                        if len(matches) >= args.max_results:
                            break
                if len(matches) >= args.max_results:
                    break
            return _result(
                self.schema.name,
                True,
                "\n".join(matches),
                truncated=len(matches) >= args.max_results,
            )
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


def make_list_files_tool():
    return _ListTool()


def make_grep_files_tool():
    return _GrepTool()
