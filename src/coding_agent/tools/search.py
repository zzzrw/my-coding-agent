import os
from itertools import islice
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
MAX_GREP_LINE_CHARS = 2_000
MAX_GREP_FILE_CHARS = 200_000
MAX_GREP_OUTPUT_CHARS = 20_000


class _ListArgs(BaseModel):
    path: str = "."
    max_entries: int = Field(default=200, ge=1, le=2000)
    recursive: bool = False


class _GrepArgs(BaseModel):
    pattern: str
    path: str = "."
    max_results: int = Field(default=100, ge=1, le=1000)
    include: str | None = None


def _files(root: Path):
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if name not in SKIP_DIRS)
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_symlink():
                yield path


def _is_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            sample = stream.read(4096)
        if b"\x00" in sample:
            return False
        sample.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return True


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
            workspace = context.workspace.resolve()
            paths = (
                _files(root)
                if args.recursive
                else sorted(
                    p for p in root.iterdir() if p.is_file() and not p.is_symlink()
                )
            )
            eligible = (p for p in paths if p.is_file())
            entries = [
                str(p.relative_to(workspace))
                if workspace in p.resolve().parents
                else str(p)
                for p in islice(eligible, args.max_entries + 1)
            ]
            truncated = len(entries) > args.max_entries
            entries = entries[: args.max_entries]
            return _result(
                self.schema.name,
                True,
                "\n".join(entries),
                truncated=truncated,
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
            output_chars = 0
            truncated = False
            for path in _files(root):
                if signal.is_set():
                    return _result(self.schema.name, False, error="cancelled")
                if not path.is_file():
                    continue
                if args.include and not path.match(args.include):
                    continue
                if not _is_text_file(path):
                    continue
                try:
                    stream = path.open(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                read_chars = 0
                with stream:
                    line_number = 0
                    while True:
                        raw_line = stream.readline(MAX_GREP_LINE_CHARS + 1)
                        if not raw_line:
                            break
                        line_number += 1
                        read_chars += len(raw_line)
                        if read_chars > MAX_GREP_FILE_CHARS:
                            truncated = True
                            break
                        line_truncated = (
                            not raw_line.endswith(("\n", "\r"))
                            and len(raw_line) > MAX_GREP_LINE_CHARS
                        )
                        while line_truncated and not raw_line.endswith(("\n", "\r")):
                            remainder = stream.readline(MAX_GREP_LINE_CHARS + 1)
                            if not remainder:
                                break
                            read_chars += len(remainder)
                            if remainder.endswith(("\n", "\r")):
                                break
                            if read_chars > MAX_GREP_FILE_CHARS:
                                break
                        line = raw_line.rstrip("\r\n")
                        display_line = line[:MAX_GREP_LINE_CHARS]
                        if args.pattern not in line:
                            truncated = truncated or line_truncated
                            continue
                        try:
                            display_path = str(
                                path.relative_to(context.workspace.resolve())
                            )
                        except ValueError:
                            display_path = str(path)
                        match = f"{display_path}:{line_number}:{display_line}"
                        if output_chars + len(match) > MAX_GREP_OUTPUT_CHARS:
                            truncated = True
                            break
                        matches.append(match)
                        output_chars += len(match) + 1
                        truncated = truncated or line_truncated
                        if len(matches) > args.max_results:
                            break
                if truncated or len(matches) > args.max_results:
                    break
            truncated = truncated or len(matches) > args.max_results
            matches = matches[: args.max_results]
            return _result(
                self.schema.name,
                True,
                "\n".join(matches),
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


def make_list_files_tool():
    return _ListTool()


def make_grep_files_tool():
    return _GrepTool()
