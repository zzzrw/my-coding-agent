"""The ``load_skill`` model tool."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from coding_agent.tools.models import ToolResult, ToolSchema

from .discovery import resolve_skill, skill_body
from .models import MAX_SKILL_CONTENT_CHARS

_TRUNCATION_NOTE = "\n\n[skill body truncated at 16000 characters]"
_MAX_LISTED_FILES = 50


def _bundled_files(skill_dir: Path) -> tuple[list[str], bool]:
    """Sorted relative paths of files bundled with a skill, ``SKILL.md`` excluded.

    Only names are listed (never contents); any content the model needs is
    read or executed later through the existing ``read_file``/``run_command``
    tools, which stay subject to the normal permission model. The listing is a
    deterministic function of the skill directory and is capped so a huge tree
    cannot bloat the tool result.
    """
    top_level_skill = skill_dir / "SKILL.md"
    try:
        files = [
            path.relative_to(skill_dir).as_posix()
            for path in skill_dir.rglob("*")
            if path.is_file() and path != top_level_skill
        ]
    except OSError:
        return [], False
    files.sort()
    truncated = len(files) > _MAX_LISTED_FILES
    return files[:_MAX_LISTED_FILES], truncated


class _LoadSkillArgs(BaseModel):
    skill: str


def _result(tool_name, ok, content="", error=None, **metadata) -> ToolResult:
    return ToolResult(
        tool_call_id="",
        tool_name=tool_name,
        ok=ok,
        content=content,
        error=error,
        metadata=metadata,
    )


class _LoadSkillTool:
    args_model = _LoadSkillArgs
    schema = ToolSchema(
        name="load_skill",
        description=(
            "Load the body of an installed skill by name. Available skills are "
            "listed in the system prompt under 'Available skills'; call this "
            "before acting when the user names a skill or the task matches a "
            "listed description. The result's metadata includes the skill "
            "directory ('skill_dir') and the sorted list of files bundled with "
            "it ('files'); follow the skill's instructions, reading or running "
            "those helper files with the existing read_file/run_command tools "
            "(subject to the normal permission checks)."
        ),
        parameters=_LoadSkillArgs.model_json_schema(),
        risk_level="read",
        is_parallel_safe=True,
    )

    def __init__(self, *, user_root: Path | None = None) -> None:
        self._user_root = user_root

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            skill = resolve_skill(
                args.skill, context.workspace, user_root=self._user_root
            )
            if skill is None:
                return _result(
                    self.schema.name,
                    False,
                    error=f"unknown skill: {args.skill}",
                )
            if signal.is_set():
                return _result(self.schema.name, False, error="cancelled")
            body = skill_body(skill)
            truncated = len(body) > MAX_SKILL_CONTENT_CHARS
            if truncated:
                body = body[:MAX_SKILL_CONTENT_CHARS] + _TRUNCATION_NOTE
            skill_dir = skill.path.parent
            files, files_truncated = _bundled_files(skill_dir)
            return _result(
                self.schema.name,
                True,
                body,
                name=skill.name,
                path=str(skill.path),
                skill_dir=str(skill_dir),
                files=files,
                files_truncated=files_truncated,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))


def make_load_skill_tool(user_root: Path | None = None):
    return _LoadSkillTool(user_root=user_root)
