import asyncio
import os
import signal as signal_mod
import time

from pydantic import BaseModel, Field

from .filesystem import _result
from .models import ToolSchema

MAX_COMMAND_OUTPUT_BYTES = 20_000


class _ShellArgs(BaseModel):
    command: str
    timeout_seconds: float = Field(default=120, gt=0, le=300)


class _ShellTool:
    args_model = _ShellArgs
    schema = ToolSchema(
        name="run_command",
        description="Run a shell command",
        parameters=_ShellArgs.model_json_schema(),
        risk_level="mutate_shell",
    )

    async def execute(self, arguments, *, context, signal):
        try:
            args = self.args_model.model_validate(arguments)
            proc = await asyncio.create_subprocess_shell(
                args.command,
                cwd=context.workspace,
                executable="/bin/sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.monotonic()
            collect = asyncio.create_task(self._collect_output(proc))
            while not collect.done():
                if time.monotonic() - started >= args.timeout_seconds:
                    os.killpg(proc.pid, signal_mod.SIGTERM)
                    try:
                        await asyncio.wait_for(asyncio.shield(collect), 1.0)
                    except TimeoutError:
                        os.killpg(proc.pid, signal_mod.SIGKILL)
                        await asyncio.shield(collect)
                    return _result(
                        self.schema.name,
                        False,
                        error="command timed out",
                        exit_code=proc.returncode,
                    )
                if signal.is_set():
                    os.killpg(proc.pid, signal_mod.SIGTERM)
                    try:
                        await asyncio.wait_for(asyncio.shield(collect), 1.0)
                    except TimeoutError:
                        os.killpg(proc.pid, signal_mod.SIGKILL)
                        await asyncio.shield(collect)
                    return _result(
                        self.schema.name,
                        False,
                        error="cancelled",
                        exit_code=proc.returncode,
                    )
                try:
                    await asyncio.wait_for(asyncio.shield(collect), 0.05)
                except TimeoutError:
                    pass
            out, truncated = collect.result()
            await self._terminate_remaining_group(proc.pid)
            return _result(
                self.schema.name,
                proc.returncode == 0,
                out.decode(errors="replace"),
                error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
                exit_code=proc.returncode,
                elapsed_seconds=time.monotonic() - started,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            return _result(self.schema.name, False, error=str(exc))

    @staticmethod
    async def _collect_output(proc) -> tuple[bytes, bool]:
        output = bytearray()
        truncated = False
        assert proc.stdout is not None
        while chunk := await proc.stdout.read(4096):
            remaining = MAX_COMMAND_OUTPUT_BYTES - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
        await proc.wait()
        return bytes(output), truncated

    @staticmethod
    async def _terminate_remaining_group(process_group_id: int) -> None:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        os.killpg(process_group_id, signal_mod.SIGTERM)
        await asyncio.sleep(0.05)
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return
        os.killpg(process_group_id, signal_mod.SIGKILL)


def make_run_command_tool():
    return _ShellTool()
