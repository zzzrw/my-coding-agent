"""Application bootstrap and command-line entry point for coding-agent."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from coding_agent.context.truncate import TruncatePolicy
from coding_agent.llm.openai_compatible import OpenAICompatibleProvider
from coding_agent.llm.protocol import LLMProvider
from coding_agent.policy.approval import DefaultApprovalPolicy, PermissionMode
from coding_agent.runtime.models import Message
from coding_agent.runtime.runner import AgentRunner
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore
from coding_agent.tools.executor import ToolExecutor
from coding_agent.tools.filesystem import (
    make_edit_file_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tools.search import make_grep_files_tool, make_list_files_tool
from coding_agent.tools.shell import make_run_command_tool
from coding_agent.tui.app import CodingAgentApp

DEFAULT_MODEL = ""
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_SESSION_DIR = Path.home() / ".coding-agent" / "sessions"
DEFAULT_CREDENTIAL_ENV = ("CODING_AGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
SYSTEM_PROMPT = Message(
    role="system",
    content=(
        "You are coding-agent, an engineering assistant operating in the user's "
        "workspace. Inspect relevant files before changing them, use the provided "
        "tools for workspace operations, verify changes when practical, and give a "
        "concise final response."
    ),
)


class ConfigurationError(ValueError):
    """Raised when required CLI configuration is absent or invalid."""


def _resolve_workspace(workspace: str | Path | None) -> Path:
    path = Path(workspace or Path.cwd()).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"workspace does not exist: {path}")
    if not path.is_dir():
        raise ConfigurationError("workspace is not a directory")
    return path


def _resolve_model(model: str | None) -> str:
    value = (
        model
        or os.getenv("CODING_AGENT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
    )
    if not value:
        raise ConfigurationError("missing model configuration")
    return value


def _resolve_api_key(
    api_key: str | None, credential_env: str | None
) -> tuple[str | None, str | None]:
    if api_key:
        return api_key, credential_env
    names = (credential_env,) if credential_env else DEFAULT_CREDENTIAL_ENV
    for name in names:
        if name:
            value = os.getenv(name)
            if value:
                return value, name
    return None, credential_env or ", ".join(DEFAULT_CREDENTIAL_ENV)


def _resolve_context_window(context_window: int | None) -> int:
    value = context_window if context_window is not None else DEFAULT_CONTEXT_WINDOW
    if value <= 0:
        raise ConfigurationError("context window must be greater than zero")
    return value


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        make_read_file_tool(),
        make_list_files_tool(),
        make_grep_files_tool(),
        make_write_file_tool(),
        make_edit_file_tool(),
        make_run_command_tool(),
    ):
        registry.register(tool)
    return registry


def create_app(
    *,
    workspace: str | Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    credential_env: str | None = None,
    context_window: int | None = None,
    session_dir: str | Path | None = None,
    provider: LLMProvider | None = None,
    permission_mode: PermissionMode = "default",
) -> CodingAgentApp:
    """Build a fully wired Textual app with injectable provider support.

    ``provider`` is intended for deterministic tests and alternative local
    providers. The regular path constructs the OpenAI-compatible provider only
    after required configuration has been resolved.
    """
    resolved_workspace = _resolve_workspace(workspace)
    resolved_model = _resolve_model(model)
    resolved_context_window = _resolve_context_window(context_window)
    resolved_key, _ = _resolve_api_key(api_key, credential_env)
    if provider is None and not resolved_key:
        env_name = credential_env or " or ".join(DEFAULT_CREDENTIAL_ENV)
        raise ConfigurationError(f"missing credential; set {env_name}")

    resolved_base_url = (
        base_url
        or os.getenv("CODING_AGENT_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
    )
    llm_provider = provider or OpenAICompatibleProvider(
        api_key=resolved_key, base_url=resolved_base_url
    )
    resolved_session_dir = Path(session_dir or DEFAULT_SESSION_DIR).expanduser()
    store = SessionStore.create(
        resolved_session_dir,
        workspace=str(resolved_workspace),
        model=resolved_model,
        context_window=resolved_context_window,
    )
    registry = _make_registry()
    approval_policy = DefaultApprovalPolicy()

    def runner_factory(store, context_policy, broker):
        executor = ToolExecutor(registry, approval_policy, broker)
        return AgentRunner(
            provider=llm_provider,
            registry=registry,
            executor=executor,
            context_policy=context_policy,
            store=store,
            event_sink=lambda event: _discard_event(event),
            system_prompt=SYSTEM_PROMPT,
            model=store.header.model,
            context_window=store.header.context_window,
            permission_mode=permission_mode,
        )

    def make_context_policy():
        return TruncatePolicy()

    runtime = AgentRuntime(
        store=store,
        runner_factory=runner_factory,
        context_policy_factory=make_context_policy,
        approval_policy=approval_policy,
        system_prompt=SYSTEM_PROMPT,
        model=resolved_model,
        permission_mode=permission_mode,
    )
    return CodingAgentApp(runtime=runtime)


async def _discard_event(event) -> None:
    del event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Interactive coding assistant for a local workspace.",
    )
    parser.add_argument("--workspace", type=Path, help="workspace directory")
    parser.add_argument("--model", help="model name")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--session-dir", type=Path, help="session storage directory")
    parser.add_argument(
        "--context-window", type=int, help="maximum context window in tokens"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, validate configuration, and launch the TUI."""
    args = build_parser().parse_args(argv)
    try:
        application = create_app(
            workspace=args.workspace,
            model=args.model,
            base_url=args.base_url,
            session_dir=args.session_dir,
            context_window=args.context_window,
        )
    except ConfigurationError:
        print("coding-agent configuration error", file=os.sys.stderr)
        return 2
    application.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
