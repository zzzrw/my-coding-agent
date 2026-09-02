"""Application bootstrap and command-line entry point for coding-agent."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Input, Static

from coding_agent.config.config import (
    Config,
    ConfigurationError,
    default_user_config_path,
    load_config,
    save_config,
)
from coding_agent.context.truncate import TruncatePolicy
from coding_agent.llm.openai_compatible import OpenAICompatibleProvider
from coding_agent.llm.protocol import LLMProvider
from coding_agent.policy.approval import DefaultApprovalPolicy, PermissionMode
from coding_agent.policy.memory import DecisionMemory
from coding_agent.runtime.models import Message
from coding_agent.runtime.runner import AgentRunner
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore
from coding_agent.skills.catalog import format_catalog
from coding_agent.skills.discovery import discover_skills
from coding_agent.skills.models import Skill
from coding_agent.skills.tool import make_load_skill_tool
from coding_agent.tools.executor import MutationJournal, ToolExecutor
from coding_agent.tools.filesystem import (
    make_clear_directory_tool,
    make_edit_file_tool,
    make_read_file_tool,
    make_remove_file_tool,
    make_write_file_tool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tools.search import make_grep_files_tool, make_list_files_tool
from coding_agent.tools.shell import make_run_command_tool
from coding_agent.tui.app import CodingAgentApp

DEFAULT_MODEL = ""
DEFAULT_CONTEXT_WINDOW = 1_000_000
DEFAULT_SESSION_DIR = Path.home() / ".coding-agent" / "sessions"
DEFAULT_CREDENTIAL_ENV = ("CODING_AGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
ONBOARDING_MODEL_ENVS = ("CODING_AGENT_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL")
ONBOARDING_BASE_URL_ENVS = (
    "CODING_AGENT_BASE_URL",
    "OPENAI_BASE_URL",
    "DEEPSEEK_BASE_URL",
)


_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "hi": "Hindi",
}


def _language_name(language: str) -> str:
    """Map a config language code to a display name (fallback: the code)."""
    name = _LANGUAGE_NAMES.get((language or "").strip().lower())
    return name or (language or "zh")


def build_system_prompt(
    workspace: Path,
    permission_mode: PermissionMode,
    *,
    skills: Sequence[Skill] = (),
    language: str = "zh",
) -> Message:
    """Return the system prompt embedding the safety boundaries for a run.

    ``skills`` is the already-discovered skill catalog; when empty no catalog
    section is emitted, so a skill-less run stays byte-identical to prior
    builds. Only the one-line catalog is appended — never a SKILL.md body.
    ``language`` is the preferred reply language code (e.g. ``"zh"``); the
    prompt instructs the model to reply in it.
    """
    section = format_catalog(skills)
    fixed = dedent(
        f"""\
        You are coding-agent, a careful senior engineer pairing with the user inside their
        workspace. Inspect before you change, edit through tools, and verify with real runs.

        How you work
        - Gather context first. Read the relevant files and trace symbols with read_file /
          grep_files / list_files before changing anything. Never invent files, symbols,
          APIs, or imports you have not seen in the repo.
        - Prefer the dedicated tools over shell: read_file not cat/head/tail, edit_file /
          write_file not sed/awk/heredocs. Edit through tools, not chat — never print code
          blocks as a substitute for editing. Batch independent reads and searches in one
          turn instead of one at a time.
        - Plan briefly before multi-step work: state the steps, then act. Keep changes
          minimal and focused — no drive-by refactors, renames, or added comments unless
          asked. Fix root causes, not symptoms.
        - Work incrementally. After an edit, verify when practical (run the tests, build,
          or a small check); if you cannot verify, say so instead of claiming success.
        - If an edit fails, re-read the file for its current exact contents before
          retrying — never repeat a stale patch. If a region fails twice, stop and ask
          rather than looping.
        - Every response either makes progress with tool calls or delivers the final
          result. Keep working until the task is actually resolved; do not end on a stub,
          a plan, or an intent-only summary.

        Final response
        - Lead with the change or answer, then the essentials — no preamble or restating
          the request. Reference code as path:line, not by pasting whole files. Offer the
          next step (run tests, commit) only if relevant.

        If skills are listed below, consult load_skill for the matching one when a task
        fits it.

        Permission boundaries
        - Hard-denied in every mode (never attempt and do not work around): rm of /, ~,
          $HOME, /root, or a whole /home/<user>; git push --force; git reset --hard;
          git clean -f; mkfs/fdisk/shutdown/reboot/poweroff.
        - For deletions prefer the workspace-bounded remove_file and clear_directory tools,
          and prefer relative in-workspace paths.
        - Silencing command output with "> /dev/null" is discouraged; redirect to a log
          file or pipe with 2>&1.
        - Active permission mode is "{permission_mode}": in "default" every shell command needs
          approval; in "workspace"/"full" only the hard-denied commands above are rejected.
        - A dev server started with "&" persists across tool calls; inspect it with pgrep
          and stop it with pkill or kill.
        - Workspace root: {workspace}"""
    )
    lead = f"Respond in {_language_name(language)}."
    return Message(
        role="system",
        content=lead + "\n\n" + fixed + (("\n\n" + section) if section else ""),
    )


# ``ConfigurationError`` lives in the config module and is re-exported here so
# both callers (and tests) can import it from ``coding_agent.app``.


class MissingConfiguration(ConfigurationError):
    """Raised when interactive launch configuration (model or credential) is absent."""


def _resolve_workspace(workspace: str | Path | None) -> Path:
    path = Path(workspace or Path.cwd()).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"workspace does not exist: {path}")
    if not path.is_dir():
        raise ConfigurationError("workspace is not a directory")
    return path


def _first_env(names: tuple[str, ...]) -> str | None:
    """Return the first environment variable in ``names`` that is set."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _resolve_model(model: str | None, config: Config | None = None) -> str:
    value = (
        model or _first_env(ONBOARDING_MODEL_ENVS) or (config.model if config else None)
    )
    if not value:
        raise MissingConfiguration("missing model configuration")
    return value


def _resolve_api_key(
    api_key: str | None,
    credential_env: str | None,
    config: Config | None = None,
) -> tuple[str | None, str | None]:
    if api_key:
        return api_key, credential_env
    names = (credential_env,) if credential_env else DEFAULT_CREDENTIAL_ENV
    for name in names:
        if name:
            value = os.getenv(name)
            if value:
                return value, name
    if config is not None and config.api_key:
        return config.api_key, None
    return None, credential_env or ", ".join(DEFAULT_CREDENTIAL_ENV)


def _resolve_base_url(base_url: str | None, config: Config | None = None) -> str | None:
    return (
        base_url
        or _first_env(ONBOARDING_BASE_URL_ENVS)
        or (config.base_url if config else None)
    )


def _resolve_context_window(
    context_window: int | None, config: Config | None = None
) -> int:
    value = context_window
    if value is None:
        value = (
            config.context_window
            if config is not None and config.context_window
            else DEFAULT_CONTEXT_WINDOW
        )
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
        make_remove_file_tool(),
        make_clear_directory_tool(),
        make_load_skill_tool(),
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
    permission_mode: PermissionMode = "workspace",
    language: str | None = None,
    config: Config | None = None,
) -> CodingAgentApp:
    """Build a fully wired Textual app with injectable provider support.

    ``provider`` is intended for deterministic tests and alternative local
    providers. The regular path constructs the OpenAI-compatible provider only
    after required configuration has been resolved.

    ``config`` supplies fallback values (model, key, base URL, context window,
    optional ``max_steps`` cap) used after explicit args and environment
    variables. When omitted, the config file is loaded lazily only if some
    field is otherwise unresolved, so a fully environment-configured launch
    never touches the filesystem.
    """
    resolved_workspace = _resolve_workspace(workspace)

    def _load_config() -> Config:
        if config is not None:
            return config
        key_names = (credential_env,) if credential_env else DEFAULT_CREDENTIAL_ENV
        needs_config = (
            not (model or _first_env(ONBOARDING_MODEL_ENVS))
            or not (api_key or _first_env(key_names))
            or not (base_url or _first_env(ONBOARDING_BASE_URL_ENVS))
            or context_window is None
        )
        if not needs_config:
            return Config()
        return load_config(workspace=resolved_workspace)

    cfg = _load_config()
    resolved_model = _resolve_model(model, cfg)
    resolved_context_window = _resolve_context_window(context_window, cfg)
    if cfg.max_steps is not None and cfg.max_steps <= 0:
        raise ConfigurationError("max_steps must be greater than zero")
    resolved_key, _ = _resolve_api_key(api_key, credential_env, cfg)
    if provider is None and not resolved_key:
        env_name = credential_env or " or ".join(DEFAULT_CREDENTIAL_ENV)
        raise MissingConfiguration(f"missing credential; set {env_name}")

    resolved_base_url = _resolve_base_url(base_url, cfg)
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
    journal = MutationJournal()
    skills = discover_skills(resolved_workspace)
    resolved_language = language or cfg.language or "zh"
    system_prompt = build_system_prompt(
        resolved_workspace, permission_mode, skills=skills, language=resolved_language
    )

    def runner_factory(store, context_policy, broker):
        executor = ToolExecutor(
            registry, approval_policy, broker, memory=DecisionMemory(), journal=journal
        )
        return AgentRunner(
            provider=llm_provider,
            registry=registry,
            executor=executor,
            context_policy=context_policy,
            store=store,
            event_sink=lambda event: _discard_event(event),
            system_prompt=system_prompt,
            model=store.header.model,
            context_window=store.header.context_window,
            permission_mode=permission_mode,
            max_steps=cfg.max_steps,
        )

    def make_context_policy():
        return TruncatePolicy()

    runtime = AgentRuntime(
        store=store,
        runner_factory=runner_factory,
        context_policy_factory=make_context_policy,
        approval_policy=approval_policy,
        system_prompt=system_prompt,
        model=resolved_model,
        permission_mode=permission_mode,
    )
    return CodingAgentApp(runtime=runtime)


async def _discard_event(event) -> None:
    del event


def onboarding_guidance() -> str:
    """Return actionable, credential-free configuration instructions."""
    model_envs = " or ".join(ONBOARDING_MODEL_ENVS)
    key_envs = " or ".join(DEFAULT_CREDENTIAL_ENV)
    base_url_envs = " or ".join(ONBOARDING_BASE_URL_ENVS)
    return (
        "coding-agent could not start: model or API key configuration is missing.\n"
        "\n"
        "Required environment variables:\n"
        f"  model:   {model_envs}\n"
        f"  api key: {key_envs}\n"
        "\n"
        "Optional settings:\n"
        f"  base url: {base_url_envs}\n"
        "  context window: --context-window <tokens>"
        f" (default {DEFAULT_CONTEXT_WINDOW})\n"
        "\n"
        "Configure the variables and relaunch, for example:\n"
        "  export CODING_AGENT_MODEL=your-model\n"
        "  export CODING_AGENT_API_KEY=your-key\n"
        "  coding-agent --workspace .\n"
        "\n"
        "Or run coding-agent from a terminal to launch the interactive setup\n"
        "wizard, which writes the same settings to a local 0600 config file.\n"
        "The API key is stored only in that file and is never printed or logged.\n"
        "Press q or Ctrl+C to exit."
    )


class ConfigurationScreen(App[None]):
    """Standalone screen shown when interactive launch configuration is absent."""

    BINDINGS: ClassVar = [
        Binding("q", "exit_app", "Quit", priority=True),
        Binding("ctrl+c", "exit_app", "Quit", priority=True),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Static(self._message, id="onboarding", markup=False)

    def action_exit_app(self) -> None:
        self.exit()


class SetupScreen(App[None]):
    """Interactive first-run wizard that writes the user config file.

    A labeled form (model, optional base url, password-masked api key, context
    window) with a ``Save`` button. Saving validates model and key, writes the
    config via :func:`save_config` to :func:`default_user_config_path`, and
    exits with ``True``. The api key input is password-masked so the key is
    never echoed on screen; it is persisted only to the ``0600`` config file.
    """

    BINDINGS: ClassVar = [
        Binding("q", "exit_app", "Quit", priority=True),
        Binding("ctrl+c", "exit_app", "Quit", priority=True),
    ]

    def __init__(self, message: str | None = None) -> None:
        super().__init__()
        self._message = message if message is not None else onboarding_guidance()
        self._model_input = Input(placeholder="model (e.g. deepseek-chat)", id="model")
        self._base_url_input = Input(placeholder="base url (optional)", id="base_url")
        self._key_input = Input(placeholder="api key", id="api_key", password=True)
        self._window_input = Input(
            placeholder="context window (optional)", id="context_window"
        )
        self._save_button = Button("Save", id="save", variant="primary")

    def compose(self) -> ComposeResult:
        yield Static("coding-agent first-run setup", id="setup-title", markup=False)
        yield Static(self._message, id="setup-guidance", markup=False)
        yield Static("Model", id="model-label", markup=False)
        yield self._model_input
        yield Static("Base URL (optional)", id="base-url-label", markup=False)
        yield self._base_url_input
        yield Static(
            "API key (stored locally, never echoed)", id="api-key-label", markup=False
        )
        yield self._key_input
        yield Static(
            "Context window (optional)", id="context-window-label", markup=False
        )
        yield self._window_input
        yield self._save_button

    def action_exit_app(self) -> None:
        self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "save":
            self._save()

    def _save(self) -> None:
        """Validate the form, write the config file, and exit on success."""
        model = self._model_input.value.strip()
        api_key = self._key_input.value.strip()
        if not model or not api_key:
            self.notify("model and api key are required")
            return
        window_text = self._window_input.value.strip()
        context_window = 0
        if window_text:
            try:
                context_window = int(window_text)
            except ValueError:
                self.notify("context window must be an integer")
                return
        config = Config(
            model=model,
            api_key=api_key,
            base_url=self._base_url_input.value.strip() or "",
            context_window=context_window,
        )
        save_config(default_user_config_path(), config)
        self.exit(True)


def _run_onboarding(message: str | None = None, interactive: bool = True) -> int:
    """Run the first-run setup wizard or fall back to the static guidance screen.

    On an interactive TTY the ``SetupScreen`` form walks the user through
    configuration and writes the config file; otherwise the static
    ``ConfigurationScreen`` presents the credential-free guidance text.
    """
    message = message if message is not None else onboarding_guidance()
    if interactive and sys.stdin.isatty():
        SetupScreen(message).run()
    else:
        ConfigurationScreen(message).run()
    return 0


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
    parser.add_argument("--config", type=Path, help="path to a TOML configuration file")
    return parser


def resolve_config(args, workspace: Path) -> Config:
    """Resolve the effective configuration for parsed CLI ``args``.

    Loads the user/workspace TOML (``args.config`` overrides the user path) and
    overlays CLI-provided fields so explicit flags keep priority over the file.
    """
    config = load_config(user_path=args.config, workspace=workspace)
    overrides: dict[str, object] = {}
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.context_window:
        overrides["context_window"] = args.context_window
    return config.model_copy(update=overrides) if overrides else config


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, validate configuration, and launch the TUI."""
    args = build_parser().parse_args(argv)
    try:
        workspace = _resolve_workspace(args.workspace)
        config = resolve_config(args, workspace)
        application = create_app(
            workspace=workspace,
            model=args.model,
            base_url=args.base_url,
            session_dir=args.session_dir,
            context_window=args.context_window,
            config=config,
        )
    except MissingConfiguration:
        return _run_onboarding()
    except ConfigurationError:
        print("coding-agent configuration error", file=os.sys.stderr)
        return 2
    application.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
