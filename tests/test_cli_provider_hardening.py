"""Hardening tests for CLI defaults, onboarding, and provider error redaction."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fakes import FakeProvider

from coding_agent import app as app_module
from coding_agent.app import (
    DEFAULT_CONTEXT_WINDOW,
    ConfigurationError,
    MissingConfiguration,
    create_app,
    onboarding_guidance,
)
from coding_agent.context.truncate import TruncatePolicy
from coding_agent.llm.openai_compatible import OpenAICompatibleProvider, redact_secrets
from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.runtime.models import Message
from coding_agent.runtime.runtime import AgentRuntime
from coding_agent.session.store import SessionStore

_MODEL_ENVS = ("CODING_AGENT_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL")
_KEY_ENVS = ("CODING_AGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
_BASE_URL_ENVS = ("CODING_AGENT_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")
_ALL_ENVS = (*_MODEL_ENVS, *_KEY_ENVS, *_BASE_URL_ENVS)


def _clear_env(monkeypatch, names=_ALL_ENVS):
    for name in names:
        monkeypatch.delenv(name, raising=False)


# --- context-window default resolution -----------------------------------


def test_default_context_window_is_exactly_one_million(tmp_path):
    assert DEFAULT_CONTEXT_WINDOW == 1_000_000
    application = create_app(workspace=tmp_path, model="fake", provider=FakeProvider([]))
    assert application.runtime.store.header.context_window == 1_000_000
    assert application.runtime._runner.context_window == 1_000_000


def test_explicit_context_window_is_preserved_in_header_and_factory(tmp_path):
    application = create_app(
        workspace=tmp_path,
        model="fake",
        context_window=2_000,
        provider=FakeProvider([]),
    )
    assert application.runtime.store.header.context_window == 2_000
    assert application.runtime._runner.context_window == 2_000


def test_non_positive_context_window_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="greater than zero"):
        create_app(
            workspace=tmp_path, model="fake", context_window=0, provider=FakeProvider([])
        )


# --- onboarding / configuration screen ------------------------------------


def test_missing_model_raises_missing_configuration(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(MissingConfiguration, match="model"):
        create_app(workspace=tmp_path)


def test_missing_credential_raises_missing_configuration(tmp_path, monkeypatch):
    _clear_env(monkeypatch, _KEY_ENVS)
    with pytest.raises(MissingConfiguration, match="credential"):
        create_app(workspace=tmp_path, model="fake")


def test_onboarding_guidance_lists_env_vars_settings_and_exit():
    text = onboarding_guidance()
    for env in (*_MODEL_ENVS, *_KEY_ENVS, *_BASE_URL_ENVS):
        assert env in text
    assert "--context-window" in text
    assert "exit" in text.lower()
    assert "ctrl+c" in text.lower()


def test_main_launches_onboarding_when_model_missing(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    ran = []

    class FakeScreen:
        def __init__(self, message):
            self.message = message

        def run(self):
            ran.append(self.message)

    monkeypatch.setattr(app_module, "ConfigurationScreen", FakeScreen)
    assert app_module.main(["--workspace", str(tmp_path)]) == 0
    assert len(ran) == 1
    assert "CODING_AGENT_MODEL" in ran[0]
    assert "CODING_AGENT_API_KEY" in ran[0]


def test_main_launches_onboarding_when_credential_missing(tmp_path, monkeypatch):
    _clear_env(monkeypatch, _KEY_ENVS)
    ran = []

    class FakeScreen:
        def __init__(self, message):
            self.message = message

        def run(self):
            ran.append(self.message)

    monkeypatch.setattr(app_module, "ConfigurationScreen", FakeScreen)
    assert (
        app_module.main(["--workspace", str(tmp_path), "--model", "fake-model"]) == 0
    )
    assert len(ran) == 1
    assert "CODING_AGENT_API_KEY" in ran[0]


def test_help_is_credential_free():
    help_text = app_module.build_parser().format_help()
    assert "API_KEY" not in help_text
    assert "api key" not in help_text.lower()
    assert "credential" not in help_text.lower()


@pytest.mark.asyncio
async def test_configuration_screen_renders_guidance_without_secrets():
    screen = app_module.ConfigurationScreen(onboarding_guidance())
    async with screen.run_test() as pilot:
        await pilot.pause()
        rendered = str(screen.query_one("#onboarding").render())
    assert "CODING_AGENT_MODEL" in rendered
    assert "CODING_AGENT_API_KEY" in rendered
    assert "CODING_AGENT_BASE_URL" in rendered
    assert "--context-window" in rendered
    assert "exit" in rendered.lower()


def test_injectable_provider_bypasses_credential_requirement(tmp_path, monkeypatch):
    _clear_env(monkeypatch, _KEY_ENVS)
    application = create_app(workspace=tmp_path, model="fake", provider=FakeProvider([]))
    assert application.runtime is not None


# --- provider error redaction ---------------------------------------------


@pytest.mark.asyncio
async def test_provider_redacts_configured_api_key_from_error():
    class Completions:
        async def create(self, **kwargs):
            raise RuntimeError("401 invalid api key sk-super-secret-key-value")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    provider = OpenAICompatibleProvider(
        api_key="sk-super-secret-key-value", client=client
    )
    events = [
        event
        async for event in provider.stream([], [], model="m", signal=asyncio.Event())
    ]
    assert events[-1].type == "error"
    assert "sk-super-secret-key-value" not in events[-1].error
    assert "REDACTED" in events[-1].error
    assert "401 invalid api key" in events[-1].error


def test_redact_secrets_removes_bearer_and_url_credentials():
    text = (
        "request failed: GET https://api.example.com/v1 "
        "Authorization: Bearer abc.def.ghi-123 "
        "via https://alice:topsecret@proxy.internal/v1?api_key=sk-live-999"
    )
    redacted = redact_secrets(text)
    assert "abc.def.ghi-123" not in redacted
    assert "topsecret" not in redacted
    assert "sk-live-999" not in redacted
    assert "api.example.com" in redacted
    assert "proxy.internal" in redacted


def test_redact_secrets_preserves_non_secret_diagnostics():
    text = "connection timed out after 30s while retrying model=deepseek-chat"
    assert redact_secrets(text) == text


@pytest.mark.asyncio
async def test_runtime_redacts_secrets_before_persist_and_emit(tmp_path):
    store = SessionStore.create(
        tmp_path / "sessions",
        workspace=str(tmp_path),
        model="fake",
        context_window=1000,
    )

    class RaisingRunner:
        event_sink = None
        permission_mode = "default"

        async def run_turn(self, prompt, *, run_id, turn_id, signal):
            raise RuntimeError(
                "connection refused https://bob:hunter2@host/v1?key=sekrit-42"
            )

    runtime = AgentRuntime(
        store=store,
        runner_factory=lambda *_: RaisingRunner(),
        context_policy_factory=lambda: TruncatePolicy(1000),
        approval_policy=DefaultApprovalPolicy(),
        system_prompt=Message(role="system", content="system"),
        model="fake",
    )
    events = []

    async def sink(event):
        events.append(event)

    runtime.subscribe(sink)
    await runtime.submit("hello")
    await asyncio.sleep(0.05)

    error_event = next(event for event in events if event.type == "run_error")
    assert "hunter2" not in error_event.payload["message"]
    assert "sekrit-42" not in error_event.payload["message"]
    assert "connection refused" in error_event.payload["message"]

    turn_end = next(
        record for record in runtime.store.records() if record.type == "turn_end"
    )
    serialized = json.dumps(turn_end.payload)
    assert "hunter2" not in serialized
    assert "sekrit-42" not in serialized
