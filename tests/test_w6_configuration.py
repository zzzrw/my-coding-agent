"""W6 Task 1 + Task 2 tests: TOML config module and config-aware resolution.

All API keys used here are fake placeholders; real keys never appear in tests.
"""

import os
from pathlib import Path

import pytest
from textual.widgets import Static

from coding_agent import app as app_module
from coding_agent.app import (
    DEFAULT_CONTEXT_WINDOW,
    ConfigurationError,
    MissingConfiguration,
    create_app,
)
from coding_agent.config.config import (
    Config,
    config_dir,
    default_user_config_path,
    load_config,
    save_config,
)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(
        path,
        Config(model="m", api_key="k", base_url="https://x", context_window=1000),
    )
    loaded = load_config(user_path=path)
    assert loaded.model == "m" and loaded.api_key == "k"
    assert loaded.base_url == "https://x" and loaded.context_window == 1000


def test_load_merges_user_then_workspace(tmp_path):
    user = tmp_path / "user.toml"
    save_config(user, Config(model="user-model", api_key="user-key"))
    # The workspace config file is `.coding-agent.toml` inside the workspace.
    save_config(tmp_path / ".coding-agent.toml", Config(model="ws-model"))
    loaded = load_config(user_path=user, workspace=tmp_path)
    assert loaded.model == "ws-model"
    assert loaded.api_key == "user-key"


def test_malformed_toml_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("model = 'unterminated\n")
    with pytest.raises(ConfigurationError):
        load_config(user_path=bad)


def test_save_mode_is_0600(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, Config(model="m", api_key="k"))
    assert (os.stat(path).st_mode & 0o777) == 0o600


def test_empty_api_key_not_written(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, Config(model="m"))
    assert "api_key" not in path.read_text(encoding="utf-8")


def test_load_ignores_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("model = 'm'\nunknown_field = 'x'\n")
    loaded = load_config(user_path=path)
    assert loaded.model == "m"


def test_load_reads_max_steps_from_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("model = 'm'\nmax_steps = 7\n")
    loaded = load_config(user_path=path)
    assert loaded.max_steps == 7


def test_max_steps_defaults_to_none_unbounded():
    assert Config().max_steps is None


def test_save_writes_max_steps_only_when_set(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, Config(model="m", api_key="k", max_steps=7))
    text = path.read_text(encoding="utf-8")
    assert "max_steps = 7" in text
    assert load_config(user_path=path).max_steps == 7

    unset = tmp_path / "unset.toml"
    save_config(unset, Config(model="m"))
    assert "max_steps" not in unset.read_text(encoding="utf-8")


def test_config_dir_prefers_xdg_when_set(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config")
    assert config_dir() == Path("/tmp/xdg-config") / "coding-agent"


def test_default_user_config_path_lives_under_config_dir(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config")
    assert default_user_config_path() == (
        Path("/tmp/xdg-config") / "coding-agent" / "config.toml"
    )


# --- W6 Task 2: config-aware app resolution --------------------------------

_MODEL_ENVS = ("CODING_AGENT_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL")
_KEY_ENVS = ("CODING_AGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
_BASE_URL_ENVS = ("CODING_AGENT_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL")


def _clear_all_env(monkeypatch):
    for var in (*_MODEL_ENVS, *_KEY_ENVS, *_BASE_URL_ENVS):
        monkeypatch.delenv(var, raising=False)


def test_create_app_uses_config_when_env_absent(tmp_path, monkeypatch):
    _clear_all_env(monkeypatch)
    cfg = Config(
        model="cfg-model",
        api_key="cfg-key",
        base_url="https://x",
        context_window=5000,
    )
    app = create_app(workspace=str(tmp_path), config=cfg)
    assert app.state.model == "cfg-model"
    assert app.state.context_window == 5000


def test_cli_flag_wins_over_config(tmp_path, monkeypatch):
    _clear_all_env(monkeypatch)
    cfg = Config(model="cfg-model", api_key="cfg-key")
    app = create_app(workspace=str(tmp_path), config=cfg, model="cli-model")
    assert app.state.model == "cli-model"


def test_env_wins_over_config(tmp_path, monkeypatch):
    _clear_all_env(monkeypatch)
    monkeypatch.setenv("CODING_AGENT_MODEL", "env-model")
    cfg = Config(model="cfg-model", api_key="cfg-key")
    app = create_app(workspace=str(tmp_path), config=cfg)
    assert app.state.model == "env-model"


def test_missing_config_and_env_still_raises(tmp_path, monkeypatch):
    _clear_all_env(monkeypatch)
    with pytest.raises(MissingConfiguration):
        create_app(workspace=str(tmp_path), config=Config())


def test_resolve_api_key_uses_config_when_env_absent(monkeypatch):
    _clear_all_env(monkeypatch)
    key, _ = app_module._resolve_api_key(None, None, Config(api_key="cfg-key"))
    assert key == "cfg-key"


def test_resolve_base_url_uses_config_when_env_absent(monkeypatch):
    for var in _BASE_URL_ENVS:
        monkeypatch.delenv(var, raising=False)
    assert (
        app_module._resolve_base_url(None, Config(base_url="https://cfg.example/v1"))
        == "https://cfg.example/v1"
    )


def test_resolve_context_window_uses_config_when_arg_absent():
    assert app_module._resolve_context_window(None, Config(context_window=5000)) == 5000
    assert app_module._resolve_context_window(2000, Config(context_window=5000)) == 2000
    assert app_module._resolve_context_window(None, Config()) == DEFAULT_CONTEXT_WINDOW


def test_resolve_config_reads_file_and_overlays_cli(tmp_path):
    path = tmp_path / "config.toml"
    save_config(
        path,
        Config(model="m", api_key="k", base_url="https://x", context_window=5000),
    )
    args = app_module.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--config",
            str(path),
            "--model",
            "cli-model",
        ]
    )
    config = app_module.resolve_config(args, tmp_path)
    assert config.model == "cli-model"  # CLI wins
    assert config.api_key == "k"  # from file
    assert config.base_url == "https://x"  # from file
    assert config.context_window == 5000  # from file


def test_resolve_config_without_flag_uses_default_user_path(monkeypatch):
    captured = {}

    def fake_load_config(user_path=None, workspace=None):
        captured["user_path"] = user_path
        return Config(model="m")

    monkeypatch.setattr(app_module, "load_config", fake_load_config)
    args = app_module.build_parser().parse_args(["--workspace", "."])
    config = app_module.resolve_config(args, Path.cwd())
    assert config.model == "m"
    assert captured["user_path"] is None  # load_config resolves the default path


def test_resolve_config_passes_config_flag_as_user_path(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    captured = {}

    def fake_load_config(user_path=None, workspace=None):
        captured["user_path"] = user_path
        return Config()

    monkeypatch.setattr(app_module, "load_config", fake_load_config)
    args = app_module.build_parser().parse_args(
        ["--workspace", str(tmp_path), "--config", str(path)]
    )
    app_module.resolve_config(args, tmp_path)
    assert captured["user_path"] == path


def test_main_passes_resolved_config_to_create_app(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    save_config(
        path,
        Config(model="m", api_key="k", context_window=5000),
    )
    captured = {}

    class FakeApp:
        def run(self):
            pass

    def fake_create_app(**kwargs):
        captured.update(kwargs)
        return FakeApp()

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    assert app_module.main(["--workspace", str(tmp_path), "--config", str(path)]) == 0
    config = captured["config"]
    assert config.model == "m"
    assert config.api_key == "k"
    assert config.context_window == 5000


# --- W6 Task 3: first-run SetupScreen -------------------------------------

_FAKE_KEY = "sk-test-fake-key-12345"
"""Fake placeholder API key used in tests; real keys never appear here."""


def test_setup_screen_composes_labeled_inputs():
    screen = app_module.SetupScreen()
    statics = list(screen.compose())
    rendered = "\n".join(str(w) for w in statics)
    assert "model" in rendered.lower()
    assert "base_url" in rendered.lower()
    assert "api_key" in rendered.lower()
    assert "context_window" in rendered.lower()


def test_setup_screen_api_key_input_is_password_masked():
    screen = app_module.SetupScreen()
    assert screen._key_input.password is True
    assert screen._model_input.password is False


@pytest.mark.asyncio
async def test_setup_save_writes_config_with_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "default_user_config_path", lambda: tmp_path / "config.toml"
    )
    screen = app_module.SetupScreen()
    async with screen.run_test() as pilot:
        screen._model_input.value = "fake-model"
        screen._key_input.value = _FAKE_KEY
        await pilot.pause()
        screen._save()
        await pilot.pause()
    content = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "model = 'fake-model'" in content
    assert f"api_key = '{_FAKE_KEY}'" in content  # stored per user intent
    assert (tmp_path / "config.toml").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_setup_save_validates_model_and_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "default_user_config_path", lambda: tmp_path / "config.toml"
    )
    screen = app_module.SetupScreen()
    async with screen.run_test() as pilot:
        screen._model_input.value = "m"  # key still empty -> no save
        await pilot.pause()
        screen._save()
        await pilot.pause()
    assert not (tmp_path / "config.toml").exists()


@pytest.mark.asyncio
async def test_setup_screen_never_echoes_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "default_user_config_path", lambda: tmp_path / "config.toml"
    )
    screen = app_module.SetupScreen()
    async with screen.run_test() as pilot:
        screen._model_input.value = "fake-model"
        screen._key_input.value = _FAKE_KEY
        await pilot.pause()
        # No Static label/guidance text may contain the key.
        static_text = "\n".join(str(w.render()) for w in screen.query(Static))
        assert _FAKE_KEY not in static_text
        screen._save()
        await pilot.pause()
    # The key is stored in the 0600 config file (that is the point of the file).
    assert _FAKE_KEY in (tmp_path / "config.toml").read_text(encoding="utf-8")


def test_run_onboarding_runs_setup_screen_on_interactive_tty(monkeypatch):
    ran = []

    class FakeSetup:
        def __init__(self, message):
            self.message = message

        def run(self):
            ran.append(self.message)

    monkeypatch.setattr(app_module, "SetupScreen", FakeSetup)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert app_module._run_onboarding("wizard-msg") == 0
    assert ran == ["wizard-msg"]


def test_run_onboarding_non_tty_uses_static_screen(monkeypatch):
    ran = []

    class FakeScreen:
        def __init__(self, message):
            self.message = message

        def run(self):
            ran.append(self.message)

    monkeypatch.setattr(app_module, "ConfigurationScreen", FakeScreen)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert app_module._run_onboarding("guidance-msg") == 0
    assert ran == ["guidance-msg"]


def test_run_onboarding_interactive_false_never_runs_setup_screen(monkeypatch):
    ran = []

    class FakeSetup:
        def __init__(self, message=None):
            self.message = message

        def run(self):
            ran.append("setup")

    class FakeScreen:
        def __init__(self, message):
            self.message = message

        def run(self):
            ran.append("static")

    monkeypatch.setattr(app_module, "SetupScreen", FakeSetup)
    monkeypatch.setattr(app_module, "ConfigurationScreen", FakeScreen)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)  # TTY but forced off
    assert app_module._run_onboarding("msg", interactive=False) == 0
    assert ran == ["static"]


def test_decision_memory_always_path_defaults_to_config_dir(monkeypatch):
    from coding_agent.policy.memory import DecisionMemory

    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config")
    assert DecisionMemory().always_path == (
        Path("/tmp/xdg-config") / "coding-agent" / "approvals.json"
    )


def test_decision_memory_always_path_explicit_override(tmp_path):
    from coding_agent.policy.memory import DecisionMemory

    path = tmp_path / "approvals.json"
    assert DecisionMemory(always_path=path).always_path == path
