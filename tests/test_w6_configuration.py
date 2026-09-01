"""W6 Task 1: TOML configuration module tests.

All API keys used here are fake placeholders; real keys never appear in tests.
"""

import os
from pathlib import Path

import pytest

from coding_agent.app import ConfigurationError
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


def test_config_dir_prefers_xdg_when_set(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config")
    assert config_dir() == Path("/tmp/xdg-config") / "coding-agent"


def test_default_user_config_path_lives_under_config_dir(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config")
    assert default_user_config_path() == (
        Path("/tmp/xdg-config") / "coding-agent" / "config.toml"
    )
