"""Config language preference and its system-prompt injection."""

from pathlib import Path

from fakes import FakeProvider

from coding_agent.app import build_system_prompt, create_app
from coding_agent.config.config import Config, _toml_lines, load_config, save_config


def test_config_language_default_is_zh():
    assert Config().language == "zh"


def test_config_language_serialized_only_when_not_zh(tmp_path):
    assert not any(l.startswith("language") for l in _toml_lines(Config()))
    assert "language = 'en'" in _toml_lines(Config(language="en"))


def test_config_language_roundtrips(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, Config(language="ja"))
    assert load_config(user_path=path).language == "ja"


def test_system_prompt_defaults_to_chinese(tmp_path):
    message = build_system_prompt(Path(tmp_path), "workspace")
    assert message.content.startswith("Respond in Chinese.")


def test_system_prompt_language_override(tmp_path):
    message = build_system_prompt(Path(tmp_path), "workspace", language="en")
    assert message.content.startswith("Respond in English.")


def test_create_app_threads_language_into_runner(tmp_path):
    application = create_app(
        workspace=tmp_path,
        config=Config(model="fake", language="en"),
        provider=FakeProvider([]),
    )
    prompt = application.runtime._runner.system_prompt.content
    assert prompt.startswith("Respond in English.")


def test_create_app_language_param_beats_config(tmp_path):
    application = create_app(
        workspace=tmp_path,
        language="en",
        config=Config(model="fake", language="ja"),
        provider=FakeProvider([]),
    )
    prompt = application.runtime._runner.system_prompt.content
    assert prompt.startswith("Respond in English.")
