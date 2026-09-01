"""TOML configuration for coding-agent.

The ``Config`` model holds the provider settings (model, API key, base URL,
context window, permission mode). ``load_config`` merges the user config file
(``$XDG_CONFIG_HOME/coding-agent/config.toml``) with an optional workspace
``.coding-agent.toml``; workspace values win. ``save_config`` persists with
mode ``0600`` so an API key is never world-readable.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

_CONFIG_FILENAME = ".coding-agent.toml"


class ConfigurationError(ValueError):
    """Raised when configuration is absent, malformed, or invalid."""


class Config(BaseModel):
    """Provider configuration loaded from a TOML config file."""

    model: str = ""
    api_key: str = ""
    base_url: str = ""
    context_window: int = 0
    permission_mode: str = "default"

    model_config = ConfigDict(extra="forbid")


def config_dir() -> Path:
    """Return the directory holding user-level configuration files."""
    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "coding-agent"
    return Path.home() / ".config" / "coding-agent"


def default_user_config_path() -> Path:
    """Return the default user config file path."""
    return config_dir() / "config.toml"


def load_config(user_path: Path | None = None, workspace: Path | None = None) -> Config:
    """Load and merge config from the user file then the workspace file.

    ``user_path`` defaults to ``default_user_config_path()``; when
    ``workspace`` is given, its ``.coding-agent.toml`` is merged second so
    workspace values override user values. Missing files are skipped; malformed
    TOML raises :class:`ConfigurationError` with the offending path.
    """
    paths: list[Path] = [
        user_path if user_path is not None else default_user_config_path()
    ]
    if workspace is not None:
        paths.append(Path(workspace) / _CONFIG_FILENAME)

    combined: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigurationError(f"invalid config: {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"invalid config: {path}: top-level TOML is not a table"
            )
        for key, value in raw.items():
            if key in Config.model_fields:
                combined[key] = value

    try:
        return Config(**combined)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid config: {exc}") from exc


def _toml_lines(config: Config) -> list[str]:
    """Render ``config`` as TOML lines, skipping empty/default fields."""
    lines: list[str] = []
    if config.model:
        lines.append(f"model = '{config.model}'")
    if config.api_key:
        lines.append(f"api_key = '{config.api_key}'")
    if config.base_url:
        lines.append(f"base_url = '{config.base_url}'")
    if config.context_window:
        lines.append(f"context_window = {config.context_window}")
    if config.permission_mode and config.permission_mode != "default":
        lines.append(f"permission_mode = '{config.permission_mode}'")
    return lines


def save_config(path: Path, config: Config) -> None:
    """Write ``config`` to ``path`` as TOML with mode ``0600``.

    The file is created atomically with ``0600`` so an API key is never
    world-readable; an empty ``api_key`` is never written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(_toml_lines(config))
    if body:
        body += "\n"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(body)
    os.chmod(path, 0o600)
