"""Tests for auto-creating an explicitly supplied, not-yet-existing workspace.

An explicit ``workspace`` argument that names a concrete directory is created
(including parents) instead of rejected. The ``None`` default keeps the
current-directory fallback and never auto-creates anything.
"""

from pathlib import Path

import pytest
from fakes import FakeProvider

from coding_agent import app as app_module
from coding_agent.app import ConfigurationError, _resolve_workspace, create_app


def test_missing_workspace_dir_is_created_and_app_builds(tmp_path):
    """A fresh missing path is created and create_app builds a working runtime."""
    target = tmp_path / "brand-new" / "workspace"
    application = create_app(
        workspace=str(target),
        model="fake",
        provider=FakeProvider([]),
        session_dir=tmp_path / "sessions",
    )
    assert target.is_dir()
    assert application.runtime.store.header.workspace == str(target.resolve())


def test_nested_missing_workspace_creates_all_parents(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    resolved = _resolve_workspace(target)
    assert resolved == target.resolve()
    assert target.is_dir()


def test_missing_workspace_with_dotdot_resolves_before_creating(tmp_path):
    """``..`` in a missing path collapses onto the real target before creation."""
    raw = tmp_path / "proj" / "sub" / ".." / "deep" / "nested"
    resolved = _resolve_workspace(raw)
    assert resolved == (tmp_path / "proj" / "deep" / "nested").resolve()
    assert (tmp_path / "proj" / "deep" / "nested").is_dir()


def test_missing_workspace_under_home_expands_and_creates(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = Path("~/new-ws/child")
    resolved = _resolve_workspace(target)
    assert resolved == (tmp_path / "new-ws" / "child").resolve()
    assert (tmp_path / "new-ws" / "child").is_dir()


def test_existing_file_workspace_still_raises_and_is_not_clobbered(tmp_path):
    target = tmp_path / "afile"
    target.write_text("contents")
    with pytest.raises(ConfigurationError, match="not a directory"):
        _resolve_workspace(target)
    assert target.is_file()
    assert target.read_text() == "contents"


def test_existing_dir_workspace_resolves_unchanged(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    assert _resolve_workspace(existing) == existing.resolve()


def test_none_workspace_keeps_cwd_and_never_auto_creates(monkeypatch):
    """``workspace=None`` falls back to the cwd and must not call makedirs."""
    calls = []

    def _fail(*args, **kwargs):
        calls.append(True)
        raise AssertionError("must not auto-create when workspace is None")

    monkeypatch.setattr(app_module.os, "makedirs", _fail)
    resolved = app_module._resolve_workspace(None)
    assert resolved == Path.cwd().resolve()
    assert calls == []
