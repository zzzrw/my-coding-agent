# W6 — Configuration & First-Run Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** A TOML config file supplies model/key/base_url/context-window; on first
run with no configuration anywhere, an interactive wizard walks the user through
setup.

**Architecture:** A new `config/config.py` module defines a pydantic `Config`,
`load_config` (merging user + workspace TOML, CLI/env winning), and `save_config`
(mode `0600`). `app.py`'s resolution helpers consult config as a fallback after
CLI/env. A Textual `SetupScreen` replaces the static onboarding screen when
interactive; non-TTY still prints guidance.

**Tech Stack:** Python 3.11+, tomllib (stdlib, 3.11), Textual 8.2.8, pydantic.

**Spec:** `docs/superpowers/specs/2026-09-01-coding-agent-feature-roadmap-design.md` §6.

## Global Constraints

- API keys are never echoed or logged; `save_config` writes `0600`.
- `tomllib` is stdlib on 3.11 — no new dependency.
- The config dir helper is shared with W2's `approvals.json` (`always` decisions).
- Existing 395+ tests stay green.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Map

- `src/coding_agent/config/config.py` (new): `Config`, `load_config`, `save_config`, `config_dir`.
- `src/coding_agent/app.py`: config-aware resolution + `--config` flag + `SetupScreen`.
- `src/coding_agent/policy/memory.py` (W2): reuse `config_dir` helper for `approvals.json`.
- `tests/test_w6_configuration.py` (new).

---

## Task 1: Config module

**Files:** Create `src/coding_agent/config/config.py`, `src/coding_agent/config/__init__.py`; test `tests/test_w6_configuration.py`.

**Interfaces:**
- Produces:
  - `Config(BaseModel)` with `model: str = ""`, `api_key: str = ""`,
    `base_url: str = ""`, `context_window: int = 0`,
    `permission_mode: str = "default"`.
  - `config_dir() -> Path` (`$XDG_CONFIG_HOME/coding-agent` or `~/.config/coding-agent`).
  - `load_config(user_path: Path | None = None, workspace: Path | None = None) -> Config` — merges user then workspace `.coding-agent.toml`; raises `ConfigurationError` on malformed TOML with a clear message.
  - `save_config(path: Path, config: Config) -> None` — writes TOML with mode `0600`.
  - `default_user_config_path() -> Path`.

- [ ] **Step 1: Failing tests**

```python
import tomllib  # noqa: F401  (stdlib)
from coding_agent.config.config import (
    Config, config_dir, default_user_config_path, load_config, save_config,
)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, Config(model="m", api_key="k", base_url="https://x",
                             context_window=1000))
    loaded = load_config(user_path=path)
    assert loaded.model == "m" and loaded.api_key == "k"
    assert loaded.base_url == "https://x" and loaded.context_window == 1000


def test_load_merges_user_then_workspace(tmp_path):
    user = tmp_path / "user.toml"
    ws = tmp_path / "ws.toml"
    save_config(user, Config(model="user-model", api_key="user-key"))
    save_config(ws, Config(model="ws-model"))
    loaded = load_config(user_path=user, workspace=tmp_path)  # ws.toml in tmp_path
    assert loaded.model == "ws-model"
    assert loaded.api_key == "user-key"


def test_malformed_toml_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("model = 'unterminated\n")
    import pytest
    from coding_agent.app import ConfigurationError
    with pytest.raises(ConfigurationError):
        load_config(user_path=bad)


def test_save_mode_is_0600(tmp_path):
    import os
    path = tmp_path / "config.toml"
    save_config(path, Config(model="m", api_key="k"))
    assert (os.stat(path).st_mode & 0o777) == 0o600
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `config.py`: pydantic `Config` with `extra="forbid"`. `load_config` reads user then workspace `coding-agent.toml` (workspace file named `.coding-agent.toml`), parsing with `tomllib.loads`, mapping unknown/extra keys leniently (only known fields), catching `tomllib.TOMLDecodeError` → raise `ConfigurationError(f"invalid config: {path}: {exc}")`. `save_config` builds TOML text manually (single quoted values; never includes an empty key field), writes via `os.open(path, O_CREAT|O_TRUNC|O_WRONLY, 0o600)`, `chmod` to `0600`.
  - `__init__.py` re-exports.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Add a TOML config module with 0600 persistence`

---

## Task 2: Config-aware app resolution + `--config` flag

**Files:** Modify `src/coding_agent/app.py`; test `tests/test_w6_configuration.py`.

**Interfaces:**
- Consumes: `load_config`, `Config`.
- Produces:
  - `create_app(..., config: Config | None = None)`.
  - Resolution order per field: explicit arg > env var > `config` value > default.
  - `main()` loads config (unless `--config` given) and passes it in; missing config + missing env still raises `MissingConfiguration`.

- [ ] **Step 1: Failing tests**

```python
def test_create_app_uses_config_when_env_absent(tmp_path, monkeypatch):
    for var in ("CODING_AGENT_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL",
                "CODING_AGENT_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = Config(model="cfg-model", api_key="cfg-key", base_url="https://x",
                 context_window=5000)
    app = create_app(workspace=str(tmp_path), config=cfg)
    assert app.state.model == "cfg-model"
    assert app.state.context_window == 5000


def test_cli_flag_wins_over_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CODING_AGENT_MODEL", "env-model")
    cfg = Config(model="cfg-model")
    app = create_app(workspace=str(tmp_path), config=cfg, model="cli-model")
    assert app.state.model == "cli-model"


def test_main_parses_config_flag(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, Config(model="m", api_key="k", context_window=5000))
    exit_code = main(["--workspace", str(tmp_path), "--config", str(path)])
    # exits 0; the app constructs and runs (will block) -> instead assert via
    # create_app path; here use a helper that returns the resolved config.
```

  For the last test, refactor `main` so config resolution is a separable function
  `resolve_config(args, workspace) -> Config` that tests call directly without
  launching the app.

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `_resolve_model(model, config)` / `_resolve_api_key(api_key, credential_env, config)` /
    base-url resolution / `_resolve_context_window(context_window, config)`: append a `config` fallback before the default.
  - `create_app(..., config=None)`: if `config is None`, `config = load_config(...)` lazily only when needed (avoid FS reads when env already set — resolve only missing fields from config).
  - `main()`: `--config` flag; build `Config` from parsed args overrides then config file; call `create_app(..., config=cfg)`.
  - `_run_onboarding`: still used when interactive setup is skipped.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Resolve app configuration from TOML with CLI and env priority`

---

## Task 3: First-run `SetupScreen`

**Files:** Modify `src/coding_agent/app.py`, `src/coding_agent/tui/widgets.py`; test `tests/test_w6_configuration.py`.

**Interfaces:**
- Produces:
  - `SetupScreen(App)` — interactive Textual form: model, base_url (optional), api_key (password-masked), context_window; `Save` writes `default_user_config_path()` and continues.
  - `_run_onboarding(message, interactive=True)`: interactive TTY runs `SetupScreen`; non-TTY prints guidance.
  - `config_dir()` reused by W2's `DecisionMemory.always_path`.

- [ ] **Step 1: Failing tests**

```python
def test_setup_screen_composes_inputs():
    from coding_agent.app import SetupScreen
    screen = SetupScreen()
    statics = list(screen.compose())
    rendered = "\n".join(str(w) for w in statics)
    assert "model" in rendered.lower()


def test_setup_save_writes_config(tmp_path, monkeypatch):
    from coding_agent.app import SetupScreen
    from coding_agent.config.config import default_user_config_path
    monkeypatch.setattr("coding_agent.app.default_user_config_path",
                        lambda: tmp_path / "config.toml")
    screen = SetupScreen()
    screen._model_input.value = "m"
    screen._key_input.value = "k"
    screen._save()
    assert (tmp_path / "config.toml").exists()
    assert "sk-" not in (tmp_path / "config.toml").read_text()  # key never echoed
```

  (Note: the key IS stored in the file — that's the point of the config file.
  The assertion above is about the *onboarding guidance/screen text*, not the
  file. Adjust: assert the file contains the key per user intent, but the
  screen output and logs never do.)

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**
  - `SetupScreen`: a Textual `Screen` with labeled `Input` widgets (`model`,
    `base_url`, `api_key` with `password=True`, `context_window`) and a
    `Save` button; on save, validate model+key non-empty, write config via
    `save_config`, then `self.exit(True)`.
  - `_run_onboarding`: keep the static `ConfigurationScreen` for non-TTY
    (`sys.stdin.isatty()`), use `SetupScreen` when interactive.
  - `policy/memory.py`: import `config_dir` from `config.config` as the default
    `always_path` location.

- [ ] **Step 4: Run focused + full suite**

- [ ] **Step 5: Commit** — `Add an interactive first-run setup wizard writing the config file`

---

## Self-Review

- Spec §6 covered: config module (T1), resolution (T2), wizard (T3). ✅
- W2 `DecisionMemory.always_path` now defaults to `config_dir()/approvals.json`. ✅
- No secrets in tests/docs; the exposed DeepSeek keys never appear. ✅
