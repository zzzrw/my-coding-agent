# coding-agent

A small, local-first coding assistant with an OpenAI-compatible model provider, explicit tool permissions, durable JSONL sessions, and a Textual terminal UI.

## Developer setup

The project targets Python 3.11 or newer and uses `uv` for reproducible development environments.

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

The package also exposes the `coding-agent` console script after installation:

```bash
uv run coding-agent --help
```

## Running locally

Configure a model and credential through environment variables, then launch from a workspace:

```bash
export CODING_AGENT_MODEL="your-model"
export CODING_AGENT_API_KEY="your-api-key"
# Optional OpenAI-compatible endpoint:
# export CODING_AGENT_BASE_URL="https://example.invalid/v1"
uv run coding-agent --workspace .
```

`OPENAI_*` and `DEEPSEEK_*` compatibility environment variables are also accepted. The CLI supports `--workspace`, `--model`, `--base-url`, `--session-dir`, and `--context-window`. Help and configuration errors do not expose credential values.

An optional workspace `.coding-agent.toml` (or user `config.toml`) may set `max_steps = N` to cap steps per turn (absent = unbounded) and `language = 'zh'` to pick the reply language (default Chinese).

## Skills

Skills are reusable instruction packages the agent can load on demand. Each skill is a directory holding a `SKILL.md` (YAML frontmatter plus a Markdown body) under one of two discovery roots, scanned workspace-first:

- `<workspace>/.coding-agent/skills/<name>/SKILL.md`
- `~/.config/coding-agent/skills/<name>/SKILL.md`

Frontmatter fields: `description` (required; a skill without one is skipped), optional `name` (defaults to the directory name), and optional `when_to_use`. Discovered skills appear as a one-line catalog in the run system prompt and can be loaded with the `load_skill` tool, which returns the `SKILL.md` body (bounded at 16,000 characters) plus the skill directory and a sorted listing of any files bundled beside it (e.g. under `scripts/`). Helper files are read or run by the agent through the normal `read_file`/`run_command` tools under the usual permission checks — `load_skill` never executes them itself. `/skills` lists the installed catalog in the TUI. No skill body is ever injected automatically.

## Architecture

`coding_agent.app.create_app()` resolves configuration and composes the application boundary. It creates an OpenAI-compatible provider, the six built-in filesystem/search/shell tools, a `ToolRegistry`, `DefaultApprovalPolicy`, and a `ToolExecutor`. The runtime supplies its approval broker through the runner factory, so all tool calls still pass through the existing permission boundary.

`AgentRuntime` owns turns, persistence, cancellation, approval, and event publication. `CodingAgentApp` receives that runtime, reduces runtime events into immutable TUI state, and renders the transcript, composer, and status line. The TUI does not call providers, tools, or stores directly.

Sessions are JSONL files under `~/.coding-agent/sessions` by default. Use `--session-dir` to select a project-local or disposable directory. Tests inject a fake provider and use temporary workspaces, so they do not make network requests.
