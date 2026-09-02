import asyncio
from pathlib import Path

from coding_agent.app import build_system_prompt, create_app
from coding_agent.runtime.models import LLMEvent
from coding_agent.skills.discovery import discover_skills


class _RecordingProvider:
    def __init__(self):
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append((messages, tools, model))
        for event in [LLMEvent(type="response_end", finish_reason="stop")]:
            if signal.is_set():
                return
            yield event


def _workspace_root(tmp_path: Path) -> Path:
    return tmp_path / ".coding-agent" / "skills"


def _write_skill(root: Path, text: str) -> None:
    path = root / "demo" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_system_prompt_with_fixture_skills_includes_catalog(tmp_path):
    _write_skill(
        _workspace_root(tmp_path),
        "---\ndescription: Do the demo thing\n---\n\nBody goes here.\n",
    )
    skills = discover_skills(tmp_path, user_root=tmp_path / "user")

    content = build_system_prompt(tmp_path, "default", skills=skills).content

    assert content is not None
    assert "## Available skills" in content
    assert "- demo: Do the demo thing" in content


def test_system_prompt_empty_skill_set_omits_catalog_and_keeps_content(tmp_path):
    content = build_system_prompt(tmp_path, "full", skills=()).content

    assert content is not None
    assert "## Available skills" not in content
    assert "Permission boundaries" in content
    assert str(tmp_path) in content
    assert "You are coding-agent" in content


def test_system_prompt_excludes_skill_bodies(tmp_path):
    _write_skill(
        _workspace_root(tmp_path),
        "---\ndescription: Do the demo thing\n---\n\nBODY-ONLY-SECRET\n",
    )
    skills = discover_skills(tmp_path, user_root=tmp_path / "user")

    content = build_system_prompt(tmp_path, "default", skills=skills).content

    assert content is not None
    assert "- demo: Do the demo thing" in content
    assert "BODY-ONLY-SECRET" not in content


async def test_create_app_wires_the_discovered_catalog(tmp_path):
    _write_skill(
        _workspace_root(tmp_path),
        "---\ndescription: Do the demo thing\n---\n\nBody.\n",
    )
    provider = _RecordingProvider()
    application = create_app(
        workspace=tmp_path,
        model="fake-model",
        session_dir=tmp_path / "sessions",
        context_window=2_000,
        provider=provider,
        permission_mode="full",
    )

    completed = asyncio.Event()

    async def observe(event):
        if event.type == "run_finished":
            completed.set()

    application.runtime.subscribe(observe)
    await application.runtime.submit("hi")
    await asyncio.wait_for(completed.wait(), timeout=5)

    assert provider.requests
    system_message = provider.requests[0][0][0]
    assert system_message.role == "system"
    assert system_message.content is not None
    assert "- demo: Do the demo thing" in system_message.content
