from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from coding_agent.skills.models import Skill

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.commands import (
    SUPPORTED_COMMANDS,
    command_suggestions,
    parse_command,
)
from coding_agent.tui.state import TuiState, initial_state
from coding_agent.tui.widgets import (
    HelpScreen,
    SkillsScreen,
    SubmitTextArea,
    help_overlay_text,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.subscribers: list[Callable[[RuntimeEvent], Awaitable[None]]] = []
        self.submitted: list[str] = []

    def subscribe(
        self, sink: Callable[[RuntimeEvent], Awaitable[None]]
    ) -> Callable[[], None]:
        self.subscribers.append(sink)

        def unsubscribe() -> None:
            if sink in self.subscribers:
                self.subscribers.remove(sink)

        return unsubscribe

    async def submit(self, prompt: str) -> str:
        self.submitted.append(prompt)
        return "run-1"


def _skill(name: str, description: str, when_to_use: str | None = None) -> Skill:
    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        path=Path(f"/x/{name}/SKILL.md"),
        root="workspace",
    )


def _write_skill(root: Path, text: str) -> None:
    path = root / "demo" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_state(workspace: Path) -> TuiState:
    return initial_state(str(workspace), "fake-model", context_window=1000)


def _write_demo_skill(tmp_path: Path) -> Path:
    ws_root = tmp_path / ".coding-agent" / "skills"
    _write_skill(
        ws_root,
        "---\n"
        "description: Demo skill for testing\n"
        "when_to_use: when a demo is required\n"
        "---\n\nDo the demo.\n",
    )
    return ws_root


def test_skills_command_registered():
    assert "skills" in SUPPORTED_COMMANDS
    assert command_suggestions("ski")[0].description == "List available skills"


def test_help_overlay_empty_default_shows_no_skills():
    body = help_overlay_text().plain
    assert "Skills" in body
    assert "no skills installed" in body


def test_help_overlay_with_catalog_lists_skills():
    body = help_overlay_text(skills=[_skill("demo", "Do it")]).plain
    assert "Skills" in body
    assert "- demo: Do it" in body


@pytest.mark.asyncio
async def test_skills_command_opens_overlay_listing_catalog(tmp_path):
    _write_demo_skill(tmp_path)
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=_make_state(tmp_path),
        skills_user_root=tmp_path / "empty-user",
        branch_detector=lambda workspace: None,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/skills"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, SkillsScreen)
        assert "demo" in app.screen.body.plain
        assert "Demo skill for testing" in app.screen.body.plain
        assert "when a demo is required" in app.screen.body.plain


@pytest.mark.asyncio
async def test_skills_overlay_empty_state(tmp_path):
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=_make_state(tmp_path),
        skills_user_root=tmp_path / "empty-user",
        branch_detector=lambda workspace: None,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/skills"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, SkillsScreen)
        assert "no skills installed" in app.screen.body.plain


@pytest.mark.asyncio
async def test_skills_command_with_args_is_a_usage_notice(tmp_path):
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=_make_state(tmp_path),
        skills_user_root=tmp_path / "empty-user",
        branch_detector=lambda workspace: None,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/skills extra"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, SkillsScreen)
        assert "usage: /skills" in app.state.transcript[-1].text


@pytest.mark.asyncio
async def test_help_typed_in_composer_lists_installed_skills(tmp_path):
    _write_demo_skill(tmp_path)
    runtime = FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=_make_state(tmp_path),
        skills_user_root=tmp_path / "empty-user",
        branch_detector=lambda workspace: None,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer-input", SubmitTextArea)
        composer.text = "/help"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, HelpScreen)
        assert "demo" in app.screen.body.plain
        assert "no skills installed" not in app.screen.body.plain


def test_arbitrary_skill_slash_is_not_a_command():
    command = parse_command("/demo")
    assert command.name == "demo"
    assert command.args == []
    assert "demo" not in SUPPORTED_COMMANDS
