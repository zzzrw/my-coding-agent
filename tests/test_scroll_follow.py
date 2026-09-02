"""Transcript keeps following the newest row when a local/notice row appends."""

import pytest
from test_tui import FakeRuntime

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.state import TranscriptItem, initial_state
from coding_agent.tui.widgets import TranscriptView


def _tall_state():
    state = initial_state("/tmp/p", "fake-model", context_window=1000)
    transcript = [
        TranscriptItem(
            kind="user", item_id=f"u{i:02d}", text=f"line {i:02d} " + "word " * 14
        )
        for i in range(80)
    ]
    return state.model_copy(update={"transcript": transcript})


@pytest.mark.asyncio
async def test_notice_appended_at_bottom_keeps_view_following():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=_tall_state())

    async with app.run_test() as pilot:
        view = pilot.app.query_one("#transcript", TranscriptView)
        view.scroll_end(animate=False)
        await pilot.pause()
        assert view.is_vertical_scroll_end

        await runtime.emit(
            RuntimeEvent(type="notice", payload={"message": "bottom notice row"})
        )
        await pilot.pause()

        # The newest local/notice row must be visible: view stays at the end.
        assert view.is_vertical_scroll_end
        assert "bottom notice row" in view.renderable_text


@pytest.mark.asyncio
async def test_local_command_appended_at_bottom_stays_visible():
    runtime = FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=_tall_state())

    async with app.run_test() as pilot:
        view = pilot.app.query_one("#transcript", TranscriptView)
        view.scroll_end(animate=False)
        await pilot.pause()

        composer = pilot.app.query_one("#composer-input")
        composer.text = "/skills"
        await pilot.press("enter")
        await pilot.pause()

        assert view.is_vertical_scroll_end
