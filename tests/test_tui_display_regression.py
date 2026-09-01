"""Regression tests for TUI transcript rendering bugs.

Covers:
- DuplicateIds caused by overlapping transcript refreshes.
- Tool output flooding the transcript (truncation with a more-lines marker).
- Idempotent user-message rows when an event is re-applied.
"""

from __future__ import annotations

import asyncio

import pytest

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import TranscriptItem, TuiState, initial_state
from coding_agent.tui.widgets import _row_text


class _MinimalRuntime:
    def __init__(self) -> None:
        self.workspace = "/tmp/project"
        self.model = "fake-model"
        self.session_id = "sess-1"
        self.permission_mode = "default"
        self.status = type(
            "S",
            (),
            {
                "usage": None,
                "status": "idle",
                "context_window": 1000,
                "context_estimated": False,
                "context_used": 0,
            },
        )()
        self.store = type(
            "ST",
            (),
            {
                "header": type(
                    "H",
                    (),
                    {
                        "workspace": "/tmp/project",
                        "model": "fake-model",
                        "context_window": 1000,
                    },
                )()
            },
        )()

    def subscribe(self, sink):
        return lambda: None

    async def submit(self, prompt: str) -> str:
        return "run-1"

    async def abort(self, run_id: str) -> None:
        pass

    async def resolve_approval(self, request_id: str, decision: str) -> None:
        pass

    async def set_permission(self, mode: str) -> None:
        pass

    async def new_session(self) -> str:
        return "new"

    async def list_sessions(self):
        return []

    async def resume(self, session_id: str) -> None:
        pass

    async def compact(self) -> None:
        pass


def _make_state(rows: list[TranscriptItem]) -> TuiState:
    return initial_state("/tmp/project", "fake-model", context_window=1000).model_copy(
        update={"transcript": rows}
    )


@pytest.mark.asyncio
async def test_concurrent_refresh_does_not_raise_duplicate_ids() -> None:
    """Overlapping transcript refreshes must not raise DuplicateIds."""
    state = _make_state(
        [
            TranscriptItem(kind="user", item_id="user-T1", text="hello"),
            TranscriptItem(kind="user", item_id="user-T2", text="world"),
            TranscriptItem(kind="assistant", item_id="msg-1", text="hi"),
        ]
    )
    runtime = _MinimalRuntime()
    async with CodingAgentApp(runtime=runtime, initial_state=state).run_test() as pilot:
        app = pilot.app
        # Two refreshes racing on the same transcript used to interleave
        # remove_children()/mount_all() and raise DuplicateIds.
        await asyncio.gather(
            app._refresh_widgets(),
            app._refresh_widgets(),
        )
        await pilot.pause()
        assert app.query_one("#transcript").children  # transcript still mounted


def test_tool_row_text_truncates_long_output() -> None:
    """Tool output is previewed compactly and capped when expanded."""
    text = "\n".join(f"line {i}" for i in range(30))
    item = TranscriptItem(
        kind="tool",
        item_id="call-1",
        tool_call_id="call-1",
        tool_name="run_command",
        text=text,
        tool_status="success",
    )
    compact = _row_text(item)
    assert "line 0" in compact
    assert "line 29" not in compact
    assert "(22 more lines)" not in compact

    expanded = _row_text(item.model_copy(update={"expanded": True}))
    assert "line 29" not in expanded
    assert "(22 more lines)" in expanded


def test_tool_row_text_truncates_long_lines() -> None:
    """A single over-long output line must be truncated with an ellipsis."""
    text = "x" * 500
    rendered = _row_text(
        TranscriptItem(
            kind="tool",
            item_id="call-2",
            tool_call_id="call-2",
            tool_name="run_command",
            text=text,
            tool_status="success",
        )
    )
    assert "x" * 300 not in rendered
    assert "…" in rendered


def test_short_tool_output_is_unchanged() -> None:
    """Small tool results keep their full text as a preview line."""
    rendered = _row_text(
        TranscriptItem(
            kind="tool",
            item_id="call-3",
            tool_call_id="call-3",
            tool_name="run_command",
            text="ok\n",
            tool_status="success",
        )
    )
    assert "✓ Bash" in rendered
    assert "  ⎿  ok" in rendered


def test_reapply_user_message_does_not_duplicate_row() -> None:
    """Re-applying the same user_message event must update, not duplicate."""
    state = initial_state("/tmp/project", "fake-model", context_window=1000)
    event = RuntimeEvent(
        type="user_message",
        run_id="run-1",
        turn_id="turn-1",
        payload={"message_id": "user-turn-1", "text": "hello"},
    )
    first = reduce(state, event)
    second = reduce(first, event)
    user_rows = [row for row in second.transcript if row.kind == "user"]
    assert len(user_rows) == 1
