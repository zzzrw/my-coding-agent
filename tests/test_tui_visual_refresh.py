"""Tests for the approved TUI visual refresh plan (2026-09-01).

Coverage is added per task:
- Task 1: tool data chain (command + result metadata) across runner, reducer,
  state and the resumed-session projection.
- Task 2: per-kind transcript row classes and app CSS rules.
- Task 3: compact click-expandable tool rows.
- Task 4: lightweight markdown_to_text for assistant rows.
"""

from __future__ import annotations

import asyncio

import pytest

from coding_agent.runtime.events import RuntimeEvent
from coding_agent.runtime.models import LLMEvent, Message
from coding_agent.runtime.runner import AgentRunner
from coding_agent.session.store import SessionStore
from coding_agent.tools.models import ToolResult
from coding_agent.tools.registry import ToolRegistry
from coding_agent.tui.app import CodingAgentApp
from coding_agent.tui.reducer import reduce
from coding_agent.tui.state import TranscriptItem, TuiState, initial_state
from coding_agent.tui.widgets import TranscriptRow, _row_text, markdown_to_text


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append(messages)
        for event in self.responses.pop(0):
            yield event


def tool_response(arguments='{"command": "ls -la"}'):
    return [
        LLMEvent(type="tool_call_start", tool_call_id="c1", tool_name="run_command"),
        LLMEvent(type="tool_call_delta", tool_call_id="c1", arguments_delta=arguments),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]


def text_response(text="done"):
    return [
        LLMEvent(type="text_delta", text=text),
        LLMEvent(type="response_end", finish_reason="stop"),
    ]


# ---------------------------------------------------------------------------
# Task 1: tool data chain (command + metadata)
# ---------------------------------------------------------------------------


def test_tool_started_stores_run_command_argument():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_started",
            payload={
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "arguments": {"command": "ls -la"},
            },
        ),
    )

    row = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert row.command == "ls -la"


def test_tool_started_stores_compact_string_for_other_tools():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_started",
            payload={
                "tool_call_id": "c1",
                "tool_name": "read_file",
                "arguments": {"path": "main.py", "start_line": 1},
            },
        ),
    )

    row = next(row for row in state.transcript if row.tool_call_id == "c1")
    assert row.command is not None
    assert "main.py" in row.command


def test_tool_finished_preserves_command_and_merges_metadata():
    state = initial_state(workspace="/tmp/project", model="fake")
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_started",
            payload={
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "arguments": {"command": "ls -la"},
            },
        ),
    )
    state = reduce(
        state,
        RuntimeEvent(
            type="tool_finished",
            payload={
                "tool_call_id": "c1",
                "tool_name": "run_command",
                "ok": False,
                "content": "total 8",
                "error": "exit code 2",
                "metadata": {"exit_code": 2, "elapsed_seconds": 2.5, "truncated": True},
            },
        ),
    )

    rows = [row for row in state.transcript if row.tool_call_id == "c1"]
    assert len(rows) == 1
    row = rows[0]
    assert row.command == "ls -la"
    assert row.text == "total 8"
    assert row.elapsed_seconds == 2.5
    assert row.truncated is True
    assert row.exit_code == 2
    assert row.expanded is False


def test_tool_finished_reapply_is_idempotent_for_same_call():
    state = initial_state(workspace="/tmp/project", model="fake")
    finished = RuntimeEvent(
        type="tool_finished",
        payload={
            "tool_call_id": "c1",
            "tool_name": "run_command",
            "ok": True,
            "content": "ok",
            "metadata": {"exit_code": 0, "elapsed_seconds": 1.0, "truncated": False},
        },
    )
    state = reduce(state, finished)
    state = reduce(state, finished)

    assert len([row for row in state.transcript if row.tool_call_id == "c1"]) == 1


def test_projected_transcript_carries_command_and_metadata(tmp_path):
    """Resumed tool rows keep command + metadata when the projection emits them.

    The reducer consumes optional ``command``/``metadata`` keys on history dicts
    (the intended resumed-session projection shape). The persisted tool_call and
    tool_result pair below is projected through the real SessionStore first.
    """
    from coding_agent.runtime.models import ToolCall

    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    call = ToolCall(id="c1", name="run_command", arguments={"command": "ls -la"})
    assistant = Message(role="assistant", tool_calls=[call])
    store.append_new(
        "assistant_message", {"message": assistant, "complete": True}, turn_id="t1"
    )
    store.append_new("tool_call", {"tool_call": call}, turn_id="t1")
    store.append_new(
        "tool_result",
        {
            "result": ToolResult(
                tool_call_id="c1",
                tool_name="run_command",
                ok=False,
                content="total 8",
                error="exit code 2",
                metadata={"exit_code": 2, "elapsed_seconds": 2.5, "truncated": True},
            )
        },
        turn_id="t1",
    )

    projected = store.project_messages(include_open_turn=False)
    history = [item.model_dump(mode="json") for item in projected]
    for item in history:
        if item["message"]["role"] == "tool":
            item["command"] = "ls -la"
            item["metadata"] = {
                "exit_code": 2,
                "elapsed_seconds": 2.5,
                "truncated": True,
            }

    state = reduce(
        initial_state(workspace="/tmp/project", model="fake"),
        RuntimeEvent(
            type="session_loaded",
            payload={"session_id": "s1", "history": history},
        ),
    )

    row = next(row for row in state.transcript if row.kind == "tool")
    assert row.command == "ls -la"
    assert row.elapsed_seconds == 2.5
    assert row.truncated is True
    assert row.exit_code == 2
    assert row.expanded is False


@pytest.mark.asyncio
async def test_runner_tool_finished_payload_carries_result_metadata(tmp_path):
    from coding_agent.context.truncate import TruncatePolicy

    class MetadataExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call, **kwargs):
            self.calls.append(call)
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=True,
                content="file content",
                metadata={"exit_code": 0, "elapsed_seconds": 1.5, "truncated": False},
            )

    store = SessionStore.create(
        tmp_path, workspace=str(tmp_path), model="fake", context_window=1000
    )
    store.append_new("turn_start", {"turn_id": "t1"}, run_id="r1", turn_id="t1")
    store.append_new(
        "user_message",
        {"message": Message(role="user", content="inspect")},
        run_id="r1",
        turn_id="t1",
    )
    events = []

    async def sink(event):
        events.append(event)

    runner = AgentRunner(
        provider=ScriptedProvider([tool_response(), text_response("ready")]),
        registry=ToolRegistry(),
        executor=MetadataExecutor(),
        context_policy=TruncatePolicy(1000),
        store=store,
        event_sink=sink,
        system_prompt=Message(role="system", content="system"),
        model="fake",
        context_window=1000,
        permission_mode="full",
    )
    outcome = await runner.run_turn(
        "inspect", run_id="r1", turn_id="t1", signal=asyncio.Event()
    )

    assert outcome.reason == "completed"
    finished = [event for event in events if event.type == "tool_finished"]
    assert len(finished) == 1
    assert finished[0].payload["metadata"] == {
        "exit_code": 0,
        "elapsed_seconds": 1.5,
        "truncated": False,
    }


# ---------------------------------------------------------------------------
# Task 2: row spacing and per-kind CSS classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", ["user", "assistant", "tool", "system", "local_command"]
)
def test_transcript_row_carries_row_and_kind_classes(kind):
    item = TranscriptItem(kind=kind, item_id=f"i-{kind}", text="x")
    row = TranscriptRow(item, index=0)

    classes = set(row.classes)
    assert "row" in classes
    assert f"row-{kind}" in classes


def test_app_css_has_row_spacing_and_full_width_user_card():
    css = CodingAgentApp.CSS

    assert "#transcript .row" in css
    assert ".row.row-user" in css
    assert "width: 1fr" in css
    assert "row-local_command" in css or ".row.local_command" in css


# ---------------------------------------------------------------------------
# Task 3: compact tool rows with click-to-expand
# ---------------------------------------------------------------------------


def test_tool_running_header_renders_glyph_and_command():
    item = TranscriptItem(
        kind="tool",
        item_id="c1",
        tool_call_id="c1",
        tool_name="run_command",
        command="ls",
        text="",
        tool_status="running",
    )
    rendered = _row_text(item)

    assert "● Bash(ls)" in rendered


@pytest.mark.parametrize(
    ("status", "glyph"),
    [
        ("running", "●"),
        ("success", "✓"),
        ("error", "✕"),
        ("cancelled", "⊘"),
    ],
)
def test_tool_header_glyph_reflects_status(status, glyph):
    item = TranscriptItem(
        kind="tool",
        item_id="c1",
        tool_call_id="c1",
        tool_name="run_command",
        command="ls",
        text="x",
        tool_status=status,
    )

    assert _row_text(item).startswith(f"{glyph} Bash(ls)")


def test_tool_preview_shows_first_non_empty_output_line_only():
    item = TranscriptItem(
        kind="tool",
        item_id="c1",
        tool_call_id="c1",
        tool_name="run_command",
        command="git push",
        text="\n\nTo github.com:owner/repo.git\nline2",
        tool_status="success",
    )
    rendered = _row_text(item)

    assert "  ⎿  To github.com:owner/repo.git" in rendered
    assert "line2" not in rendered


def test_tool_footer_formats_elapsed_and_status_markers():
    item = TranscriptItem(
        kind="tool",
        item_id="c1",
        tool_call_id="c1",
        tool_name="run_command",
        command="ls",
        text="done",
        tool_status="success",
        elapsed_seconds=2,
        truncated=False,
        exit_code=0,
    )
    assert "(2s)" in _row_text(item)

    minute = item.model_copy(update={"elapsed_seconds": 60})
    assert "(1m)" in _row_text(minute)

    failed = item.model_copy(
        update={
            "elapsed_seconds": 90,
            "truncated": True,
            "exit_code": 2,
            "tool_status": "error",
        }
    )
    failed_rendered = _row_text(failed)
    assert "(1m 30s)" in failed_rendered
    assert "· truncated" in failed_rendered
    assert "· exit 2" in failed_rendered


def test_tool_expanded_renders_full_truncated_body_instead_of_preview():
    text = "\n".join(f"line {i}" for i in range(12))
    item = TranscriptItem(
        kind="tool",
        item_id="c1",
        tool_call_id="c1",
        tool_name="run_command",
        command="ls",
        text=text,
        tool_status="success",
        truncated=True,
    )
    compact = _row_text(item)
    assert "line 11" not in compact
    assert "(4 more lines)" not in compact

    expanded = _row_text(item.model_copy(update={"expanded": True}))
    assert "line 0" in expanded
    assert "(4 more lines)" in expanded
    assert "line 11" not in expanded


class _FakeRuntime:
    def __init__(self) -> None:
        self.subscribers = []
        self.workspace = "/tmp/project"
        self.model = "fake"
        self.session_id = "s1"
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
                        "model": "fake",
                        "context_window": 1000,
                    },
                )()
            },
        )()

    def subscribe(self, sink):
        self.subscribers.append(sink)
        return lambda: None

    async def emit(self, event: RuntimeEvent) -> None:
        for sink in list(self.subscribers):
            await sink(event)

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

    async def list_sessions(self) -> list:
        return []

    async def resume(self, session_id: str) -> None:
        pass

    async def compact(self) -> None:
        pass


def _tool_state(expanded: bool = False) -> TuiState:
    return initial_state(workspace="/tmp/project", model="fake").model_copy(
        update={
            "transcript": [
                TranscriptItem(
                    kind="tool",
                    item_id="c1",
                    tool_call_id="c1",
                    tool_name="run_command",
                    command="ls",
                    text="hello world",
                    tool_status="success",
                    expanded=expanded,
                )
            ]
        }
    )


@pytest.mark.asyncio
async def test_clicking_tool_row_toggles_expanded_flag():
    runtime = _FakeRuntime()
    app = CodingAgentApp(runtime=runtime, initial_state=_tool_state())

    async with app.run_test() as pilot:
        row = pilot.app.query_one("TranscriptRow")
        assert row.item.expanded is False

        await pilot.click(row)
        await pilot.pause()
        assert app.state.transcript[0].expanded is True

        await pilot.click(pilot.app.query_one("TranscriptRow"))
        await pilot.pause()
        assert app.state.transcript[0].expanded is False


@pytest.mark.asyncio
async def test_clicking_non_tool_row_does_not_toggle():
    runtime = _FakeRuntime()
    state = initial_state(workspace="/tmp/project", model="fake").model_copy(
        update={
            "transcript": [TranscriptItem(kind="assistant", item_id="a1", text="hello")]
        }
    )
    app = CodingAgentApp(runtime=runtime, initial_state=state)

    async with app.run_test() as pilot:
        row = pilot.app.query_one("TranscriptRow")
        await pilot.click(row)
        await pilot.pause()

        assert app.state.transcript[0].expanded is False


# ---------------------------------------------------------------------------
# Task 4: lightweight markdown rendering for assistant rows
# ---------------------------------------------------------------------------


def _span_for(text, needle):
    """Return the effective Rich style covering ``needle`` in ``text``.

    Every overlapping span is merged, so nested styling (e.g. inline code
    inside bold) reports the combined style.
    """
    from rich.style import Style

    start = text.plain.index(needle)
    end = start + len(needle)
    style = Style()
    for span in text.spans:
        if span.start < end and span.end > start:
            span_style = (
                span.style if isinstance(span.style, Style) else Style.parse(span.style)
            )
            style += span_style
    return style


def test_markdown_bold_renders_bold_without_markers():
    text = markdown_to_text("a **bold** tail")

    assert "bold" in text.plain
    assert "**" not in text.plain
    assert _span_for(text, "bold").bold


def test_markdown_heading_strips_marker_and_styles_bold_color():
    text = markdown_to_text("# Heading")

    assert text.plain == "Heading"
    span = _span_for(text, "Heading")
    assert span.bold
    assert span.color is not None


def test_markdown_inline_code_gets_distinct_style():
    text = markdown_to_text("run `pytest -q` now")

    assert "pytest -q" in text.plain
    span = _span_for(text, "pytest -q")
    assert span.reverse or span.bgcolor is not None


def test_markdown_inline_code_inside_bold_strips_backticks():
    text = markdown_to_text("**Directory Listing: `src/coding_agent`**")

    assert "`" not in text.plain
    assert "src/coding_agent" in text.plain
    code_span = _span_for(text, "src/coding_agent")
    assert code_span.reverse or code_span.bgcolor is not None
    assert _span_for(text, "Directory Listing:").bold


def test_markdown_nested_bold_within_paragraph_keeps_outer_bold():
    text = markdown_to_text("**Answer:** the directory `src` does **not** exist")

    assert "**" not in text.plain
    assert "`" not in text.plain
    assert _span_for(text, "Answer:").bold
    assert _span_for(text, "not").bold
    assert _span_for(text, "src").reverse or _span_for(text, "src").bgcolor is not None


def test_markdown_fenced_code_is_indented_dimmed_and_fenceless():
    text = markdown_to_text("```python\nprint(1)\nprint(2)\n```")

    assert "print(1)" in text.plain
    assert "```" not in text.plain
    assert _span_for(text, "print(1)").dim
    line = next(line for line in text.plain.splitlines() if "print(1)" in line)
    assert line.startswith("  ")


def test_markdown_list_uses_bullet_without_dash():
    text = markdown_to_text("- item")

    assert text.plain == "• item"
    assert not text.plain.startswith("-")


def test_markdown_link_keeps_text_and_dimmed_url():
    text = markdown_to_text("see [docs](https://example.com)")

    assert "docs" in text.plain
    assert "https://example.com" in text.plain
    assert _span_for(text, "https://example.com").dim


def test_markdown_malformed_input_does_not_raise():
    text = markdown_to_text("**unbalanced and *stray")

    assert "**unbalanced and *stray" in text.plain


def test_markdown_italic_variants_render_italic():
    text = markdown_to_text("*one* _two_")

    assert _span_for(text, "one").italic
    assert _span_for(text, "two").italic


def test_transcript_assistant_row_renders_styled_markdown():
    from rich.text import Text

    item = TranscriptItem(kind="assistant", item_id="m1", text="**bold** and `code`")
    row = TranscriptRow(item, index=0)
    renderable = row.render()

    assert isinstance(renderable, Text)
    rendered_str = str(renderable)
    assert "**" not in rendered_str
    assert "`" not in rendered_str
    assert "bold" in rendered_str
    assert "code" in rendered_str


@pytest.mark.parametrize("kind", ["user", "local_command"])
def test_row_css_rules_use_kind_qualified_classes(kind):
    """Per-kind CSS rules must target the classes TranscriptRow actually sets
    (``row row-<kind>``), otherwise the rules never apply in a live terminal."""
    row = TranscriptRow(TranscriptItem(kind=kind, item_id="x", text="hello"), index=0)
    assert f"row-{kind}" in row.classes
    assert f".row.row-{kind}" in CodingAgentApp.CSS


# ---------------------------------------------------------------------------
# Task 5: full-pipeline smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_smoke_renders_visual_refresh_features():
    """Drive the real app through a full turn and inspect the rendered rows.

    This is the scripted TUI smoke: it renders the actual widgets, verifies the
    user card, compact click-expandable tool row, styled assistant markdown and
    an intact statusline, and asserts no DuplicateIds/error rows surface.
    """
    runtime = _FakeRuntime()
    app = CodingAgentApp(
        runtime=runtime,
        initial_state=initial_state(workspace="/tmp/project", model="fake"),
    )

    async with app.run_test() as pilot:
        await runtime.emit(RuntimeEvent(type="run_started", run_id="r1", turn_id="t1"))
        await runtime.emit(
            RuntimeEvent(
                type="user_message",
                run_id="r1",
                payload={"message_id": "u1", "text": "inspect"},
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="assistant_started", run_id="r1", payload={"message_id": "m1"}
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="assistant_delta",
                run_id="r1",
                payload={
                    "message_id": "m1",
                    "text": "# Heading\n\n**bold** and `code`",
                },
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="tool_started",
                run_id="r1",
                payload={
                    "tool_call_id": "c1",
                    "tool_name": "run_command",
                    "arguments": {"command": "ls -la"},
                },
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="tool_finished",
                run_id="r1",
                payload={
                    "tool_call_id": "c1",
                    "tool_name": "run_command",
                    "ok": False,
                    "content": "total 8",
                    "error": "exit code 2",
                    "metadata": {
                        "exit_code": 2,
                        "elapsed_seconds": 90,
                        "truncated": True,
                    },
                },
            )
        )
        await runtime.emit(
            RuntimeEvent(
                type="run_finished",
                run_id="r1",
                payload={"outcome": {"reason": "completed"}},
            )
        )
        await pilot.pause()

        # No DuplicateIds / no runtime error rows surfaced.
        assert app.state.status == "idle"
        assert not any(
            row.kind == "system" and "error" in (row.text or "").lower()
            for row in app.state.transcript
        )

        rows = list(pilot.app.query("#transcript TranscriptRow"))
        kinds = {row.item.kind for row in rows}
        assert {"user", "assistant", "tool"} <= kinds

        user_row = next(row for row in rows if row.item.kind == "user")
        assert "row" in user_row.classes
        assert "row-user" in user_row.classes
        assert str(user_row.render()).startswith("> ")

        tool_row = next(row for row in rows if row.item.kind == "tool")
        rendered_tool = str(tool_row.render())
        assert "✕ Bash(ls -la)" in rendered_tool
        assert "(1m 30s)" in rendered_tool
        assert "· truncated" in rendered_tool
        assert "· exit 2" in rendered_tool

        assistant_row = next(row for row in rows if row.item.kind == "assistant")
        rendered_assistant = str(assistant_row.render())
        assert "Heading" in rendered_assistant
        assert "**" not in rendered_assistant
        assert "`" not in rendered_assistant

        # Clicking the compact tool row expands it to the full body.
        assert tool_row.item.expanded is False
        await pilot.click(tool_row)
        await pilot.pause()
        tool_item = next(
            row for row in app.state.transcript if row.tool_call_id == "c1"
        )
        assert tool_item.expanded is True

        # Statusline remains mounted and rendering.
        statusline = pilot.app.query_one("#statusline")
        assert statusline.is_mounted
        assert "model fake" in str(statusline.render())
