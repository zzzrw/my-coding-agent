from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Iterable
from datetime import datetime
from difflib import unified_diff
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Select, Static, TextArea
from textual.widgets.option_list import Option

from coding_agent.policy.memory import Scope
from coding_agent.runtime.models import RuntimeStatus, Usage
from coding_agent.session.models import ApprovalRequest, SessionSummary
from coding_agent.tui.commands import command_suggestions
from coding_agent.tui.state import TranscriptItem, TuiState


class TranscriptRow(Static):
    """A single immutable transcript snapshot row."""

    class ToolRowClicked(Message):
        """Mouse-clicked a compact tool row to expand or collapse it."""

        def __init__(self, item_id: str) -> None:
            self.item_id = item_id
            super().__init__()

    def __init__(self, item: TranscriptItem, *, index: int) -> None:
        row_id = _row_id(item, index)
        self._renderable: object = (
            markdown_to_text(item.text) if item.kind == "assistant" else _row_text(item)
        )
        super().__init__(
            self._renderable,
            id=row_id,
            markup=False,
            classes=f"row row-{item.kind}",
        )
        self.item = item

    def render(self) -> object:
        # Return the raw renderable so tests can inspect it without an active
        # App; Textual still visualizes it with markup disabled in _render().
        return self._renderable

    def on_click(self, event: events.Click) -> None:
        if self.item.kind == "tool":
            event.stop()
            self.post_message(self.ToolRowClicked(self.item.item_id))


class TranscriptView(VerticalScroll):
    """Scrollable transcript renderer.

    Rows are mounted as styled child widgets; ``_rendered_text`` is kept only
    as a plain-text snapshot for the ``renderable_text`` property and tests.
    No ``render()`` override is used: painting a second raw fallback text on a
    container that also mounts styled children produced offset overlapping
    duplicates in a real terminal.
    """

    def __init__(self, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self._rendered_text = ""

    @property
    def renderable_text(self) -> str:
        return self._rendered_text

    async def render_state(self, items: Iterable[TranscriptItem]) -> None:
        """Replace rendered rows with a state snapshot."""
        await self.remove_children()
        rows = [TranscriptRow(item, index=index) for index, item in enumerate(items)]
        self._rendered_text = "\n".join(_row_text(row.item) for row in rows)
        if rows:
            await self.mount_all(rows)


class CommandComposer(Vertical):
    """The composer region containing its input and transient command palette."""

    def __init__(self, text: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.text = text

    def compose(self) -> ComposeResult:
        yield SubmitTextArea(
            self.text,
            id="composer-input",
            placeholder="Ask coding-agent...",
        )
        yield OptionList(id="command-palette")


class SubmitTextArea(TextArea):
    """TextArea that submits plain Enter and preserves modified Enter keys."""

    class Submitted(Message):
        def __init__(self, text_area: SubmitTextArea, text: str) -> None:
            self.text_area = text_area
            self.text = text
            super().__init__()

    class ComposerHistoryRequested(Message):
        """Up/Down pressed while the composer is focused (palette hidden)."""

        def __init__(self, offset: int) -> None:
            self.offset = offset
            super().__init__()

    def on_mount(self) -> None:
        self._update_command_palette()

    async def _on_key(self, event) -> None:
        if event.key == "escape" and self._palette_visible:
            event.stop()
            event.prevent_default()
            self._set_palette_visible(False)
            return
        if event.key in {"up", "down"} and self._palette_visible:
            event.stop()
            event.prevent_default()
            palette = self.app.query_one("#command-palette", OptionList)
            if palette.option_count:
                delta = -1 if event.key == "up" else 1
                current = palette.highlighted if palette.highlighted is not None else 0
                palette.highlighted = (current + delta) % palette.option_count
            return
        if event.key in {"up", "down"} and not self._palette_visible:
            event.stop()
            event.prevent_default()
            self.post_message(
                self.ComposerHistoryRequested(-1 if event.key == "up" else 1)
            )
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            palette = self.app.query_one("#command-palette", OptionList)
            if self._palette_visible and palette.highlighted is not None:
                option = palette.get_option_at_index(palette.highlighted)
                self.text = option.id or option.prompt.plain.split()[0]
            self.post_message(self.Submitted(self, self.text))
            return
        await super()._on_key(event)

    def watch_text(self, text: str) -> None:
        self._update_command_palette()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self:
            self._update_command_palette()

    @property
    def _palette_visible(self) -> bool:
        return self.text.startswith("/") and " " not in self.text.strip()

    def _set_palette_visible(self, visible: bool) -> None:
        palette = self.app.query_one("#command-palette", OptionList)
        palette.display = visible

    def _update_command_palette(self) -> None:
        if not self.is_mounted:
            return
        palette = self.app.query_one("#command-palette", OptionList)
        suggestions = command_suggestions(
            self.text[1:] if self.text.startswith("/") else ""
        )
        palette.set_options(
            [
                Option(f"/{item.name}  {item.description}", id=f"/{item.name}")
                for item in suggestions
            ]
        )
        palette.highlighted = 0 if suggestions else None
        palette.display = self._palette_visible and bool(suggestions)


class StatusLine(Static):
    """Compact one-line status renderer."""

    def render_state(
        self, state: TuiState, runtime_status: RuntimeStatus | None = None
    ) -> None:
        self.update(
            format_statusline(
                state, runtime_status=runtime_status, width=self.size.width or None
            )
        )


class SessionSelector(ModalScreen[str]):
    """Session picker that returns the selected persisted session id.

    The cached session list is filtered to the current ``workspace`` (an exact
    ``SessionSummary.workspace`` match) unless the user toggles ``browse all``
    via the footer button or the ``b`` binding; toggling re-filters the cached
    list without re-listing. When ``workspace`` is ``None`` every session is
    shown.
    """

    BINDINGS: ClassVar = [
        ("escape", "cancel", "Cancel"),
        ("b", "toggle_filter", "Browse all"),
    ]

    def __init__(
        self, sessions: list[SessionSummary], workspace: str | None = None
    ) -> None:
        super().__init__()
        self.sessions = sessions
        self._workspace = workspace
        self._browse_all = False

    def compose(self) -> ComposeResult:
        options = [
            Option(_session_text(summary), id=summary.id)
            for summary in self.visible_sessions()
        ]
        with Container(id="session-selector"):
            yield Static("Sessions", id="session-selector-title")
            yield OptionList(*options, id="session-options", markup=False)
            yield Button(self.toggle_label(), id="session-toggle")

    def visible_sessions(self) -> list[SessionSummary]:
        """Sessions shown: the current workspace unless browsing all."""
        if self._browse_all or not self._workspace:
            return list(self.sessions)
        return [
            summary for summary in self.sessions if summary.workspace == self._workspace
        ]

    def toggle_label(self) -> str:
        """Footer toggle label describing the currently shown scope."""
        if self._browse_all or not self._workspace:
            return "[browse: all]"
        return f"[browse: {self._workspace}]"

    def toggle_filter(self) -> None:
        """Flip between the current workspace and every session."""
        self._browse_all = not self._browse_all
        if not self.is_mounted:
            return
        self.query_one("#session-options", OptionList).set_options(
            [
                Option(_session_text(summary), id=summary.id)
                for summary in self.visible_sessions()
            ]
        )
        toggle = self.query_one("#session-toggle", Button)
        toggle.label = self.toggle_label()

    def action_toggle_filter(self) -> None:
        self.toggle_filter()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "session-toggle":
            self.toggle_filter()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id:
            self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    """Modal overlay listing commands with usage, keybindings, and permissions.

    ``compose`` yields a single bordered ``Static`` built from
    ``help_overlay_text()``; ``body`` keeps the renderable accessible so tests
    can inspect the content without an active App.
    """

    BINDINGS: ClassVar = [("escape", "close_help", "Close")]

    def __init__(self) -> None:
        super().__init__()
        self.body = help_overlay_text()

    def compose(self) -> ComposeResult:
        yield Static(self.body, id="help", markup=False)

    def action_close_help(self) -> None:
        self.dismiss(None)


_HELP_KEYBINDINGS: tuple[tuple[str, str], ...] = (
    ("Ctrl+C", "abort the current run"),
    ("↑/↓", "recall composer history"),
    ("?", "open this help"),
)
"""Global keybindings shown in the help overlay."""

_PERMISSION_LEGEND: tuple[tuple[str, str], ...] = (
    ("default", "approve outside-workspace actions"),
    ("workspace", "run tools inside the workspace without approval"),
    ("full", "run all tools without approval"),
)
"""Permission-mode legend shown in the help overlay."""


def help_overlay_text() -> Text:
    """Build the styled help overlay body: commands, keybindings, permissions."""
    body = Text()
    body.append("Commands", style="bold")
    body.append("\n")
    for item in command_suggestions(""):
        usage = item.usage or f"/{item.name}"
        body.append(f"  /{item.name:<11}")
        body.append(usage, style="dim")
        body.append(f"\n     {item.description}\n")
    body.append("Keybindings", style="bold")
    body.append("\n")
    for key, description in _HELP_KEYBINDINGS:
        body.append(f"  {key:<10}")
        body.append(description)
        body.append("\n")
    body.append("Permissions", style="bold")
    body.append("\n")
    for mode, description in _PERMISSION_LEGEND:
        body.append(f"  {mode:<10}")
        body.append(description)
        body.append("\n")
    return body


class PermissionFullScreen(ModalScreen[bool]):
    """Visible high-risk confirmation before enabling unrestricted permissions."""

    BINDINGS: ClassVar = [
        ("enter", "approve", "Enable full"),
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="permission-full"):
            yield Static(
                "Enable full permission?\n"
                "This permits ordinary tools outside the workspace without approval.",
                id="permission-full-details",
                markup=False,
            )
            yield Button("Enable full", id="permission-full-approve", variant="error")
            yield Button("Cancel", id="permission-full-cancel")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "permission-full-approve")


class PermissionModeScreen(ModalScreen[str]):
    """Discoverable permission-mode selector covering default/workspace/full."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, current: str) -> None:
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        with Container(id="permission-mode"):
            yield Static("Permission mode", id="permission-mode-title")
            yield OptionList(
                *[
                    Option(
                        f"{mode}{'  (current)' if mode == self.current else ''}",
                        id=mode,
                    )
                    for mode in ("default", "workspace", "full")
                ],
                id="permission-mode-options",
                markup=False,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id:
            self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApprovalScreen(ModalScreen[tuple[str, Scope, str | None]]):
    """Focused approval view with a diff preview, remember scope, and feedback.

    For ``write_file``/``edit_file`` requests it shows a colorized unified diff
    of the proposed change. A remember selector picks the ``DecisionMemory``
    scope (``once|turn|session|always``) applied to the decision, and an optional
    feedback input is passed to the model when the request is denied.

    Dismisses with ``(decision, remember, feedback)``.
    """

    BINDINGS: ClassVar = [
        ("a", "approve", "Approve"),
        ("d", "deny", "Deny"),
        ("escape", "deny", "Deny"),
    ]

    def __init__(self, request: ApprovalRequest, *, workspace: Path | str) -> None:
        super().__init__()
        self.request = request
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        request = self.request
        details = (
            f"Approval required\n"
            f"Tool: {request.tool_name}\n"
            f"Risk: {request.risk_level}\n"
            f"Arguments: {request.arguments}\n"
            f"Reason: {request.reason}"
        )
        with Container(id="approval"):
            yield Static(details, id="approval-details", markup=False)
            if request.tool_name in {"write_file", "edit_file"}:
                yield Static(
                    render_approval_diff(request, workspace=self.workspace),
                    id="approval-diff",
                    markup=False,
                )
            yield Static("Remember decision", id="approval-remember-label")
            yield Select(
                [
                    ("once", "once"),
                    ("turn", "turn"),
                    ("session", "session"),
                    ("always", "always"),
                ],
                value="once",
                id="approval-remember",
            )
            yield Input(
                placeholder="Feedback on denial (optional)",
                id="approval-feedback",
            )
            yield Button("Approve", id="approve", variant="success")
            yield Button("Deny", id="deny", variant="error")

    def _result(self, decision: str) -> tuple[str, Scope, str | None]:
        remember = self.query_one("#approval-remember", Select).value or "once"
        feedback_value = self.query_one("#approval-feedback", Input).value
        feedback = feedback_value.strip() if isinstance(feedback_value, str) else ""
        return (decision, remember, feedback or None)

    def action_approve(self) -> None:
        self.dismiss(self._result("approve"))

    def action_deny(self) -> None:
        self.dismiss(self._result("deny"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "approve":
            self.action_approve()
        else:
            self.action_deny()


_DIFF_MAX_LINES = 40
"""Maximum unified-diff lines rendered in the approval panel."""


def render_approval_diff(request: ApprovalRequest, *, workspace: Path | str) -> Text:
    """Render a colorized unified diff of a proposed file mutation.

    ``write_file`` replaces the file with ``arguments["content"]``; ``edit_file``
    applies the ``old_text`` → ``new_text`` substitution to the current content.
    Added lines are green, removed lines red, and headers/hunk line numbers are
    dim. Diffs longer than ``_DIFF_MAX_LINES`` lines are truncated with a
    ``… (N more lines)`` note. Missing files diff as all-added content.
    """
    if request.tool_name not in {"write_file", "edit_file"}:
        return Text("")
    raw_path = request.arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return Text("")
    user_path = Path(raw_path).expanduser()
    path = user_path if user_path.is_absolute() else Path(workspace) / user_path
    try:
        current = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        current = ""
    if request.tool_name == "write_file":
        proposed = str(request.arguments.get("content", ""))
    else:
        old_text = str(request.arguments.get("old_text", ""))
        new_text = str(request.arguments.get("new_text", ""))
        proposed = current.replace(old_text, new_text, 1) if old_text else current
    diff_lines = list(
        unified_diff(
            current.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    text = Text()
    shown = diff_lines[:_DIFF_MAX_LINES]
    for line in shown:
        _append_diff_line(text, line)
    if len(diff_lines) > _DIFF_MAX_LINES:
        text.append(
            f"… ({len(diff_lines) - _DIFF_MAX_LINES} more lines)",
            style="dim",
        )
    return text


def _append_diff_line(text: Text, line: str) -> None:
    """Append one unified-diff line, styling added/removed/header content."""
    if line.startswith("+"):
        text.append(line, style="green")
    elif line.startswith("-"):
        text.append(line, style="red")
    elif line.startswith("@@"):
        text.append(line, style="dim")
    else:
        text.append(line)


SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
"""Animated frames for the running statusline, advanced by the app timer."""

_PAUSED_GLYPH = "⏸"
"""Static glyph shown in place of the animated frame while approval is pending."""


def format_statusline(
    state: TuiState,
    *,
    runtime_status: RuntimeStatus | None = None,
    usage: Usage | None = None,
    width: int | None = None,
    now: float | None = None,
) -> Text:
    """Format required metadata in one line, hiding low-priority fields.

    Returns a styled ``rich.text.Text``: each ``key value`` field renders its
    key dimmed and its value emphasized (not dim), and the runtime ``status``
    field is colored by state (running cyan, error red, aborted dim, idle
    default). While a run is active the status field also carries the spinner
    frame (``SPINNER_FRAMES[spinner_frame % len(SPINNER_FRAMES)]``) and a live
    ``⏱`` elapsed timer derived from ``state.run_started_at``; ``waiting_approval``
    keeps the elapsed timer but shows a static pause glyph instead of an
    animated frame. ``now`` injects the monotonic reference so callers (tests)
    can render deterministically; it defaults to ``time.monotonic()``. Width
    truncation still operates on plain text lengths, so ``len(...) <= width``
    and substring checks behave as before.
    """
    runtime_status = runtime_status or RuntimeStatus(status=state.status)
    usage = usage or getattr(runtime_status, "usage", None)
    used = state.context_used
    status = _status_value(state, now)
    runtime_window = getattr(runtime_status, "context_window", None)
    window = state.context_window or runtime_window or 0
    remaining = max(0, window - used) if window else None
    context: tuple[str, str] | str = ""
    if window:
        estimated = state.context_estimated or getattr(
            runtime_status, "context_estimated", False
        )
        ctx_value = f"{used}/{remaining}/{window}"
        ctx_value += " (estimated)" if estimated else " (configured)"
        context = ("ctx", ctx_value)
    usage_text = ""
    if usage is not None:
        usage_text = f"in {usage.input_tokens} out {usage.output_tokens}"
    branch = state.git_branch or "-"
    fields: list[object] = [
        state.workspace,
        ("branch", branch),
        ("model", state.model),
        ("reasoning", state.reasoning or "-"),
        ("perm", state.policy),
        ("session", _short_id(state.session_id)),
        ("status", status, _status_style(state.status)),
        context,
        usage_text,
    ]
    fields = [field for field in fields if _field_text(field)]
    if width is None:
        return _statusline_text(fields)
    if width <= 0:
        return Text("")
    # Preserve identity and lifecycle fields first; progressively hide details.
    while fields and len("  ".join(_field_text(field) for field in fields)) > width:
        removable = (
            usage_text,
            _field_text(context),
            f"reasoning {state.reasoning or '-'}",
            f"branch {branch}",
            state.workspace,
            f"session {_short_id(state.session_id)}",
            f"model {state.model}",
        )
        for candidate in removable:
            if not candidate:
                continue
            index = next(
                (
                    i
                    for i, field in enumerate(fields)
                    if _field_text(field) == candidate
                ),
                None,
            )
            if index is not None:
                fields.pop(index)
                break
        else:
            break
    result = _statusline_text(fields)
    if len(result.plain) > width:
        result = result[:width]
    return result


def _status_value(state: TuiState, now: float | None) -> str:
    """Status field value, decorated with the spinner frame and elapsed time."""
    if state.status == "running":
        frame = SPINNER_FRAMES[state.spinner_frame % len(SPINNER_FRAMES)]
        return f"{frame} running ⏱{_format_elapsed(_elapsed_seconds(state, now))}"
    if state.status == "waiting_approval":
        # Elapsed keeps ticking while approval is pending, but the animation
        # pauses on a static glyph until the run resumes.
        return (
            f"{_PAUSED_GLYPH} waiting_approval "
            f"⏱{_format_elapsed(_elapsed_seconds(state, now))}"
        )
    return state.status


def _elapsed_seconds(state: TuiState, now: float | None) -> int:
    if state.run_started_at is None:
        return 0
    reference = time.monotonic() if now is None else now
    return max(0, int(reference - state.run_started_at))


def _field_text(field: object) -> str:
    """Plain text of a statusline field for width/removal logic."""
    if isinstance(field, tuple):
        if field[0] == "status":
            return field[1]
        return f"{field[0]} {field[1]}"
    return str(field)


def _status_style(status: str) -> str | None:
    if status == "running":
        return "cyan"
    if status == "error":
        return "red"
    if status == "aborted":
        return "dim"
    return None


def _statusline_text(fields: list[object]) -> Text:
    """Assemble the styled statusline Text from the surviving fields."""
    result = Text()
    for index, field in enumerate(fields):
        if index:
            result.append("  ")
        if isinstance(field, tuple) and field[0] == "status":
            result.append(field[1], style=field[2])
        elif isinstance(field, tuple):
            result.append(field[0], style="dim")
            result.append(" ")
            result.append(field[1])
        else:
            result.append(field)
    return result


def _session_text(summary: SessionSummary) -> str:
    updated = _format_datetime(summary.updated_at)
    return (
        f"{_short_id(summary.id)}  {updated}  {summary.workspace}  {summary.title[:80]}"
    )


def _format_datetime(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


_TOOL_MAX_LINES = 8
_TOOL_MAX_LINE_CHARS = 200
_TOOL_PREVIEW_CHARS = 160
_TOOL_GLYPHS = {
    "running": "●",
    "success": "✓",
    "error": "✕",
    "cancelled": "⊘",
}


def _truncate_tool_output(text: str) -> str:
    """Cap tool output to a bounded number of lines and line length."""
    lines = text.splitlines()
    if not lines:
        return text
    result: list[str] = []
    for line in lines[:_TOOL_MAX_LINES]:
        if len(line) > _TOOL_MAX_LINE_CHARS:
            line = line[:_TOOL_MAX_LINE_CHARS].rstrip() + "…"
        result.append(line)
    if len(lines) > _TOOL_MAX_LINES:
        result.append(f"… ({len(lines) - _TOOL_MAX_LINES} more lines)")
    return "\n".join(result)


def _tool_display_name(tool_name: str) -> str:
    """Map a tool name to its compact header label (run_command -> Bash)."""
    if tool_name == "run_command":
        return "Bash"
    return tool_name.capitalize()


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds compactly: 2s, 1m, 1m 30s."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes and secs:
        return f"{minutes}m {secs}s"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def _tool_header(item: TranscriptItem) -> str:
    glyph = _TOOL_GLYPHS.get(item.tool_status or "pending", "●")
    name = _tool_display_name(item.tool_name or "tool")
    if item.command:
        return f"{glyph} {name}({item.command})"
    return f"{glyph} {name}"


def _tool_preview(item: TranscriptItem) -> str | None:
    """Return the compact first non-empty output line, capped in length."""
    if item.tool_status == "running":
        return None
    lines = [line.strip() for line in item.text.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if len(first) > _TOOL_PREVIEW_CHARS:
        first = first[:_TOOL_PREVIEW_CHARS].rstrip() + "…"
    return f"  ⎿  {first}"


def _tool_footer(item: TranscriptItem) -> str | None:
    parts: list[str] = []
    if item.elapsed_seconds is not None:
        parts.append(f"({_format_elapsed(item.elapsed_seconds)})")
    if item.truncated:
        parts.append("· truncated")
    if item.exit_code not in (None, 0):
        parts.append(f"· exit {item.exit_code}")
    if item.retries:
        parts.append(f"· retried {item.retries}×")
    if not parts:
        return None
    return "  ⎿  " + " ".join(parts)


def _tool_row_text(item: TranscriptItem) -> str:
    lines = [_tool_header(item)]
    if item.expanded:
        body = _truncate_tool_output(item.text)
        if body.strip():
            for line in body.splitlines():
                lines.append(f"  ⎿  {line}")
    else:
        preview = _tool_preview(item)
        if preview is not None:
            lines.append(preview)
    footer = _tool_footer(item)
    if footer is not None:
        lines.append(footer)
    return "\n".join(lines)


_INLINE_TOKEN_RE = re.compile(
    r"\*\*[^*]+\*\*"
    r"|`[^`]+`"
    r"|\[[^\]]+\]\([^)]+\)"
    r"|(?<!\*)\*[^*\s][^*]*\*(?!\*)"
    r"|_[^_\s][^_]*_"
)


def _append_inline(text: Text, line: str) -> None:
    """Append one line to ``text``, styling inline markdown tokens.

    Tokens are matched by pattern and nested content is processed
    recursively, so `` `code` `` inside `` **bold** `` renders without
    literal backticks. Unbalanced tokens fall back to literal text.
    """
    pos = 0
    for match in _INLINE_TOKEN_RE.finditer(line):
        token = match.group(0)
        if match.start() > pos:
            text.append(line[pos : match.start()])
        if token.startswith("**"):
            sub = Text("", style="bold")
            _append_inline(sub, token[2:-2])
            text.append(sub)
        elif token.startswith("`"):
            text.append(token[1:-1], style="reverse")
        elif token.startswith("["):
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            if link is not None:
                text.append(link.group(1))
                text.append(f" ({link.group(2)})", style="dim")
            else:
                text.append(token)
        elif token.startswith(("*", "_")):
            text.append(token[1:-1], style="italic")
        else:
            text.append(token)
        pos = match.end()
    if pos < len(line):
        text.append(line[pos:])


def markdown_to_text(text: str) -> Text:
    """Convert a lightweight markdown subset into a styled Rich Text.

    Supports ATX headings, bold/italic/code, fenced code blocks, list prefixes
    and links. Malformed input never raises and falls back to literal text.
    """
    result = Text()
    lines = text.splitlines()
    in_fence = False
    first = True
    for line in lines:
        if not first:
            result.append("\n")
        first = False
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            result.append("  " + line, style="dim")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            result.append(heading.group(2), style="bold bright_cyan")
            continue
        bullet = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if bullet:
            result.append(bullet.group(1) + "• ")
            _append_inline(result, bullet.group(2))
            continue
        numbered = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if numbered:
            result.append(numbered.group(1) + numbered.group(2) + ". ")
            _append_inline(result, numbered.group(3))
            continue
        _append_inline(result, line)
    return result


def _row_text(item: TranscriptItem) -> str:
    if item.kind == "user":
        return f"> {item.text}"
    if item.kind == "local_command":
        return f"$ {item.text}"
    if item.kind == "assistant":
        return item.text
    if item.kind == "tool":
        return _tool_row_text(item)
    return f"[{item.level or 'notice'}] {item.text}"


def _row_id(item: TranscriptItem, index: int) -> str:
    kind = re.sub(r"[^a-zA-Z0-9_-]", "-", item.kind)
    identifier = re.sub(r"[^a-zA-Z0-9_-]", "-", item.item_id).strip("-")
    if not identifier:
        identifier = str(index)
    return f"row-{kind}-{identifier}"


def _short_id(value: str | None) -> str:
    return value[:8] if value else "-"


_GIT_BRANCH_TIMEOUT_SECONDS = 1.0


def detect_git_branch(workspace: str) -> str | None:
    """Return the workspace's current git branch, or None on any failure.

    Runs ``git -C <workspace> rev-parse --abbrev-ref HEAD`` with a short
    timeout. Never raises: a non-git directory, a missing ``git`` binary, or a
    slow repository all yield None so the statusline renders ``branch -``.
    """
    if not workspace:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", workspace, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_BRANCH_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None
