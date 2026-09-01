from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from coding_agent.runtime.models import RuntimeStatus, Usage
from coding_agent.session.models import ApprovalRequest, SessionSummary
from coding_agent.tui.commands import command_suggestions
from coding_agent.tui.state import TranscriptItem, TuiState


class TranscriptRow(Static):
    """A single immutable transcript snapshot row."""

    def __init__(self, item: TranscriptItem, *, index: int) -> None:
        row_id = _row_id(item, index)
        super().__init__(
            _row_text(item),
            id=row_id,
            markup=False,
            classes=f"row row-{item.kind}",
        )
        self.item = item


class TranscriptView(VerticalScroll):
    """Scrollable transcript renderer."""

    def __init__(self, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self._rendered_text = ""

    def render(self) -> Text:
        return Text(self._rendered_text)

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
    """Session picker that returns the selected persisted session id."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, sessions: list[SessionSummary]) -> None:
        super().__init__()
        self.sessions = sessions

    def compose(self) -> ComposeResult:
        options = [
            Option(_session_text(summary), id=summary.id) for summary in self.sessions
        ]
        with Container(id="session-selector"):
            yield Static("Sessions", id="session-selector-title")
            yield OptionList(*options, id="session-options", markup=False)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id:
            self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


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


class ApprovalScreen(ModalScreen[str]):
    """Focused approval view; it only returns the user's decision."""

    BINDINGS: ClassVar = [
        ("a", "approve", "Approve"),
        ("d", "deny", "Deny"),
        ("escape", "deny", "Deny"),
    ]

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

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
            yield Button("Approve", id="approve", variant="success")
            yield Button("Deny", id="deny", variant="error")

    def action_approve(self) -> None:
        self.dismiss("approve")

    def action_deny(self) -> None:
        self.dismiss("deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("approve" if event.button.id == "approve" else "deny")


def format_statusline(
    state: TuiState,
    *,
    runtime_status: RuntimeStatus | None = None,
    usage: Usage | None = None,
    width: int | None = None,
) -> str:
    """Format required metadata in one line, hiding low-priority fields."""
    runtime_status = runtime_status or RuntimeStatus(status=state.status)
    usage = usage or getattr(runtime_status, "usage", None)
    used = state.context_used
    status = state.status
    runtime_window = getattr(runtime_status, "context_window", None)
    window = state.context_window or runtime_window or 0
    remaining = max(0, window - used) if window else None
    context = ""
    if window:
        estimated = state.context_estimated or getattr(
            runtime_status, "context_estimated", False
        )
        context = f"ctx {used}/{remaining}/{window}"
        context += " (estimated)" if estimated else " (configured)"
    usage_text = ""
    if usage is not None:
        usage_text = f"in {usage.input_tokens} out {usage.output_tokens}"
    fields = [
        state.workspace,
        f"branch {state.git_branch or '-'}",
        f"model {state.model}",
        f"reasoning {state.reasoning or '-'}",
        f"perm {state.policy}",
        f"session {_short_id(state.session_id)}",
        status,
        context,
        usage_text,
    ]
    fields = [field for field in fields if field]
    if width is None:
        return "  ".join(fields)
    if width <= 0:
        return ""
    # Preserve identity and lifecycle fields first; progressively hide details.
    while fields and len("  ".join(fields)) > width:
        removable = (
            usage_text,
            context,
            f"reasoning {state.reasoning or '-'}",
            f"branch {state.git_branch or '-'}",
            state.workspace,
            f"session {_short_id(state.session_id)}",
            f"model {state.model}",
        )
        for candidate in removable:
            if candidate and candidate in fields:
                fields.remove(candidate)
                break
        else:
            break
    return "  ".join(fields)[:width]


def _session_text(summary: SessionSummary) -> str:
    updated = _format_datetime(summary.updated_at)
    return (
        f"{_short_id(summary.id)}  {updated}  {summary.workspace}  {summary.title[:80]}"
    )


def _format_datetime(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


_TOOL_MAX_LINES = 8
_TOOL_MAX_LINE_CHARS = 200


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


def _row_text(item: TranscriptItem) -> str:
    if item.kind == "user":
        return f"> {item.text}"
    if item.kind == "local_command":
        return f"$ {item.text}"
    if item.kind == "assistant":
        return item.text
    if item.kind == "tool":
        status = item.tool_status or "pending"
        name = item.tool_name or "tool"
        return f"[{status}] {name}: {_truncate_tool_output(item.text)}".rstrip()
    return f"[{item.level or 'notice'}] {item.text}"


def _row_id(item: TranscriptItem, index: int) -> str:
    kind = re.sub(r"[^a-zA-Z0-9_-]", "-", item.kind)
    identifier = re.sub(r"[^a-zA-Z0-9_-]", "-", item.item_id).strip("-")
    if not identifier:
        identifier = str(index)
    return f"row-{kind}-{identifier}"


def _short_id(value: str | None) -> str:
    return value[:8] if value else "-"
