"""Pure parsing utilities for local TUI slash commands."""

import shlex
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandSuggestion:
    """A command shown by the composer palette."""

    name: str
    description: str
    usage: str = ""


_COMMANDS = (
    CommandSuggestion("clear", "Clear the visible transcript"),
    CommandSuggestion("compact", "Compact the current context"),
    CommandSuggestion("context", "Show context usage"),
    CommandSuggestion("exit", "Exit the application"),
    CommandSuggestion("help", "List available commands"),
    CommandSuggestion("new", "Start a new session"),
    CommandSuggestion("permission", "Show or change permission mode"),
    CommandSuggestion("quit", "Exit the application"),
    CommandSuggestion("resume", "Resume a session by id or prefix"),
    CommandSuggestion("session", "Choose a session to resume"),
    CommandSuggestion("undo", "Undo the last file write/edit", usage="/undo"),
)
SUPPORTED_COMMANDS = frozenset(item.name for item in _COMMANDS)


@dataclass(frozen=True, slots=True)
class Command:
    """A parsed local command and its positional arguments."""

    name: str
    args: list[str]


def command_suggestions(prefix: str) -> tuple[CommandSuggestion, ...]:
    """Return palette entries matching a slash command prefix."""
    prefix = prefix.strip()
    prefix = prefix.removeprefix("/")
    prefix = prefix.split(maxsplit=1)[0].lower() if prefix else ""
    return tuple(item for item in _COMMANDS if item.name.startswith(prefix))


def parse_command(text: str) -> Command | None:
    """Parse a slash-prefixed input, or return ``None`` for a prompt.

    Parsing is intentionally local: this function does not validate command
    arguments, invoke runtime methods, or persist input. Unknown command names
    are returned to the caller so it can produce a local notice.
    """
    if not text.lstrip().startswith("/"):
        return None

    try:
        tokens = shlex.split(text, comments=False)
    except ValueError:
        tokens = text.strip().split()

    if not tokens or not tokens[0].startswith("/"):
        return None

    return Command(name=tokens[0][1:], args=tokens[1:])
