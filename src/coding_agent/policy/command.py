import re
import shlex
from pathlib import Path

from pydantic import BaseModel


class CommandClassification(BaseModel):
    catastrophic: bool = False
    outside_or_unknown: bool = False
    reason: str = ""


_CATASTROPHIC = [
    r"\b(?:\S*/)?mkfs(?:\.[\w-]+)?(?:\s|$)",
    r"\b(?:\S*/)?fdisk(?:\s|$)",
    r"\bdd\s+.*of=/dev/",
    r">{1,2}\s*/dev/",
    r"\b(shutdown|reboot|poweroff)\b",
    r"git\s+push\s+--force",
    r"git\s+reset\s+--hard",
    r"\bgit\s+clean\b(?=[^\n]*(?:\s--force(?:\s|=|$)|\s-[a-zA-Z]*f))",
    r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*;?\s*\}\s*;\s*:",
]

_SHELL_SYNTAX = re.compile(
    r"\||&&|\|\||;|`|\$\(|\$\{|<\(|>\(|\b(?:sh|bash|zsh|fish)\s+-[a-z]*c\b",
    re.IGNORECASE,
)


def _rm_is_catastrophic(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if Path(token).name != "rm":
            continue
        operands = [
            item
            for item in tokens[index + 1 :]
            if item == "--" or not item.startswith("-")
        ]
        if operands and operands[0] == "--":
            operands = operands[1:]
        destructive_flags = any(
            item in {"--recursive", "--force"}
            or (item.startswith("-") and ("r" in item or "f" in item))
            for item in tokens[index + 1 :]
        )
        if any(
            item == "/"
            or item == "~"
            or item in {"$HOME", "${HOME}"}
            or item.startswith(("/*", "/home", "/root", "~/", "$HOME/", "${HOME}/"))
            or (destructive_flags and ("$" in item or ".." in item))
            for item in operands
        ):
            return True
    return False


def _git_push_is_catastrophic(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if Path(token).name != "git" or tokens[index + 1 : index + 2] != ["push"]:
            continue
        return any(
            item == "-f" or item.startswith("--force") for item in tokens[index + 2 :]
        )
    return False


def _nested_shell_is_catastrophic(tokens: list[str]) -> bool:
    shells = {"sh", "bash", "zsh", "dash", "fish"}
    for index, token in enumerate(tokens):
        if Path(token).name not in shells:
            continue
        command_index = None
        for option_index in range(index + 1, len(tokens)):
            option = tokens[option_index]
            if option == "--":
                continue
            if option.startswith("-") and "c" in option[1:]:
                command_index = option_index + 1
                if command_index < len(tokens) and tokens[command_index] == "--":
                    command_index += 1
                break
            if not option.startswith("-"):
                break
        if (
            command_index is not None
            and command_index < len(tokens)
            and classify_command(tokens[command_index]).catastrophic
        ):
            return True
    return False


def classify_command(command: str) -> CommandClassification:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    if (
        _rm_is_catastrophic(tokens)
        or _git_push_is_catastrophic(tokens)
        or _nested_shell_is_catastrophic(tokens)
    ):
        return CommandClassification(catastrophic=True, reason="catastrophic command")
    for pattern in _CATASTROPHIC:
        if re.search(pattern, command, re.IGNORECASE):
            return CommandClassification(
                catastrophic=True, reason="catastrophic command"
            )
    outside = bool(
        re.search(
            r"(^|\s)(?:/|\.\./)|\b(?:cd|git\s+-C)\b|[<>]\s*[/~]|\$\(|`|\b(?:python|python3|node|ruby|perl)\s+-c\b",
            command,
        )
        or _SHELL_SYNTAX.search(command)
        or not tokens
    )
    return CommandClassification(
        outside_or_unknown=outside,
        reason="outside or unknown command" if outside else "safe command",
    )
