import re
import shlex

from pydantic import BaseModel


class CommandClassification(BaseModel):
    catastrophic: bool = False
    outside_or_unknown: bool = False
    reason: str = ""


_CATASTROPHIC = [
    r"\bmkfs(?:\.|\s)",
    r"\bdd\s+.*of=/dev/",
    r"\b(shutdown|reboot|poweroff)\b",
    r"git\s+push\s+--force",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[a-z]*f",
    r":\s*\(\s*\)\s*\{.*\|.*&\s*\}\s*;\s*:",
]


def classify_command(command: str) -> CommandClassification:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    for index, token in enumerate(tokens):
        if token != "rm":
            continue
        operands = [item for item in tokens[index + 1 :] if item == "--" or not item.startswith("-")]
        if operands and operands[0] == "--":
            operands = operands[1:]
        if any(item in {"/", "~", "/*"} for item in operands):
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
    )
    return CommandClassification(
        outside_or_unknown=outside,
        reason="outside or unknown command" if outside else "safe command",
    )
