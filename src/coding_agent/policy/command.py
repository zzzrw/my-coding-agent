import re

from pydantic import BaseModel


class CommandClassification(BaseModel):
    catastrophic: bool = False
    outside_or_unknown: bool = False
    reason: str = ""


_CATASTROPHIC = [
    r"rm\s+(?:(?:-[rf]+|--(?:recursive|force))\s+|--\s+)*[/~]",
    r"\bmkfs(?:\.|\s)",
    r"\bdd\s+.*of=/dev/",
    r"\b(shutdown|reboot|poweroff)\b",
    r"git\s+push\s+--force",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[a-z]*f",
    r":\s*\(\s*\)\s*\{.*\|.*&\s*\}\s*;\s*:",
]


def classify_command(command: str) -> CommandClassification:
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
