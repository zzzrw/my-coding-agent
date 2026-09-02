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
    r"\bdd\s+.*\bof=/dev/(?!null(?:\s|$))",
    r">{1,2}\s*/dev/(?!null(?:\s|$))",
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


_RM_PROTECTED_ROOT_OPERANDS = {"/", "~", "$HOME", "${HOME}", "/home", "/root"}
_RM_HOME_CONTENT_GLOBS = {"~/*", "$HOME/*", "${HOME}/*"}


def _rm_operand_wipes_protected_root(operand: str) -> bool:
    """True when an ``rm`` operand can wipe a protected root or a whole home.

    A single trailing slash is equivalent to none (``rm -rf ~/`` wipes the
    home directory, ``/home/me/`` wipes that user's home). The doubled-slash
    root form ``//`` is left untouched, preserving prior classification.
    """
    candidates = [operand]
    if len(operand) > 1 and operand.endswith("/") and not operand.startswith("//"):
        candidates.append(operand[:-1])
    for candidate in candidates:
        if candidate in _RM_PROTECTED_ROOT_OPERANDS:
            return True
        if candidate in _RM_HOME_CONTENT_GLOBS:
            return True
        if candidate.startswith("/root/") or re.fullmatch(
            r"/home/[^/]+(?:/|/\*)?", candidate
        ):
            return True
    return False


def _rm_references_unresolvable_variable(operand: str) -> bool:
    """True when ``operand`` references a ``$VAR`` other than ``$HOME``."""
    if "$" not in operand:
        return False
    remainder = operand.replace("$HOME", "").replace("${HOME}", "")
    return "$" in remainder


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
            _rm_operand_wipes_protected_root(item)
            or (destructive_flags and ".." in item)
            or (destructive_flags and _rm_references_unresolvable_variable(item))
            for item in operands
        ):
            return True
    return False


_GIT_GLOBAL_OPTIONS_WITH_ARGUMENTS = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--work-tree",
}
_GIT_GLOBAL_FLAGS = {
    "-p",
    "-P",
    "--bare",
    "--no-advice",
    "--no-lazy-fetch",
    "--no-pager",
    "--no-replace-objects",
    "--no-optional-locks",
    "--paginate",
    "--literal-pathspecs",
    "--glob-pathspecs",
    "--noglob-pathspecs",
    "--icase-pathspecs",
}


def _git_push_subcommand_index(tokens: list[str], git_index: int) -> int | None:
    index = git_index + 1
    while index < len(tokens):
        token = tokens[index]
        option, separator, _value = token.partition("=")
        if option in _GIT_GLOBAL_OPTIONS_WITH_ARGUMENTS:
            if not separator:
                index += 1
            index += 1
            continue
        if token in _GIT_GLOBAL_FLAGS:
            index += 1
            continue
        if token.startswith("-C") and token != "-C":
            index += 1
            continue
        if token.startswith("-c") and token != "-c" and "=" in token[2:]:
            index += 1
            continue
        if token.startswith("--exec-path="):
            index += 1
            continue
        return index
    return None


def _git_push_arguments_are_catastrophic(arguments: list[str]) -> bool:
    value_options = {
        "--exec",
        "--push-option",
        "--receive-pack",
        "--recurse-submodules",
        "--repo",
    }
    destructive_options = {"--delete", "-d", "--mirror", "--prune"}
    skip_next = False
    for item in arguments:
        if skip_next:
            skip_next = False
            continue
        option, separator, _value = item.partition("=")
        if option in value_options:
            skip_next = not separator
            continue
        if option in destructive_options:
            return True
        if item == "-o":
            skip_next = True
            continue
        if item.startswith("-") and not item.startswith("--"):
            short_options = item[1:]
            option_value = short_options.find("o")
            flags = (
                short_options if option_value == -1 else short_options[:option_value]
            )
            if "f" in flags or "d" in flags:
                return True
            skip_next = option_value == len(short_options) - 1
            continue
        if item.startswith("--force"):
            return True
        if item.startswith(("+", ":")) and len(item) > 1:
            return True
    return False


def _matching_parenthesis(command: str, open_index: int) -> int:
    """Return the index of the ``)`` matching the ``(`` at ``open_index``."""
    depth = 0
    single_quoted = False
    double_quoted = False
    escaped = False
    index = open_index
    while index < len(command):
        character = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and not single_quoted:
            escaped = True
            index += 1
            continue
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            index += 1
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            index += 1
            continue
        if not single_quoted and not double_quoted and character == "(":
            depth += 1
        elif not single_quoted and not double_quoted and character == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _command_substitutions_are_catastrophic(command: str) -> bool:
    """Inspect executable command/process substitutions respecting shell quoting."""
    single_quoted = False
    double_quoted = False
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and not single_quoted:
            escaped = True
            index += 1
            continue
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            index += 1
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            index += 1
            continue
        if not single_quoted and command.startswith("$(", index):
            end = _matching_parenthesis(command, index + 1)
            if end != -1 and classify_command(command[index + 2 : end]).catastrophic:
                return True
            index += 2
            continue
        if (
            not single_quoted
            and not double_quoted
            and (command.startswith("<(", index) or command.startswith(">(", index))
        ):
            end = _matching_parenthesis(command, index + 1)
            if end != -1 and classify_command(command[index + 2 : end]).catastrophic:
                return True
            index += 2
            continue
        if not single_quoted and character == "`":
            end = index + 1
            while end < len(command):
                if command[end] == "`" and command[end - 1] != "\\":
                    break
                end += 1
            if (
                end < len(command)
                and classify_command(command[index + 1 : end]).catastrophic
            ):
                return True
            index = end + 1
            continue
        index += 1
    return False


def _git_push_is_catastrophic(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if Path(token).name != "git":
            continue
        push_index = _git_push_subcommand_index(tokens, index)
        if push_index is None or tokens[push_index] != "push":
            continue
        return _git_push_arguments_are_catastrophic(tokens[push_index + 1 :])
    return False


_GIT_REMOTE_DESTRUCTIVE_ACTIONS = {"remove", "rm", "delete", "prune"}


def _git_remote_is_catastrophic(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if Path(token).name != "git":
            continue
        subcommand_index = _git_push_subcommand_index(tokens, index)
        if subcommand_index is None or tokens[subcommand_index] != "remote":
            continue
        action_index = subcommand_index + 1
        if (
            action_index < len(tokens)
            and tokens[action_index] in _GIT_REMOTE_DESTRUCTIVE_ACTIONS
        ):
            return True
    return False


def _git_config_alias_value_is_shell(value: str) -> str | None:
    """Return the shell body when ``value`` defines a ``!``-prefixed git alias."""
    key, separator, body = value.partition("=")
    if not separator or not key.startswith("alias."):
        return None
    return body[1:] if body.startswith("!") else None


def _git_config_shell_alias_is_catastrophic(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if Path(token).name != "git":
            continue
        option_index = index + 1
        while option_index < len(tokens):
            option, separator, value = tokens[option_index].partition("=")
            if option == "--config-env":
                # The value is resolved from an environment variable we cannot
                # inspect, so it could inject a shell alias (alias.<name>=!...)
                # that bypasses catastrophic policy. Conservatively reject.
                return True
            if option == "-c":
                if not separator:
                    option_index += 1
                    if option_index >= len(tokens):
                        break
                    value = tokens[option_index]
                body = _git_config_alias_value_is_shell(value)
                if body is not None and classify_command(body).catastrophic:
                    return True
                option_index += 1
                continue
            if option.startswith("-"):
                option_index += 1
                continue
            break
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
        or _git_remote_is_catastrophic(tokens)
        or _git_config_shell_alias_is_catastrophic(tokens)
        or _nested_shell_is_catastrophic(tokens)
        or _command_substitutions_are_catastrophic(command)
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
