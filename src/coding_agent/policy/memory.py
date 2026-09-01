"""Scoped approval decision memory.

`DecisionMemory` maps normalized ``(tool_name, arguments)`` signatures to
remembered allow/deny decisions. Decisions are scoped one of four ways:

- ``once``: the next approval for this signature only.
- ``turn``: cleared when the current turn ends (``clear_turn``).
- ``session``: cleared when the session ends (``clear_session``).
- ``always``: persisted to ``approvals.json`` and reloaded on startup.

The module is pure and synchronous aside from the ``always`` filesystem
read/write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from coding_agent.config.config import config_dir

Decision = Literal["allow", "deny"]
Scope = Literal["once", "turn", "session", "always"]

Signature = tuple[str, str]


def signature(tool_name: str, arguments: dict) -> Signature:
    """Normalize a tool call into a stable ``(name, arguments_json)`` key."""
    return tool_name, json.dumps(arguments, sort_keys=True)


class DecisionMemory:
    def __init__(
        self,
        config_dir: Path | None = None,
        always_path: Path | None = None,
    ) -> None:
        self._config_dir = config_dir
        self._explicit_always_path = always_path
        self._once: dict[Signature, Decision] = {}
        self._turn: dict[Signature, Decision] = {}
        self._session: dict[Signature, Decision] = {}
        self._always: dict[Signature, Decision] = {}

    @property
    def always_path(self) -> Path:
        """Where ``always`` decisions are persisted by default."""
        if self._explicit_always_path is not None:
            return self._explicit_always_path
        return (self._config_dir or config_dir()) / "approvals.json"

    def remember(self, sig: Signature, decision: Decision, scope: Scope) -> None:
        """Remember ``decision`` for ``sig`` under ``scope``."""
        target = {
            "once": self._once,
            "turn": self._turn,
            "session": self._session,
            "always": self._always,
        }[scope]
        target[sig] = decision

    def lookup(self, sig: Signature) -> Decision | None:
        """Return the strongest remembered decision for ``sig``, or ``None``."""
        for table in (self._always, self._session, self._turn, self._once):
            if sig in table:
                return table[sig]
        return None

    def clear_turn(self) -> None:
        """Forget turn- and once-scoped decisions."""
        self._turn.clear()
        self._once.clear()

    def clear_session(self) -> None:
        """Forget session-, turn-, and once-scoped decisions."""
        self._session.clear()
        self._turn.clear()
        self._once.clear()

    def load_always(self, path: Path | None = None) -> None:
        """Load ``always`` decisions from ``path`` (default ``always_path``)."""
        target = path or self.always_path
        self._always = {}
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        for key, decision in raw.items():
            name, args_json = json.loads(key)
            self._always[(name, args_json)] = decision

    def persist_always(self, path: Path | None = None) -> None:
        """Write ``always`` decisions to ``path`` (default ``always_path``).

        The file is created with mode ``0600`` so remembered approvals do not
        leak secrets to other local users.
        """
        target = path or self.always_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            json.dumps([name, args_json], sort_keys=True): decision
            for (name, args_json), decision in self._always.items()
        }
        fd = os.open(target, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.chmod(target, 0o600)
