from pathlib import Path

import pytest

from coding_agent.policy.approval import DefaultApprovalPolicy
from coding_agent.policy.command import classify_command
from coding_agent.tools.models import ToolSchema

READ = ToolSchema(
    name="read_file", description="read", parameters={}, risk_level="read"
)
WRITE = ToolSchema(
    name="write_file", description="write", parameters={}, risk_level="mutate_file"
)
SHELL = ToolSchema(
    name="run_command", description="shell", parameters={}, risk_level="mutate_shell"
)


def test_default_allows_read_and_asks_for_mutations():
    policy = DefaultApprovalPolicy()
    assert policy.decide(READ, {}, workspace=Path("."), mode="default").kind == "allow"
    assert (
        policy.decide(WRITE, {"path": "a"}, workspace=Path("."), mode="default").kind
        == "ask"
    )
    assert (
        policy.decide(
            READ, {"path": "/tmp/out"}, workspace=Path("."), mode="default"
        ).kind
        == "ask"
    )


def test_workspace_allows_internal_write_but_asks_for_outside_path(tmp_path):
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(
            WRITE, {"path": "src/index.html"}, workspace=tmp_path, mode="workspace"
        ).kind
        == "allow"
    )
    decision = policy.decide(
        WRITE, {"path": "../secret.txt"}, workspace=tmp_path, mode="workspace"
    )
    assert decision.kind == "ask"
    assert decision.allow_outside_once is True


def test_full_allows_ordinary_tools_but_catastrophic_shell_is_denied():
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(WRITE, {"path": "/tmp/x"}, workspace=Path("."), mode="full").kind
        == "allow"
    )
    decision = policy.decide(
        SHELL, {"command": "rm -rf /"}, workspace=Path("."), mode="full"
    )
    assert decision.kind == "deny"


def test_unknown_workspace_shell_requires_approval():
    decision = classify_command('python -c \'open("/tmp/out", "w").write("x")\'')
    assert decision.outside_or_unknown is True


def test_catastrophic_command_patterns_are_always_denied():
    policy = DefaultApprovalPolicy()
    for command in (
        "mkfs.ext4 /dev/sda",
        "git reset --hard HEAD",
        "shutdown now",
        "rm -- -rf /",
        "rm --recursive --force /",
    ):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode="full"
            ).kind
            == "deny"
        )


@pytest.mark.parametrize("command", ["rm --no-preserve-root -rf /", "rm -i -rf /", "rm -rf -- /"])
def test_root_removal_option_variants_are_always_denied(command):
    assert classify_command(command).catastrophic is True
