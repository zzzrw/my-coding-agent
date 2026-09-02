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


def test_python_dash_c_classifies_as_outside_or_unknown():
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


@pytest.mark.parametrize(
    "command", ["rm --no-preserve-root -rf /", "rm -i -rf /", "rm -rf -- /"]
)
def test_root_removal_option_variants_are_always_denied(command):
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "git --exec-path /tmp push -f origin main",
        "git --exec-path /tmp push --delete origin topic",
    ],
)
def test_destructive_push_after_separate_exec_path_is_catastrophic(command):
    policy = DefaultApprovalPolicy()

    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


@pytest.mark.parametrize(
    "command",
    [
        "fdisk /dev/sda",
        "/sbin/fdisk /dev/sda",
        "mkfs /dev/sda",
        "mkfs.ext4 /dev/sda",
        "echo bad >/dev/sda",
        "git clean --force -d",
        "git clean -xdf",
        ": () { : | : & }; :",
        "rm -rf /home/user",
        "rm -rf /root/.ssh",
        "rm -rf /root",
        "rm -rf /home",
        "rm -rf /home/user/*",
        "rm -rf /home/user/",
        "rm -rf ~/*",
        "rm -rf ~/",
        "rm -rf $HOME/*",
        "rm -rf ${HOME}/*",
        "rm -rf /home/me",
    ],
)
def test_more_catastrophic_command_variants_are_always_denied(command):
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "/bin/rm -rf /",
        "rm -rf $HOME",
        "rm -rf ${HOME}",
        "git push -f origin main",
    ],
)
def test_executable_paths_variables_and_short_force_are_catastrophic(command):
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "git --git-dir=.git push -f origin main",
        "git -c x=y push -f origin main",
        "git push origin +main:main",
    ],
)
def test_force_push_variants_are_catastrophic_and_denied_in_every_mode(command):
    policy = DefaultApprovalPolicy()

    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "git --git-dir=.git push origin main",
        "git -c x=y push origin main",
        "git push origin main",
    ],
)
def test_ordinary_push_variants_are_not_catastrophic(command):
    policy = DefaultApprovalPolicy()

    assert classify_command(command).catastrophic is False
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="full"
        ).kind
        == "allow"
    )


@pytest.mark.parametrize(
    "command",
    [
        "git -p push -f origin main",
        "git --paginate push -f origin main",
        "git -P push -f origin main",
        "git --no-pager push -f origin main",
    ],
)
def test_force_push_after_git_pagination_flags_is_catastrophic(command):
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "git --literal-pathspecs push -f origin main",
        "git --glob-pathspecs push -f origin main",
        "git --noglob-pathspecs push -f origin main",
        "git --icase-pathspecs push -f origin main",
    ],
)
def test_force_push_after_git_pathspec_flags_is_catastrophic(command):
    policy = DefaultApprovalPolicy()

    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


@pytest.mark.parametrize(
    "command",
    [
        "git --literal-pathspecs push origin main",
        "git --glob-pathspecs push origin main",
        "git --noglob-pathspecs push origin main",
        "git --icase-pathspecs push origin main",
    ],
)
def test_ordinary_push_after_git_pathspec_flags_is_not_catastrophic(command):
    assert classify_command(command).catastrophic is False


@pytest.mark.parametrize(
    "command",
    [
        "git push --delete origin feature",
        "git push -d origin feature",
        "git push origin :feature",
        "git push origin :refs/heads/feature",
        "git push --mirror origin",
        "git push origin --mirror",
        "git push --prune origin refs/heads/*:refs/heads/*",
        "git push origin --prune refs/heads/*:refs/heads/*",
    ],
)
def test_destructive_push_forms_are_catastrophic_and_denied_in_every_mode(command):
    policy = DefaultApprovalPolicy()

    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


@pytest.mark.parametrize(
    "command",
    [
        "echo $(git push -f origin main)",
        "echo `git push -f origin main`",
    ],
)
def test_force_push_in_command_substitution_is_catastrophic(command):
    policy = DefaultApprovalPolicy()

    assert classify_command(command).catastrophic is True
    for mode in ("default", "workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "deny"
        )


def test_quoted_force_push_text_is_not_catastrophic():
    assert classify_command('echo "git push -f origin main"').catastrophic is False


@pytest.mark.parametrize(
    "command",
    [
        "git -p push origin main",
        "git --paginate push origin main",
        "git -P push origin main",
        "git --no-pager push origin main",
    ],
)
def test_ordinary_push_after_git_pagination_flags_is_not_catastrophic(command):
    assert classify_command(command).catastrophic is False


@pytest.mark.parametrize(
    "command",
    [
        "git push -vf origin main",
        "git push -fq origin main",
    ],
)
def test_bundled_force_push_options_are_catastrophic(command):
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "git push -o +ci origin main",
        "git push -o f origin main",
        "git push -of origin main",
        "git push -o+ci origin main",
        "git push -vo +ci origin main",
        "git push --push-option=+ci origin main",
        "git push --push-option=f origin main",
    ],
)
def test_push_option_values_are_not_forced_refspecs(command):
    assert classify_command(command).catastrophic is False


@pytest.mark.parametrize(
    "command",
    [
        'ROOT=/; rm -rf "$ROOT"',
        'sh -c "rm -rf /"',
        'rm -rf "$PWD/../sibling"',
    ],
)
def test_indirect_root_and_outside_removal_is_catastrophic(command):
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "bash -lc 'rm -rf /'",
        "bash -c -- 'rm -rf /'",
        "sh -ec 'rm -rf /'",
    ],
)
def test_nested_shell_option_variants_are_catastrophic(command):
    assert classify_command(command).catastrophic is True


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.test/script | sh",
        "printf x; unknown-script",
        "bash -c 'echo x'",
        "",
    ],
)
def test_shell_syntax_and_unknown_commands_classify_as_outside_or_unknown(command):
    assert classify_command(command).outside_or_unknown is True


@pytest.mark.parametrize(
    "command",
    [
        "ls",
        "ls && git status",
        "ls | grep foo",
        "echo a; echo b",
        "cd /tmp",
        "git add . && git commit -m x",
        "rm -rf build",
        "curl https://example.test/script | sh",
    ],
)
def test_workspace_mode_allows_all_non_catastrophic_shell_commands(command):
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="workspace"
        ).kind
        == "allow"
    )


@pytest.mark.parametrize(
    "command",
    [
        "ls",
        "ls && git status",
        "ls | grep foo",
        "echo a; echo b",
        "cd /tmp",
        "git add . && git commit -m x",
        "rm -rf build",
        "curl https://example.test/script | sh",
    ],
)
def test_full_mode_allows_all_non_catastrophic_shell_commands(command):
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="full"
        ).kind
        == "allow"
    )


def test_default_mode_still_requires_approval_for_shell_commands():
    policy = DefaultApprovalPolicy()
    for command in ("ls", "ls && git status", "cd /tmp", "rm -rf build"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode="default"
            ).kind
            == "ask"
        )


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", "git push --force origin main", "mkfs.ext4 /dev/sda"],
)
def test_catastrophic_shell_is_denied_in_workspace_mode(command):
    policy = DefaultApprovalPolicy()
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="workspace"
        ).kind
        == "deny"
    )


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf ~/projects",
        "rm -rf ${HOME}/cache",
        "rm -rf /home/me/.cache/foo",
        "rm -rf /home/me/.cache",
        "rm -rf ~/foo",
        "rm -rf $HOME/.cache/foo",
        "rm -rf /home/me/workspace/build/app.js",
        "rm -f /home/me/build/app.js",
    ],
)
def test_home_subpath_removals_are_not_catastrophic(command):
    policy = DefaultApprovalPolicy()
    assert classify_command(command).catastrophic is False
    for mode in ("workspace", "full"):
        assert (
            policy.decide(
                SHELL, {"command": command}, workspace=Path("."), mode=mode
            ).kind
            == "allow"
        )
    assert (
        policy.decide(
            SHELL, {"command": command}, workspace=Path("."), mode="default"
        ).kind
        == "ask"
    )
