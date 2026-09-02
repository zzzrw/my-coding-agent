from pathlib import Path

from coding_agent.app import build_system_prompt


def _content(workspace: Path, mode: str) -> str:
    return build_system_prompt(workspace, mode).content  # type: ignore[arg-type]


def test_system_prompt_lists_hard_denied_commands(tmp_path):
    content = _content(tmp_path, "default")
    for fragment in (
        "/root",
        "/home/<user>",
        "git push --force",
        "git reset --hard",
        "git clean -f",
        "mkfs",
        "fdisk",
        "shutdown",
        "reboot",
        "poweroff",
    ):
        assert fragment in content


def test_system_prompt_recommends_the_sanctioned_delete_tools(tmp_path):
    content = _content(tmp_path, "default")
    assert "remove_file" in content
    assert "clear_directory" in content


def test_system_prompt_discourages_dev_null_redirection(tmp_path):
    content = _content(tmp_path, "default")
    assert "> /dev/null" in content
    assert "discouraged" in content
    assert "(blocked)" not in content
    assert "2>&1" in content


def test_system_prompt_states_the_active_permission_mode(tmp_path):
    content = _content(tmp_path, "workspace")
    assert '"workspace"' in content


def test_system_prompt_names_the_workspace_root(tmp_path):
    content = _content(tmp_path, "full")
    assert str(tmp_path) in content


def test_system_prompt_guides_dev_server_management(tmp_path):
    content = _content(tmp_path, "default")
    assert "pgrep" in content
    assert "pkill" in content
