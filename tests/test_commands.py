from coding_agent.tui.commands import (
    SUPPORTED_COMMANDS,
    Command,
    command_suggestions,
    parse_command,
)


def test_permission_command_parses_name_and_argument():
    assert parse_command("/permission workspace") == Command(
        name="permission", args=["workspace"]
    )


def test_resume_command_parses_session_prefix():
    assert parse_command("/resume ab") == Command(name="resume", args=["ab"])


def test_non_slash_input_is_a_prompt():
    assert parse_command("inspect the project") is None


def test_whitespace_and_quoted_arguments_are_preserved_as_tokens():
    assert parse_command('  /resume   "session with spaces"  ') == Command(
        name="resume", args=["session with spaces"]
    )


def test_unknown_command_is_returned_without_raising():
    command = parse_command("/unknown")

    assert command == Command(name="unknown", args=[])
    assert command.name not in SUPPORTED_COMMANDS


def test_supported_commands_match_the_mvp_registry():
    assert SUPPORTED_COMMANDS == {
        "help",
        "new",
        "session",
        "resume",
        "compact",
        "context",
        "permission",
        "clear",
        "quit",
        "exit",
    }


def test_exit_is_a_supported_alias_for_quit():
    assert parse_command("/exit") == Command(name="exit", args=[])


def test_command_suggestions_filter_by_typed_command_prefix():
    assert [(item.name, item.description) for item in command_suggestions("comp")] == [
        ("compact", "Compact the current context"),
    ]


def test_command_suggestions_show_all_commands_for_slash_prefix():
    suggestions = command_suggestions("")

    assert {item.name for item in suggestions} == SUPPORTED_COMMANDS
