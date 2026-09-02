import asyncio
import json

import pytest

from coding_agent import app as app_module
from coding_agent.app import create_app
from coding_agent.runtime.models import LLMEvent


class SequencedFakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def stream(self, messages, tools, *, model, signal):
        self.requests.append((messages, tools, model))
        for event in self.responses.pop(0):
            if signal.is_set():
                return
            yield event


def tool_response(call_id, name, arguments):
    return [
        LLMEvent(type="tool_call_start", tool_call_id=call_id, tool_name=name),
        LLMEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=json.dumps(arguments),
        ),
        LLMEvent(type="tool_call_end", finish_reason="tool_calls"),
    ]


@pytest.mark.asyncio
async def test_factory_runs_write_verification_and_completion(tmp_path):
    provider = SequencedFakeProvider(
        [
            tool_response(
                "write-1",
                "write_file",
                {"path": "message.txt", "content": "hello\n"},
            ),
            tool_response(
                "run-1",
                "run_command",
                {"command": "test -f message.txt"},
            ),
            [
                LLMEvent(type="text_delta", text="verified"),
                LLMEvent(type="response_end", finish_reason="stop"),
            ],
        ]
    )
    application = create_app(
        workspace=tmp_path,
        model="fake-model",
        session_dir=tmp_path / "sessions",
        context_window=2_000,
        provider=provider,
        permission_mode="full",
    )

    completed = asyncio.Event()

    async def observe(event):
        if event.type == "run_finished":
            completed.set()

    application.runtime.subscribe(observe)
    await application.runtime.submit("create and verify message.txt")
    await asyncio.wait_for(completed.wait(), timeout=5)

    assert (tmp_path / "message.txt").read_text() == "hello\n"
    assert application.runtime.last_outcome is not None
    assert application.runtime.last_outcome.reason == "completed"
    assert len(provider.requests) == 3
    assert {tool.name for tool in provider.requests[0][1]} == {
        "read_file",
        "list_files",
        "grep_files",
        "write_file",
        "edit_file",
        "remove_file",
        "clear_directory",
        "load_skill",
        "run_command",
    }


def test_main_passes_base_url_to_create_app(monkeypatch, tmp_path):
    captured = {}

    class FakeApp:
        def run(self):
            pass

    def fake_create_app(**kwargs):
        captured.update(kwargs)
        return FakeApp()

    monkeypatch.setattr(app_module, "create_app", fake_create_app)

    assert (
        app_module.main(
            [
                "--workspace",
                str(tmp_path),
                "--model",
                "fake-model",
                "--base-url",
                "https://cli.example/v1",
            ]
        )
        == 0
    )
    assert captured["base_url"] == "https://cli.example/v1"


def test_main_redacts_secret_from_configuration_error(monkeypatch, capsys):
    secret = "super-secret-api-key"

    def fail_create_app(**kwargs):
        del kwargs
        raise app_module.ConfigurationError(f"invalid credential: {secret}")

    monkeypatch.setattr(app_module, "create_app", fail_create_app)

    assert app_module.main([]) == 2
    assert secret not in capsys.readouterr().err
