from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from coding_agent.runtime.models import (
    LLMEvent,
    Message,
    RuntimeStatus,
    ToolCall,
    Usage,
)
from coding_agent.session.models import SessionRecord
from coding_agent.tools.models import ToolResult


def test_message_uses_internal_tool_call_shape():
    message = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
    )
    assert message.tool_calls[0].arguments == {"path": "a.py"}


def test_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        Usage(total_tokens=3, unexpected=1)


def test_session_record_requires_sequence_and_type():
    record = SessionRecord(
        id="r1",
        seq=0,
        timestamp=datetime.now(UTC),
        type="user_message",
        payload={"message": Message(role="user", content="hi")},
    )
    assert record.parent_id is None


def test_tool_result_has_structured_error_fields():
    result = ToolResult(
        tool_call_id="c1", tool_name="read_file", ok=False, content="", error="missing"
    )
    assert result.ok is False


def test_provider_event_and_runtime_status_have_boundary_fields():
    assert LLMEvent(type="text_delta", text="hi").text == "hi"
    assert RuntimeStatus(status="idle").status == "idle"
