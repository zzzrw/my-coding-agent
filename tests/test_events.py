from coding_agent.runtime.events import RuntimeEvent


def test_runtime_event_envelope():
    event = RuntimeEvent(
        type="assistant_delta", payload={"message_id": "m1", "text": "hi"}
    )
    assert event.event_id and event.payload["text"] == "hi"
