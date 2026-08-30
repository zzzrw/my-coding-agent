import asyncio
import os

import pytest

from coding_agent.llm.openai_compatible import OpenAICompatibleProvider
from coding_agent.runtime.models import Message


@pytest.mark.live
@pytest.mark.asyncio
async def test_deepseek_live_smoke():
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1" or not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("live test disabled")
    provider = OpenAICompatibleProvider(
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    events = [
        event
        async for event in provider.stream(
            [Message(role="user", content="Reply with OK")],
            [],
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            signal=asyncio.Event(),
        )
    ]
    assert events and events[-1].type == "response_end"
