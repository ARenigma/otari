"""The Cohere tool-call ``content`` stopgap (mozilla-ai/any-llm#1467)."""

import copy
from typing import Any
from unittest.mock import patch

import pytest

from gateway.services.inference import with_cohere_tool_call_content
from gateway.services.mcp_loop import _ChatToolLoopStrategy

_CALL = {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}


def _kwargs(model: str = "cohere:command-r7b-12-2024", **extra: Any) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "Weather in Rome?"},
            {"role": "assistant", "tool_calls": [_CALL]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny 32C"},
        ],
        **extra,
    }


def test_fills_content_on_a_cohere_tool_call_turn_without_it() -> None:
    kwargs = _kwargs()
    original = copy.deepcopy(kwargs)

    patched = with_cohere_tool_call_content(kwargs)

    assert patched["messages"][1] == {"role": "assistant", "tool_calls": [_CALL], "content": None}
    assert patched["messages"][0] is kwargs["messages"][0]
    assert patched["messages"][2] is kwargs["messages"][2]
    assert kwargs == original, "the caller's kwargs and messages are left as they were"


def test_reads_the_provider_keyword_when_given() -> None:
    patched = with_cohere_tool_call_content(_kwargs(model="command-r7b-12-2024", provider="cohere"))

    assert patched["messages"][1]["content"] is None


@pytest.mark.parametrize("model", ["openai:gpt-4o-mini", "mistral:mistral-small-latest", "not-a-selector"])
def test_leaves_other_providers_and_unparseable_selectors_alone(model: str) -> None:
    kwargs = _kwargs(model=model)

    assert with_cohere_tool_call_content(kwargs) is kwargs


@pytest.mark.parametrize(
    "assistant",
    [
        {"role": "assistant", "content": "Let me check.", "tool_calls": [_CALL]},
        {"role": "assistant", "content": None, "tool_calls": [_CALL]},
        {"role": "assistant", "content": "No tools needed."},
        {"role": "assistant", "tool_calls": []},
    ],
)
def test_leaves_turns_that_already_work_alone(assistant: dict[str, Any]) -> None:
    kwargs = {"model": "cohere:command-r7b-12-2024", "messages": [{"role": "user", "content": "hi"}, assistant]}

    assert with_cohere_tool_call_content(kwargs) is kwargs


@pytest.mark.asyncio
async def test_the_chat_tool_loop_sends_its_own_tool_call_turns_with_content() -> None:
    # The loop appends ``{"role": "assistant", "tool_calls": ...}`` between rounds.
    seen: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return object()

    with patch("gateway.services.mcp_loop.acompletion", new=fake_acompletion):
        await _ChatToolLoopStrategy().call(_kwargs())

    assert seen["messages"][1]["content"] is None
