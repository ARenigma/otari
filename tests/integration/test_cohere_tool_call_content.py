"""A Cohere chat history whose assistant tool-call turn has no ``content`` key.

The provider SDK is mocked below any-llm, so the request runs through any-llm's
own Cohere message conversion, which is where the missing key used to fail.
"""

from collections.abc import AsyncIterator, Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from cohere import AssistantMessageResponse, TextAssistantMessageResponseContentItem, Usage, UsageTokens, V2ChatResponse
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig

from .conftest import build_test_client

_MASTER_KEY_HEADER = {API_KEY_HEADER: "Bearer test-master-key"}

_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "weather", "arguments": '{"city":"Rome"}'},
}

_BODY: dict[str, Any] = {
    "model": "cohere:command-r7b-12-2024",
    "messages": [
        {"role": "user", "content": "Weather in Rome?"},
        {"role": "assistant", "tool_calls": [_TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny 32C"},
    ],
    "user": "agent",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ],
}


@pytest.fixture
def cohere_client(postgres_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        providers={"cohere": {"api_key": "test-cohere-key"}},
    )
    for client in build_test_client(config):
        created = client.post(f"{API_ROOT}/users", json={"user_id": "agent"}, headers=_MASTER_KEY_HEADER)
        assert created.status_code == 200, created.text
        yield client


def _assistant_turn(sent: list[dict[str, Any]]) -> dict[str, Any]:
    return next(message for message in sent if message.get("role") == "assistant")


def test_non_streamed_tool_call_turn_without_content_reaches_cohere(cohere_client: TestClient) -> None:
    sent: list[dict[str, Any]] = []

    async def chat(self: Any, *, messages: list[dict[str, Any]], **kwargs: Any) -> V2ChatResponse:
        sent.extend(messages)
        return V2ChatResponse(
            id="r1",
            finish_reason="COMPLETE",
            message=AssistantMessageResponse(
                role="assistant",
                content=[TextAssistantMessageResponseContentItem(type="text", text="Sunny in Rome.")],
            ),
            usage=Usage(tokens=UsageTokens(input_tokens=10, output_tokens=4)),
        )

    with patch("cohere.AsyncClientV2.chat", new=chat):
        response = cohere_client.post(f"{API_ROOT}/chat/completions", json=_BODY, headers=_MASTER_KEY_HEADER)

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "Sunny in Rome."
    assistant = _assistant_turn(sent)
    assert assistant["tool_plan"] is None
    assert assistant["tool_calls"] == [_TOOL_CALL]


def test_streamed_tool_call_turn_without_content_reaches_cohere(cohere_client: TestClient) -> None:
    sent: list[dict[str, Any]] = []

    def chat_stream(self: Any, *, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[Any]:
        sent.extend(messages)

        async def events() -> AsyncIterator[Any]:
            text = SimpleNamespace(text="Sunny in Rome.")
            yield SimpleNamespace(type="content-delta", delta=SimpleNamespace(message=SimpleNamespace(content=text)))
            tokens = SimpleNamespace(input_tokens=10, output_tokens=4)
            yield SimpleNamespace(
                type="message-end",
                delta=SimpleNamespace(finish_reason="COMPLETE", usage=SimpleNamespace(tokens=tokens)),
            )

        return events()

    with patch("cohere.AsyncClientV2.chat_stream", new=chat_stream):
        response = cohere_client.post(
            f"{API_ROOT}/chat/completions", json={**_BODY, "stream": True}, headers=_MASTER_KEY_HEADER
        )

    assert response.status_code == 200, response.text
    assert "Sunny in Rome." in response.text
    assert response.text.rstrip().endswith("data: [DONE]")
    assert _assistant_turn(sent)["tool_plan"] is None
