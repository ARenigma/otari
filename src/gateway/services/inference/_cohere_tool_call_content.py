"""Give Cohere an assistant tool-call turn's ``content`` key when the caller left it out.

Stopgap for an any-llm gap: its Cohere provider pops ``content`` from an
assistant message that carries ``tool_calls`` and raises ``KeyError`` when the
key is absent, although the OpenAI format allows leaving it out. Remove this
module and its call sites once mozilla-ai/any-llm#1467 is released and the
``any-llm-sdk`` pin includes it.
"""

from __future__ import annotations

from typing import Any

from any_llm import AnyLLM, LLMProvider


def _lacks_content(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and bool(message.get("tool_calls"))
        and "content" not in message
    )


def with_cohere_tool_call_content(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return chat-completion ``kwargs`` with ``content: None`` on Cohere's content-less tool-call turns.

    Anything else, including every other provider, comes back as the same
    object. A change is made on copies, so the caller's messages are untouched.
    """
    messages = kwargs.get("messages")
    if not isinstance(messages, list) or not any(_lacks_content(message) for message in messages):
        return kwargs
    try:
        provider = (
            LLMProvider(kwargs["provider"])
            if kwargs.get("provider") is not None
            else AnyLLM.split_model_provider(str(kwargs.get("model", "")))[0]
        )
    except ValueError:
        return kwargs
    if provider is not LLMProvider.COHERE:
        return kwargs
    patched = [{**message, "content": None} if _lacks_content(message) else message for message in messages]
    return {**kwargs, "messages": patched}
