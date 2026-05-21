from __future__ import annotations

from collections.abc import AsyncIterator

import litellm

from sr2.models import TextBlock, TokenUsage
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


class RelayLLMCallable:
    def __init__(self, model: str, base_url: str | None = None, **kwargs) -> None:
        # When hitting an OpenAI-compatible endpoint with a bare model name,
        # litellm needs a provider prefix to route the call correctly.
        if base_url is not None and "/" not in model:
            model = f"openai/{model}"
        self._model = model
        self._kwargs: dict = kwargs
        if base_url is not None:
            self._kwargs["base_url"] = base_url

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        return [
            {
                "role": msg.role,
                "content": "".join(
                    b.text for b in msg.content if hasattr(b, "text")
                ),
            }
            for msg in request.messages
        ]

    def _build_extra(self, request: CompletionRequest) -> dict:
        extra: dict = {}
        if request.system is not None:
            extra["system"] = "".join(b.text for b in request.system)
        return extra

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        messages = self._build_messages(request)
        extra = self._build_extra(request)

        resp = await litellm.acompletion(
            model=self._model,
            messages=messages,
            **self._kwargs,
            **extra,
        )

        choice = resp.choices[0]
        content_text = choice.message.content or ""
        return CompletionResponse(
            id=resp.id,
            content=[TextBlock(text=content_text)],
            stop_reason=choice.finish_reason,
            usage=TokenUsage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            ),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        messages = self._build_messages(request)
        extra = self._build_extra(request)

        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            stream=True,
            **self._kwargs,
            **extra,
        )

        async for chunk in response:
            # Text content
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    yield StreamEvent(type="text", text=delta_content)

            # Usage
            if getattr(chunk, "usage", None) is not None:
                yield StreamEvent(
                    type="usage",
                    usage=TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    ),
                )

        yield StreamEvent(type="end")
