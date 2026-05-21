"""FastAPI HTTP server for sr2-relay.

Exposes:
- GET  /health
- POST /v1/chat/completions  (non-streaming + streaming)
- DELETE /v1/sessions/{session_id}
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

import sr2_relay.translators  # noqa: F401 — triggers auto-registration of translators
from sr2_relay.translators.base import get_translator
from sr2_relay.models import SR2RelayConfig
from sr2_relay.session import RelaySession
from sr2.protocols.llm import StreamEvent


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(
    config: SR2RelayConfig,
    relay_session: RelaySession | None = None,
) -> FastAPI:
    if relay_session is None:
        relay_session = RelaySession(config)

    app = FastAPI()

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    # -----------------------------------------------------------------------
    # Chat completions
    # -----------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    ):
        # Translate to canonical request
        translator = get_translator("acompletion")
        if translator is None:
            raise HTTPException(status_code=400, detail="No translator for call_type")

        canonical = translator.to_canonical(body.model_dump(), call_type="acompletion")

        result = await relay_session.complete(
            canonical,
            session_id=x_session_id,
            stream=body.stream,
        )

        if body.stream:
            # result is an async iterator of StreamEvent
            async def _event_stream(gen: AsyncIterator[StreamEvent]):
                async for event in gen:
                    if event.type == "text":
                        chunk = {
                            "id": "relay-stream",
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": event.text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif event.type == "tool_use":
                        chunk = {
                            "id": "relay-stream",
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": event.tool_use_id,
                                                "type": "function",
                                                "function": {
                                                    "name": event.tool_name,
                                                    "arguments": json.dumps(event.tool_input),
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _event_stream(result),
                media_type="text/event-stream",
            )
        else:
            # result is a CompletionResponse
            from sr2.models import TextBlock as SR2TextBlock, ToolUseBlock as SR2ToolUseBlock

            content_text = "".join(
                block.text
                for block in result.content
                if isinstance(block, SR2TextBlock)
            )
            tool_calls = [
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
                for block in result.content
                if isinstance(block, SR2ToolUseBlock)
            ]
            message: dict = {"role": "assistant", "content": content_text or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            finish_reason = "tool_calls" if tool_calls else "stop"
            usage = result.usage
            return {
                "id": result.id,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                },
            }

    # -----------------------------------------------------------------------
    # Models list — Hermes and other clients probe this on startup
    # -----------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models() -> dict:
        model_id = config.model.model if config.model and config.model.model else "relay"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "sr2-relay",
                }
            ],
        }

    # -----------------------------------------------------------------------
    # Session deletion
    # -----------------------------------------------------------------------

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        try:
            relay_session.delete_session(session_id)
        except KeyError:
            pass
        return {"deleted": True, "session_id": session_id}

    return app
