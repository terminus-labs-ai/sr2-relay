"""Tests for sr2_relay.server — FastAPI HTTP layer.

Covers:
- Health endpoint
- POST /v1/chat/completions (non-streaming + streaming)
- DELETE /v1/sessions/{session_id}
- Error cases (missing model/messages, no translator)
- create_app with relay_session=None
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from httpx import ASGITransport

from sr2.models import TextBlock as SR2TextBlock, TokenUsage
from sr2.protocols.llm import CompletionResponse, StreamEvent
from sr2_relay.models import SR2RelayConfig, ModelSlotConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> SR2RelayConfig:
    return SR2RelayConfig(
        api_base="http://localhost:11434",
        api_key="test-key",
        model=ModelSlotConfig(model="test-model", api_base="http://localhost:11434"),
    )


def _make_completion_response(text: str = "Hello from relay") -> CompletionResponse:
    return CompletionResponse(
        id="test-response-id",
        content=[SR2TextBlock(text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(),
    )


async def _async_stream_events(events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
    """Real async generator yielding the given StreamEvent list."""
    for event in events:
        yield event


def _make_mock_relay_session(
    *,
    response_text: str = "Hello from relay",
    stream_events: list[StreamEvent] | None = None,
    delete_raises: Exception | None = None,
) -> MagicMock:
    """Return a mock RelaySession with configurable complete() and delete_session() behaviour.

    delete_raises: if set, delete_session() will raise that exception.
    """
    mock = MagicMock()

    # delete_session is synchronous on RelaySession
    if delete_raises is not None:
        mock.delete_session = MagicMock(side_effect=delete_raises)
    else:
        mock.delete_session = MagicMock()

    completion = _make_completion_response(response_text)

    if stream_events is None:
        stream_events = [
            StreamEvent(type="text", text="Hello"),
            StreamEvent(type="text", text=" from relay"),
            StreamEvent(type="end"),
        ]

    async def _complete(request, session_id=None, stream=False):
        if stream:
            return _async_stream_events(stream_events)
        return completion

    mock.complete = _complete
    return mock


def _make_app(relay_session: MagicMock | None = None):
    """Import create_app and build a FastAPI instance with a mock session."""
    from sr2_relay.server import create_app
    config = _make_config()
    return create_app(config, relay_session=relay_session or _make_mock_relay_session())


def _oai_chat_body(
    *,
    model: str = "test-model",
    messages: list[dict] | None = None,
    stream: bool = False,
    **kwargs,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages or [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }
    body.update(kwargs)
    return body


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_ok_body(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# create_app smoke test
# ---------------------------------------------------------------------------


class TestCreateApp:
    @pytest.mark.asyncio
    async def test_create_app_with_relay_session_none_starts_and_health_returns_200(self):
        """create_app(relay_session=None) must not raise; health check must pass.

        When relay_session is None the server uses its own internal session
        management. We can't easily exercise the full relay path here, but
        we verify the app initialises and the health endpoint is reachable.
        """
        from sr2_relay.server import create_app
        config = _make_config()
        app = create_app(config, relay_session=None)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Non-streaming completions
# ---------------------------------------------------------------------------


class TestChatCompletionsNonStreaming:
    @pytest.mark.asyncio
    async def test_returns_200(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_expected_shape(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        body = resp.json()
        assert "id" in body
        assert body["object"] == "chat.completion"
        assert isinstance(body["choices"], list)
        assert len(body["choices"]) >= 1
        assert "usage" in body

    @pytest.mark.asyncio
    async def test_choice_has_index_zero(self):
        """Non-streaming choices[0] must carry index: 0."""
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        assert resp.json()["choices"][0]["index"] == 0

    @pytest.mark.asyncio
    async def test_choice_message_content_matches_relay_response(self):
        session = _make_mock_relay_session(response_text="relay says hi")
        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        body = resp.json()
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "relay says hi"

    @pytest.mark.asyncio
    async def test_choice_has_finish_reason_stop(self):
        """Non-streaming finish_reason must be 'stop'."""
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_usage_has_token_fields(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        usage = resp.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    @pytest.mark.asyncio
    async def test_x_session_id_header_forwarded_to_complete(self):
        """X-Session-ID header value must be passed as session_id to complete()."""
        calls: list[dict] = []

        async def _capturing_complete(request, session_id=None, stream=False):
            calls.append({"session_id": session_id, "stream": stream})
            return _make_completion_response()

        session = _make_mock_relay_session()
        session.complete = _capturing_complete

        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_oai_chat_body(stream=False),
                headers={"X-Session-ID": "my-session-123"},
            )
        assert resp.status_code == 200
        assert len(calls) == 1
        assert calls[0]["session_id"] == "my-session-123"

    @pytest.mark.asyncio
    async def test_no_x_session_id_header_passes_none(self):
        """Without X-Session-ID, session_id=None is forwarded."""
        calls: list[dict] = []

        async def _capturing_complete(request, session_id=None, stream=False):
            calls.append({"session_id": session_id, "stream": stream})
            return _make_completion_response()

        session = _make_mock_relay_session()
        session.complete = _capturing_complete

        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        assert resp.status_code == 200
        assert len(calls) == 1
        assert calls[0]["session_id"] is None

    @pytest.mark.asyncio
    async def test_stream_false_calls_complete_with_stream_false(self):
        calls: list[dict] = []

        async def _capturing_complete(request, session_id=None, stream=False):
            calls.append({"stream": stream})
            return _make_completion_response()

        session = _make_mock_relay_session()
        session.complete = _capturing_complete

        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions", json=_oai_chat_body(stream=False)
            )
        assert calls[0]["stream"] is False

    @pytest.mark.asyncio
    async def test_stream_absent_treated_as_false(self):
        """Body without 'stream' key should default to stream=False."""
        calls: list[dict] = []

        async def _capturing_complete(request, session_id=None, stream=False):
            calls.append({"stream": stream})
            return _make_completion_response()

        session = _make_mock_relay_session()
        session.complete = _capturing_complete

        app = _make_app(relay_session=session)
        body = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert calls[0]["stream"] is False


# ---------------------------------------------------------------------------
# Streaming completions
# ---------------------------------------------------------------------------


class TestChatCompletionsStreaming:
    @pytest.mark.asyncio
    async def test_stream_true_returns_200(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                status = resp.status_code
        assert status == 200

    @pytest.mark.asyncio
    async def test_stream_content_type_is_event_stream(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type

    @pytest.mark.asyncio
    async def test_stream_calls_complete_with_stream_true(self):
        calls: list[dict] = []

        async def _capturing_complete(request, session_id=None, stream=False):
            calls.append({"stream": stream})
            return _async_stream_events([StreamEvent(type="end")])

        session = _make_mock_relay_session()
        session.complete = _capturing_complete

        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                async for _ in resp.aiter_lines():
                    pass
        assert len(calls) == 1
        assert calls[0]["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_chunks_have_sse_data_prefix(self):
        events = [
            StreamEvent(type="text", text="chunk1"),
            StreamEvent(type="end"),
        ]
        session = _make_mock_relay_session(stream_events=events)
        app = _make_app(relay_session=session)

        lines: list[str] = []
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                async for line in resp.aiter_lines():
                    lines.append(line)

        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 1

    @pytest.mark.asyncio
    async def test_stream_text_chunks_contain_delta_content(self):
        events = [
            StreamEvent(type="text", text="hello"),
            StreamEvent(type="end"),
        ]
        session = _make_mock_relay_session(stream_events=events)
        app = _make_app(relay_session=session)

        chunks: list[dict] = []
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        payload = line[len("data: "):]
                        chunks.append(json.loads(payload))

        text_chunks = [
            c for c in chunks
            if c.get("choices", [{}])[0].get("delta", {}).get("content")
        ]
        assert len(text_chunks) >= 1
        assert text_chunks[0]["choices"][0]["delta"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_stream_chunk_has_expected_shape(self):
        events = [StreamEvent(type="text", text="hi"), StreamEvent(type="end")]
        session = _make_mock_relay_session(stream_events=events)
        app = _make_app(relay_session=session)

        chunks: list[dict] = []
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks.append(json.loads(line[len("data: "):]))

        assert len(chunks) >= 1
        chunk = chunks[0]
        assert "id" in chunk
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
        assert isinstance(chunk["choices"], list)

    @pytest.mark.asyncio
    async def test_stream_chunk_choice_has_index_zero(self):
        """Streaming choices[0] must carry index: 0."""
        events = [StreamEvent(type="text", text="hi"), StreamEvent(type="end")]
        session = _make_mock_relay_session(stream_events=events)
        app = _make_app(relay_session=session)

        chunks: list[dict] = []
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks.append(json.loads(line[len("data: "):]))

        # At least one text chunk must have index: 0
        text_chunks = [
            c for c in chunks
            if c.get("choices", [{}])[0].get("delta", {}).get("content")
        ]
        assert len(text_chunks) >= 1
        assert text_chunks[0]["choices"][0]["index"] == 0

    @pytest.mark.asyncio
    async def test_stream_chunk_finish_reason_is_null_for_text_chunks(self):
        """Text-carrying streaming chunks must have finish_reason: null (None)."""
        events = [StreamEvent(type="text", text="hi"), StreamEvent(type="end")]
        session = _make_mock_relay_session(stream_events=events)
        app = _make_app(relay_session=session)

        chunks: list[dict] = []
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunks.append(json.loads(line[len("data: "):]))

        text_chunks = [
            c for c in chunks
            if c.get("choices", [{}])[0].get("delta", {}).get("content")
        ]
        assert len(text_chunks) >= 1
        # finish_reason must be present and null for in-progress chunks
        assert "finish_reason" in text_chunks[0]["choices"][0]
        assert text_chunks[0]["choices"][0]["finish_reason"] is None

    @pytest.mark.asyncio
    async def test_stream_ends_with_done_sentinel(self):
        events = [StreamEvent(type="text", text="hi"), StreamEvent(type="end")]
        session = _make_mock_relay_session(stream_events=events)
        app = _make_app(relay_session=session)

        lines: list[str] = []
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=_oai_chat_body(stream=True)
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        lines.append(line)

        assert lines[-1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_stream_x_session_id_forwarded(self):
        calls: list[dict] = []

        async def _capturing_complete(request, session_id=None, stream=False):
            calls.append({"session_id": session_id, "stream": stream})
            return _async_stream_events([StreamEvent(type="end")])

        session = _make_mock_relay_session()
        session.complete = _capturing_complete

        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=_oai_chat_body(stream=True),
                headers={"X-Session-ID": "stream-session-42"},
            ) as resp:
                async for _ in resp.aiter_lines():
                    pass

        assert calls[0]["session_id"] == "stream-session-42"

    @pytest.mark.asyncio
    async def test_stream_no_x_session_id_passes_none(self):
        """Streaming POST with no X-Session-ID header must call complete with session_id=None."""
        calls: list[dict] = []

        async def _capturing_complete(request, session_id=None, stream=False):
            calls.append({"session_id": session_id})
            return _async_stream_events([StreamEvent(type="end")])

        session = _make_mock_relay_session()
        session.complete = _capturing_complete

        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=_oai_chat_body(stream=True),
            ) as resp:
                async for _ in resp.aiter_lines():
                    pass

        assert len(calls) == 1
        assert calls[0]["session_id"] is None


# ---------------------------------------------------------------------------
# Session deletion
# ---------------------------------------------------------------------------


class TestSessionDeletion:
    @pytest.mark.asyncio
    async def test_delete_session_returns_200(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/v1/sessions/test-session-id")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_session_returns_deleted_true(self):
        app = _make_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/v1/sessions/test-session-id")
        body = resp.json()
        assert body["deleted"] is True
        assert body["session_id"] == "test-session-id"

    @pytest.mark.asyncio
    async def test_delete_session_calls_delete_session_method(self):
        """Server must call relay_session.delete_session(session_id), not _pool.delete."""
        session = _make_mock_relay_session()
        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.delete("/v1/sessions/target-id")
        session.delete_session.assert_called_once_with("target-id")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session_still_returns_200(self):
        """delete_session raises KeyError for unknown session; server must still return 200."""
        session = _make_mock_relay_session(delete_raises=KeyError("ghost-session"))
        app = _make_app(relay_session=session)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/v1/sessions/ghost-session")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    @pytest.mark.asyncio
    async def test_missing_model_returns_422(self):
        """Body without 'model' key is malformed — expect 422 from FastAPI."""
        app = _make_app()
        body = {"messages": [{"role": "user", "content": "Hi"}]}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_messages_returns_422(self):
        """Body with 'model' but no 'messages' is malformed — expect 422 from FastAPI."""
        app = _make_app()
        body = {"model": "test-model"}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_call_type_returns_400(self):
        """When get_translator returns None the server must return 400.

        The patch target is 'sr2_relay.server.get_translator' — this assumes
        server.py uses `from sr2_relay.translators.base import get_translator`.
        If the import style changes, update both the import in server.py and
        this patch path to match.
        """
        app = _make_app()
        with patch(
            "sr2_relay.server.get_translator", return_value=None
        ):
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions", json=_oai_chat_body()
                )
        assert resp.status_code == 400
