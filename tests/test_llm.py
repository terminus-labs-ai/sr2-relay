from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import Message, TextBlock, TokenUsage
from sr2.protocols.llm import (
  CompletionRequest,
  CompletionResponse,
  LLMCallable,
  StreamEvent,
)
from sr2_relay.llm import RelayLLMCallable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(role: str, *texts: str) -> Message:
  return Message(role=role, content=[TextBlock(text=t) for t in texts])


def _make_request(
  messages: list[Message] | None = None,
  system: list[TextBlock] | None = None,
) -> CompletionRequest:
  return CompletionRequest(
    messages=messages or [_make_message("user", "Hello")],
    system=system,
  )


def _mock_litellm_response(
  id: str = "resp-1",
  content: str = "Hi there",
  finish_reason: str = "stop",
  prompt_tokens: int = 10,
  completion_tokens: int = 5,
) -> MagicMock:
  usage = MagicMock()
  usage.prompt_tokens = prompt_tokens
  usage.completion_tokens = completion_tokens

  choice = MagicMock()
  choice.message.content = content
  choice.finish_reason = finish_reason

  resp = MagicMock()
  resp.id = id
  resp.choices = [choice]
  resp.usage = usage
  return resp


async def _async_gen(*chunks):
  for chunk in chunks:
    yield chunk


def _make_stream_chunk(
  content: str | None = None,
  prompt_tokens: int | None = None,
  completion_tokens: int | None = None,
) -> MagicMock:
  delta = MagicMock()
  delta.content = content

  choice = MagicMock()
  choice.delta = delta

  chunk = MagicMock()
  chunk.choices = [choice]

  if prompt_tokens is not None:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    chunk.usage = usage
  else:
    chunk.usage = None

  return chunk


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
  def test_isinstance_llm_callable(self):
    instance = RelayLLMCallable("test-model")
    assert isinstance(instance, LLMCallable)


# ---------------------------------------------------------------------------
# 2. complete — basic call
# ---------------------------------------------------------------------------


class TestCompleteBasic:
  @pytest.mark.asyncio
  async def test_calls_litellm_with_correct_model(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("claude-sonnet-4-5")
      await client.complete(_make_request())
      mock_ac.assert_called_once()
      assert mock_ac.call_args.kwargs["model"] == "claude-sonnet-4-5"

  @pytest.mark.asyncio
  async def test_returns_completion_response(self):
    resp = _mock_litellm_response(id="resp-42", content="Answer", finish_reason="stop")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = RelayLLMCallable("claude-sonnet-4-5")
      result = await client.complete(_make_request())

    assert isinstance(result, CompletionResponse)
    assert result.id == "resp-42"
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == "Answer"
    assert result.stop_reason == "stop"

  @pytest.mark.asyncio
  async def test_messages_passed_to_litellm(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model")
      request = _make_request(messages=[_make_message("user", "Hello")])
      await client.complete(request)
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert len(messages_arg) == 1
      assert messages_arg[0]["role"] == "user"
      assert messages_arg[0]["content"] == "Hello"


# ---------------------------------------------------------------------------
# 3. complete — system prompt passed as kwarg
# ---------------------------------------------------------------------------


class TestCompleteSystemPrompt:
  @pytest.mark.asyncio
  async def test_system_prompt_passed_as_kwarg(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model")
      request = _make_request(system=[TextBlock(text="You are a helpful assistant.")])
      await client.complete(request)
      assert mock_ac.call_args.kwargs.get("system") == "You are a helpful assistant."

  @pytest.mark.asyncio
  async def test_multiple_system_blocks_joined(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model")
      request = _make_request(
        system=[TextBlock(text="Block one. "), TextBlock(text="Block two.")]
      )
      await client.complete(request)
      assert mock_ac.call_args.kwargs.get("system") == "Block one. Block two."


# ---------------------------------------------------------------------------
# 4. complete — no system prompt
# ---------------------------------------------------------------------------


class TestCompleteNoSystemPrompt:
  @pytest.mark.asyncio
  async def test_no_system_kwarg_when_none(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model")
      await client.complete(_make_request(system=None))
      assert "system" not in mock_ac.call_args.kwargs


# ---------------------------------------------------------------------------
# 5. complete — message content joining
# ---------------------------------------------------------------------------


class TestCompleteMessageContentJoining:
  @pytest.mark.asyncio
  async def test_multiple_text_blocks_joined_into_one_string(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model")
      request = _make_request(
        messages=[_make_message("user", "Part one. ", "Part two.")]
      )
      await client.complete(request)
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert messages_arg[0]["content"] == "Part one. Part two."

  @pytest.mark.asyncio
  async def test_multiple_messages_each_joined_independently(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model")
      request = _make_request(
        messages=[
          _make_message("user", "Hello"),
          _make_message("assistant", "Hi", " there"),
        ]
      )
      await client.complete(request)
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert messages_arg[0]["content"] == "Hello"
      assert messages_arg[1]["content"] == "Hi there"


# ---------------------------------------------------------------------------
# 6. complete — usage populated
# ---------------------------------------------------------------------------


class TestCompleteUsage:
  @pytest.mark.asyncio
  async def test_input_tokens_from_litellm_response(self):
    resp = _mock_litellm_response(prompt_tokens=42, completion_tokens=17)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = RelayLLMCallable("model")
      result = await client.complete(_make_request())
      assert result.usage.input_tokens == 42

  @pytest.mark.asyncio
  async def test_output_tokens_from_litellm_response(self):
    resp = _mock_litellm_response(prompt_tokens=42, completion_tokens=17)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = RelayLLMCallable("model")
      result = await client.complete(_make_request())
      assert result.usage.output_tokens == 17

  @pytest.mark.asyncio
  async def test_usage_is_token_usage_instance(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = RelayLLMCallable("model")
      result = await client.complete(_make_request())
      assert isinstance(result.usage, TokenUsage)


# ---------------------------------------------------------------------------
# 7. complete — kwargs forwarded
# ---------------------------------------------------------------------------


class TestCompleteKwargsForwarded:
  @pytest.mark.asyncio
  async def test_init_kwargs_forwarded_to_litellm(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model", temperature=0.7, max_tokens=256)
      await client.complete(_make_request())
      assert mock_ac.call_args.kwargs.get("temperature") == 0.7
      assert mock_ac.call_args.kwargs.get("max_tokens") == 256

  @pytest.mark.asyncio
  async def test_base_url_forwarded_to_litellm(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("model", base_url="http://localhost:4000")
      await client.complete(_make_request())
      assert mock_ac.call_args.kwargs.get("base_url") == "http://localhost:4000"

  @pytest.mark.asyncio
  async def test_no_base_url_kwarg_when_none(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = RelayLLMCallable("gpt-4o")  # no base_url
      await client.complete(_make_request())
      assert "base_url" not in mock_ac.call_args.kwargs


# ---------------------------------------------------------------------------
# 8. stream — yields text events
# ---------------------------------------------------------------------------


class TestStreamTextEvents:
  @pytest.mark.asyncio
  async def test_yields_text_event_per_content_chunk(self):
    chunks = [
      _make_stream_chunk(content="Hello"),
      _make_stream_chunk(content=" world"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 2
    assert text_events[0].text == "Hello"
    assert text_events[1].text == " world"

  @pytest.mark.asyncio
  async def test_stream_event_type_is_text(self):
    chunks = [_make_stream_chunk(content="Hi")]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert all(isinstance(e, StreamEvent) for e in text_events)

  @pytest.mark.asyncio
  async def test_stream_true_passed_to_litellm(self):
    chunk = _make_stream_chunk(content="Hello")
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
      mock_ac.return_value = _async_gen(chunk)
      client = RelayLLMCallable("gpt-4o")
      async for _ in client.stream(_make_request()):
        pass
    assert mock_ac.call_args.kwargs.get("stream") is True


# ---------------------------------------------------------------------------
# 9. stream — yields usage event
# ---------------------------------------------------------------------------


class TestStreamUsageEvent:
  @pytest.mark.asyncio
  async def test_usage_event_yielded_when_chunk_has_usage(self):
    chunks = [
      _make_stream_chunk(content="Hello"),
      _make_stream_chunk(content=None, prompt_tokens=8, completion_tokens=3),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage.input_tokens == 8
    assert usage_events[0].usage.output_tokens == 3

  @pytest.mark.asyncio
  async def test_no_usage_event_when_no_chunk_has_usage(self):
    chunks = [
      _make_stream_chunk(content="Hello"),
      _make_stream_chunk(content=" there"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 0


# ---------------------------------------------------------------------------
# 10. stream — yields end event
# ---------------------------------------------------------------------------


class TestStreamEndEvent:
  @pytest.mark.asyncio
  async def test_end_event_always_last(self):
    chunks = [_make_stream_chunk(content="Hi")]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    assert events[-1].type == "end"

  @pytest.mark.asyncio
  async def test_end_event_present_even_with_no_content_chunks(self):
    async def fake_acompletion(*args, **kwargs):
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    assert len(events) == 1
    assert events[0].type == "end"


# ---------------------------------------------------------------------------
# 11. stream — system prompt passed as kwarg
# ---------------------------------------------------------------------------


class TestStreamSystemPrompt:
  @pytest.mark.asyncio
  async def test_system_prompt_passed_as_kwarg_in_stream(self):
    captured: dict = {}

    async def fake_acompletion(*args, **kwargs):
      captured.update(kwargs)
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      request = _make_request(system=[TextBlock(text="Be concise.")])
      _ = [e async for e in client.stream(request)]

    assert captured.get("system") == "Be concise."

  @pytest.mark.asyncio
  async def test_no_system_kwarg_in_stream_when_none(self):
    captured: dict = {}

    async def fake_acompletion(*args, **kwargs):
      captured.update(kwargs)
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      _ = [e async for e in client.stream(_make_request(system=None))]

    assert "system" not in captured


# ---------------------------------------------------------------------------
# 12. stream — skips empty/None content chunks
# ---------------------------------------------------------------------------


class TestStreamSkipsEmptyChunks:
  @pytest.mark.asyncio
  async def test_none_content_chunk_does_not_yield_text_event(self):
    chunks = [
      _make_stream_chunk(content=None),
      _make_stream_chunk(content="Real content"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 1
    assert text_events[0].text == "Real content"

  @pytest.mark.asyncio
  async def test_empty_string_content_chunk_does_not_yield_text_event(self):
    chunks = [
      _make_stream_chunk(content=""),
      _make_stream_chunk(content="Valid"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = RelayLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 1
    assert text_events[0].text == "Valid"
