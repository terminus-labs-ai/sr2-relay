"""Tests for sr2_relay.session — fingerprint, SessionPool, RelaySession."""
from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2_relay.models import (
  CanonicalMessage,
  CanonicalRequest,
  CanonicalTextBlock,
  ModelSlotConfig,
  SR2RelayConfig,
)
from sr2_relay.session import RelaySession, SessionPool, fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system(*texts: str) -> list[CanonicalTextBlock]:
  return [CanonicalTextBlock(text=t) for t in texts]


def _make_message(role: str, *texts: str) -> CanonicalMessage:
  return CanonicalMessage(
    role=role,
    content=[CanonicalTextBlock(text=t) for t in texts],
  )


def _make_request(
  model: str = "test-model",
  system: list[CanonicalTextBlock] | None = None,
  messages: list[CanonicalMessage] | None = None,
) -> CanonicalRequest:
  return CanonicalRequest(
    call_type="messages",
    model=model,
    system=system,
    messages=messages or [_make_message("user", "Hello")],
  )


def _sha256(text: str) -> str:
  return hashlib.sha256(text.encode()).hexdigest()


def _mock_sr2() -> MagicMock:
  """Return a mock SR2 instance with async turn() and sync seed_session()."""
  sr2 = MagicMock()
  sr2.seed_session = MagicMock()

  async def _turn(user_input):
    yield MagicMock(type="text", text="response")
    yield MagicMock(type="end")

  sr2.turn = _turn
  return sr2


def _mock_llm_callable() -> MagicMock:
  return MagicMock()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
  def test_same_system_prompt_same_fingerprint(self):
    req1 = _make_request(system=_make_system("You are helpful."))
    req2 = _make_request(system=_make_system("You are helpful."))
    assert fingerprint(req1) == fingerprint(req2)

  def test_different_system_prompt_different_fingerprint(self):
    req1 = _make_request(system=_make_system("You are helpful."))
    req2 = _make_request(system=_make_system("You are a pirate."))
    assert fingerprint(req1) != fingerprint(req2)

  def test_no_system_uses_first_3_messages_as_fallback(self):
    messages = [
      _make_message("user", "msg1"),
      _make_message("assistant", "msg2"),
      _make_message("user", "msg3"),
    ]
    req = _make_request(system=None, messages=messages)
    result = fingerprint(req)
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 hex

  def test_no_system_same_first_3_messages_same_fingerprint(self):
    messages = [
      _make_message("user", "alpha"),
      _make_message("assistant", "beta"),
      _make_message("user", "gamma"),
      _make_message("assistant", "delta"),  # extra — not used
    ]
    req1 = _make_request(system=None, messages=messages)
    req2 = _make_request(system=None, messages=messages)
    assert fingerprint(req1) == fingerprint(req2)

  def test_system_present_takes_priority_over_messages(self):
    # Same first-3 messages, different system → should differ
    messages = [_make_message("user", "Hello")]
    req_with_system = _make_request(
      system=_make_system("System A"), messages=messages
    )
    req_no_system = _make_request(system=None, messages=messages)
    assert fingerprint(req_with_system) != fingerprint(req_no_system)

  def test_empty_system_list_falls_back_to_messages(self):
    messages = [
      _make_message("user", "first"),
      _make_message("assistant", "second"),
    ]
    req_empty_system = _make_request(system=[], messages=messages)
    req_no_system = _make_request(system=None, messages=messages)
    assert fingerprint(req_empty_system) == fingerprint(req_no_system)

  def test_fingerprint_is_sha256_hex(self):
    req = _make_request(system=_make_system("System prompt"))
    result = fingerprint(req)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# SessionPool
# ---------------------------------------------------------------------------


class TestSessionPool:
  def _pool(self) -> SessionPool:
    return SessionPool()

  def _make_sr2_class_mock(self):
    """Patch SR2 class so get_or_create can instantiate it."""
    sr2_instance = _mock_sr2()
    sr2_class = MagicMock(return_value=sr2_instance)
    return sr2_class, sr2_instance

  def test_get_or_create_returns_sr2_instance(self):
    pool = self._pool()
    llm = _mock_llm_callable()
    with patch("sr2_relay.session.SR2") as mock_sr2_class:
      mock_sr2_class.return_value = _mock_sr2()
      result = pool.get_or_create("session-1", llm)
    assert result is not None

  def test_same_session_id_returns_same_instance(self):
    pool = self._pool()
    llm = _mock_llm_callable()
    with patch("sr2_relay.session.SR2") as mock_sr2_class:
      mock_sr2_class.return_value = _mock_sr2()
      first = pool.get_or_create("session-abc", llm)
      second = pool.get_or_create("session-abc", llm)
    assert first is second

  def test_different_session_ids_return_different_instances(self):
    pool = self._pool()
    llm = _mock_llm_callable()
    with patch("sr2_relay.session.SR2") as mock_sr2_class:
      mock_sr2_class.side_effect = [_mock_sr2(), _mock_sr2()]
      first = pool.get_or_create("session-1", llm)
      second = pool.get_or_create("session-2", llm)
    assert first is not second

  def test_get_returns_none_for_unknown_session(self):
    pool = self._pool()
    assert pool.get("nonexistent") is None

  def test_get_returns_instance_for_known_session(self):
    pool = self._pool()
    llm = _mock_llm_callable()
    with patch("sr2_relay.session.SR2") as mock_sr2_class:
      mock_sr2_class.return_value = _mock_sr2()
      created = pool.get_or_create("session-x", llm)
    assert pool.get("session-x") is created

  def test_delete_removes_session(self):
    pool = self._pool()
    llm = _mock_llm_callable()
    with patch("sr2_relay.session.SR2") as mock_sr2_class:
      mock_sr2_class.return_value = _mock_sr2()
      pool.get_or_create("session-del", llm)
    pool.delete("session-del")
    assert pool.get("session-del") is None

  def test_delete_unknown_session_does_not_raise(self):
    pool = self._pool()
    pool.delete("does-not-exist")  # should not raise

  def test_len_reflects_active_session_count(self):
    pool = self._pool()
    llm = _mock_llm_callable()
    assert len(pool) == 0
    with patch("sr2_relay.session.SR2") as mock_sr2_class:
      mock_sr2_class.side_effect = [_mock_sr2(), _mock_sr2(), _mock_sr2()]
      pool.get_or_create("s1", llm)
      assert len(pool) == 1
      pool.get_or_create("s2", llm)
      assert len(pool) == 2
      pool.get_or_create("s3", llm)
      assert len(pool) == 3
    pool.delete("s2")
    assert len(pool) == 2


# ---------------------------------------------------------------------------
# RelaySession.complete — helpers
# ---------------------------------------------------------------------------


def _make_relay_config() -> SR2RelayConfig:
  """Minimal real SR2RelayConfig for tests."""
  return SR2RelayConfig(
    api_base="http://localhost:11434",
    api_key="test-key",
    model=ModelSlotConfig(model="gpt-4o", api_base="http://localhost:11434"),
  )


def _make_canonical_request(
  model: str = "test-model",
  system: str | None = None,
  messages: list[CanonicalMessage] | None = None,
) -> CanonicalRequest:
  """Build a CanonicalRequest; system is a plain string shorthand."""
  sys_blocks = [CanonicalTextBlock(text=system)] if system else None
  return _make_request(model=model, system=sys_blocks, messages=messages)


async def _async_stream(events):
  """Yield a sequence as an async generator."""
  for event in events:
    yield event


async def _collect_stream(gen: AsyncIterator) -> list:
  return [event async for event in gen]


# ---------------------------------------------------------------------------
# RelaySession.complete — non-streaming
# ---------------------------------------------------------------------------


class TestRelaySessionCompleteNonStreaming:
  @pytest.mark.asyncio
  async def test_returns_completion_response(self):
    from sr2.protocols.llm import CompletionResponse

    config = _make_relay_config()
    session = RelaySession(config)

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_instance = _mock_sr2()
      mock_sr2_class.return_value = mock_sr2_instance

      result = await session.complete(
        _make_request(messages=[_make_message("user", "Hello")]),
        stream=False,
      )

    assert isinstance(result, CompletionResponse)

  @pytest.mark.asyncio
  async def test_new_session_no_prior_messages_seed_not_called(self):
    config = _make_relay_config()
    session = RelaySession(config)

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_instance = _mock_sr2()
      mock_sr2_class.return_value = mock_sr2_instance

      # Single message: no prior messages
      await session.complete(
        _make_request(messages=[_make_message("user", "Hello")]),
        stream=False,
      )

    mock_sr2_instance.seed_session.assert_not_called()

  @pytest.mark.asyncio
  async def test_new_session_with_prior_messages_seed_called(self):
    from sr2.models import Message as SR2Message

    config = _make_relay_config()
    session = RelaySession(config)

    messages = [
      _make_message("user", "First"),
      _make_message("assistant", "Second"),
      _make_message("user", "Third"),  # last user turn — NOT in prior messages
    ]

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_instance = _mock_sr2()
      mock_sr2_class.return_value = mock_sr2_instance

      await session.complete(_make_request(messages=messages), stream=False)

    mock_sr2_instance.seed_session.assert_called_once()
    seeded = mock_sr2_instance.seed_session.call_args[0][0]
    # Prior messages = all except the last
    assert len(seeded) == 2
    assert all(isinstance(m, SR2Message) for m in seeded)
    assert seeded[0].role == "user"
    assert seeded[1].role == "assistant"

  @pytest.mark.asyncio
  async def test_existing_session_reseeded_on_subsequent_calls(self):
    """Relay re-seeds from caller-provided history on every turn.

    Clients like Hermes send the full conversation history with each request
    and are the authoritative source of truth. Re-seeding on every turn
    ensures the compiled context is always correct without a one-turn lag.
    """
    config = _make_relay_config()
    session = RelaySession(config)

    messages = [
      _make_message("user", "First"),
      _make_message("assistant", "Second"),
      _make_message("user", "Third"),
    ]
    request = _make_request(messages=messages)

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_instance = _mock_sr2()
      mock_sr2_class.return_value = mock_sr2_instance

      # First call creates the session
      await session.complete(request, session_id="fixed-id", stream=False)
      mock_sr2_instance.seed_session.reset_mock()

      # Second call: should re-seed with the same prior history
      await session.complete(request, session_id="fixed-id", stream=False)

    mock_sr2_instance.seed_session.assert_called_once()

  @pytest.mark.asyncio
  async def test_turn_called_with_last_message_content(self):
    from sr2.models import TextBlock as SR2TextBlock

    config = _make_relay_config()
    session = RelaySession(config)

    messages = [
      _make_message("user", "First"),
      _make_message("assistant", "Reply"),
      _make_message("user", "Final input"),
    ]

    turn_calls = []

    async def _capturing_turn(user_input):
      turn_calls.append(user_input)
      yield MagicMock(type="text", text="ok")
      yield MagicMock(type="end")

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_instance = _mock_sr2()
      mock_sr2_instance.turn = _capturing_turn
      mock_sr2_class.return_value = mock_sr2_instance

      await session.complete(_make_request(messages=messages), stream=False)

    assert len(turn_calls) == 1
    user_input = turn_calls[0]
    # turn() receives sr2 ContentBlock objects, not CanonicalContentBlock
    assert isinstance(user_input, list)
    assert all(isinstance(b, SR2TextBlock) for b in user_input)
    assert user_input[0].text == "Final input"

  @pytest.mark.asyncio
  async def test_provided_session_id_used_directly(self):
    config = _make_relay_config()
    session = RelaySession(config)

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
      patch("sr2_relay.session.fingerprint") as mock_fp,
    ):
      mock_sr2_class.return_value = _mock_sr2()

      await session.complete(
        _make_request(),
        session_id="explicit-id",
        stream=False,
      )

    mock_fp.assert_not_called()

  @pytest.mark.asyncio
  async def test_no_session_id_computes_fingerprint(self):
    config = _make_relay_config()
    session = RelaySession(config)

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
      patch("sr2_relay.session.fingerprint", return_value="fp-abc") as mock_fp,
    ):
      mock_sr2_class.return_value = _mock_sr2()
      req = _make_request()

      await session.complete(req, session_id=None, stream=False)

    mock_fp.assert_called_once_with(req)

  @pytest.mark.asyncio
  async def test_fingerprint_same_request_no_reseed_on_second_call(self):
    config = _make_relay_config()
    relay = RelaySession(config)

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_instance = AsyncMock()
      mock_sr2_instance.turn.return_value = _async_stream([])
      mock_sr2_class.return_value = mock_sr2_instance

      # Single-message request: no prior messages, so no seed on first call either
      request = _make_canonical_request(system="fixed system prompt")
      await relay.complete(request, session_id=None)
      await relay.complete(request, session_id=None)

    # seed_session never called: no prior messages, and second call reuses existing session
    mock_sr2_instance.seed_session.assert_not_called()
    # SR2 constructor called only once (same fingerprint → same session reused)
    assert mock_sr2_class.call_count == 1


# ---------------------------------------------------------------------------
# RelaySession.complete — streaming
# ---------------------------------------------------------------------------


class TestRelaySessionCompleteStreaming:
  @pytest.mark.asyncio
  async def test_stream_true_returns_async_iterator(self):
    config = _make_relay_config()
    session = RelaySession(config)

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_class.return_value = _mock_sr2()

      result = await session.complete(_make_request(), stream=True)

    # Should be an async iterator, not a CompletionResponse
    assert hasattr(result, "__aiter__")

  @pytest.mark.asyncio
  async def test_stream_yields_events_from_turn(self):
    config = _make_relay_config()
    session = RelaySession(config)

    expected_events = [
      MagicMock(type="text", text="Hello"),
      MagicMock(type="text", text=" world"),
      MagicMock(type="end"),
    ]

    async def _turn_with_events(user_input):
      for event in expected_events:
        yield event

    with (
      patch("sr2_relay.session.SR2") as mock_sr2_class,
      patch("sr2_relay.session.RelayLLMCallable"),
    ):
      mock_sr2_instance = _mock_sr2()
      mock_sr2_instance.turn = _turn_with_events
      mock_sr2_class.return_value = mock_sr2_instance

      gen = await session.complete(_make_request(), stream=True)
      collected = await _collect_stream(gen)

    assert len(collected) == 3
    assert collected[0].type == "text"
    assert collected[0].text == "Hello"
    assert collected[-1].type == "end"
