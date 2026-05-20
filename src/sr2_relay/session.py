"""Session management for sr2-relay.

Provides:
- fingerprint()  — stable session key from a CanonicalRequest
- SessionPool    — create/retrieve/delete SR2 instances by session_id
- RelaySession   — high-level turn driver for non-streaming and streaming calls
"""
from __future__ import annotations

import hashlib
import inspect
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig
from sr2.models import (
    ContentBlock as SR2ContentBlock,
    Message as SR2Message,
    TextBlock as SR2TextBlock,
    ThinkingBlock as SR2ThinkingBlock,
    TokenUsage,
    ToolResultBlock as SR2ToolResultBlock,
    ToolUseBlock as SR2ToolUseBlock,
)
from sr2.orchestrator import SR2
from sr2.pipeline.protocols import TokenCounter
from sr2.protocols.llm import CompletionResponse, LLMCallable

from sr2_relay.llm import RelayLLMCallable
from sr2_relay.models import CanonicalRequest, SR2RelayConfig


# ---------------------------------------------------------------------------
# Default pipeline config — minimal single-layer config for relay use
# ---------------------------------------------------------------------------


def _default_pipeline_config() -> PipelineConfig:
    """Return a minimal PipelineConfig suitable for relay sessions."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="session",
                resolvers=[
                    ResolverConfig(type="input"),
                    ResolverConfig(type="session"),
                ],
            )
        ]
    )


# ---------------------------------------------------------------------------
# Default token counter — character-based fallback (no tiktoken required)
# ---------------------------------------------------------------------------


class _CharacterTokenCounter:
    """Approximate token counter: 4 characters ≈ 1 token."""

    def count(self, content: list[SR2ContentBlock]) -> int:  # type: ignore[override]
        total = 0
        for block in content:
            if hasattr(block, "text") and block.text:
                total += len(block.text) // 4
        return total


def _default_token_counter() -> TokenCounter:
    return _CharacterTokenCounter()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Type translation helpers
# ---------------------------------------------------------------------------


def _to_sr2_block(block) -> SR2ContentBlock:
    """Translate a CanonicalContentBlock to an sr2 ContentBlock."""
    t = block.type
    if t == "text":
        return SR2TextBlock(text=block.text)
    if t == "tool_use":
        return SR2ToolUseBlock(id=block.id, name=block.name, input=block.input)
    if t == "tool_result":
        return SR2ToolResultBlock(
            tool_use_id=block.tool_use_id,
            content=block.content,
            is_error=block.is_error,
        )
    if t == "thinking":
        return SR2ThinkingBlock(text=block.text)
    raise ValueError(f"Unknown block type: {block.type!r}")


def _to_sr2_message(msg) -> SR2Message:
    return SR2Message(
        role=msg.role,
        content=[_to_sr2_block(b) for b in msg.content],
    )


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def fingerprint(request: CanonicalRequest) -> str:
    """Derive a stable session key from a CanonicalRequest.

    Uses the system prompt text when present; falls back to the first three
    message role+content pairs when the system list is absent or empty.
    """
    if request.system:
        text = "".join(b.text for b in request.system)
    else:
        text = "".join(
            f"{m.role}:{''.join(b.text for b in m.content if hasattr(b, 'text'))}"
            for m in request.messages[:3]
        )
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# SessionPool
# ---------------------------------------------------------------------------


class SessionPool:
    """Maps session_id strings to SR2 orchestrator instances."""

    def __init__(self) -> None:
        self._sessions: dict[str, SR2] = {}

    def get_or_create(self, session_id: str, llm: LLMCallable) -> SR2:
        """Return existing SR2 instance for *session_id*, or create a new one."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        sr2 = SR2(
            pipeline_config=_default_pipeline_config(),
            llm={"default": llm},
            token_counter=_default_token_counter(),
        )
        self._sessions[session_id] = sr2
        return sr2

    def get(self, session_id: str) -> SR2 | None:
        """Return the SR2 instance for *session_id*, or None if not found."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        """Remove a session; no-op if not present."""
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)


# ---------------------------------------------------------------------------
# RelaySession
# ---------------------------------------------------------------------------


class RelaySession:
    """Drives SR2 turns for relay requests."""

    def __init__(self, config: SR2RelayConfig) -> None:
        self._config = config
        self._pool = SessionPool()

    async def complete(
        self,
        request: CanonicalRequest,
        session_id: str | None = None,
        stream: bool = False,
    ) -> CompletionResponse | AsyncIterator:
        # 1. Resolve session_id
        sid = session_id if session_id is not None else fingerprint(request)

        # 2. Check if session already exists BEFORE get_or_create
        is_new = self._pool.get(sid) is None

        # 3. Build LLM callable
        llm = RelayLLMCallable(model=request.model, base_url=self._config.api_base)

        # 4. Get or create SR2 instance
        sr2 = self._pool.get_or_create(sid, llm)

        # 5. Split messages: prior = all but last; current = last
        messages = request.messages
        prior = messages[:-1]
        current = messages[-1]

        # 6. Seed new sessions that have prior history
        if is_new and prior:
            sr2.seed_session([_to_sr2_message(m) for m in prior])

        # 7. Build sr2 content blocks for the current user turn
        user_input = [_to_sr2_block(b) for b in current.content]

        # 8. Stream or accumulate
        # sr2.turn() is an async generator function in production; tests may
        # mock it as an AsyncMock (coroutine returning an async generator).
        turn_result = sr2.turn(user_input)
        if inspect.isawaitable(turn_result):
            turn_gen = await turn_result
        else:
            turn_gen = turn_result

        if stream:
            return turn_gen
        else:
            accumulated: list[str] = []
            async for event in turn_gen:
                if event.type == "text" and hasattr(event, "text"):
                    accumulated.append(event.text)
            return CompletionResponse(
                id="relay-response",
                content=[SR2TextBlock(text="".join(accumulated))],
                stop_reason="end_turn",
                usage=TokenUsage(),
            )
