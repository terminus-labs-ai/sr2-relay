from __future__ import annotations

from typing import Any

from sr2_relay.models.canonical import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalThinkingBlock,
    CanonicalThinkingConfig,
    CanonicalToolDef,
    CanonicalToolResultBlock,
    CanonicalToolUseBlock,
)
from sr2_relay.translators.base import RequestTranslator

# Keys that map directly to CanonicalRequest fields
_CANONICAL_FIELDS = {"model", "max_tokens", "temperature", "stream", "messages", "tools", "thinking", "system"}


class AnthropicTranslator(RequestTranslator):
    call_types = ("anthropic_messages",)

    def to_canonical(self, data: dict, *, call_type: str | None = None) -> CanonicalRequest:
        # System
        system_raw = data.get("system")
        system: list[CanonicalTextBlock] | None = None
        if system_raw is not None:
            if isinstance(system_raw, str):
                system = [CanonicalTextBlock(text=system_raw)]
            else:
                system = [
                    CanonicalTextBlock(
                        text=block["text"],
                        cache_control=block.get("cache_control"),
                    )
                    for block in system_raw
                ]

        # Messages
        messages = [self._convert_message(msg) for msg in data.get("messages", [])]

        # Tools
        tools: list[CanonicalToolDef] | None = None
        if data.get("tools") is not None:
            tools = [
                CanonicalToolDef(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t["input_schema"],
                )
                for t in data["tools"]
            ]

        # Thinking config
        thinking: CanonicalThinkingConfig | None = None
        if data.get("thinking") is not None:
            thinking = CanonicalThinkingConfig(type=data["thinking"]["type"])

        # Provider extras — everything not in canonical fields
        provider_extras: dict[str, Any] = {}
        for key, value in data.items():
            if key not in _CANONICAL_FIELDS:
                provider_extras[key] = value

        return CanonicalRequest(
            call_type="anthropic_messages",
            model=data["model"],
            max_tokens=data.get("max_tokens", 16384),
            temperature=data.get("temperature"),
            stream=data.get("stream", False),
            system=system,
            messages=messages,
            tools=tools,
            thinking=thinking,
            provider_extras=provider_extras,
        )

    def _convert_message(self, msg: dict) -> CanonicalMessage:
        role = msg["role"]
        content_raw = msg.get("content", [])
        content = [self._convert_block(b) for b in content_raw]
        return CanonicalMessage(role=role, content=content)

    def _convert_block(self, block: dict) -> Any:
        btype = block["type"]
        if btype == "text":
            return CanonicalTextBlock(
                text=block["text"],
                cache_control=block.get("cache_control"),
            )
        elif btype == "tool_use":
            return CanonicalToolUseBlock(
                id=block["id"],
                name=block["name"],
                input=block["input"],
            )
        elif btype == "tool_result":
            return CanonicalToolResultBlock(
                tool_use_id=block["tool_use_id"],
                content=block["content"],
                is_error=block.get("is_error", False),
                cache_control=block.get("cache_control"),
            )
        elif btype == "thinking":
            return CanonicalThinkingBlock(text=block.get("text", "") or block.get("thinking", ""))
        else:
            # Pass through unknown block types as text
            return CanonicalTextBlock(text=str(block))

    def from_canonical(self, request: CanonicalRequest) -> dict:
        out: dict[str, Any] = {}

        # Unpack provider_extras first (lowest priority)
        out.update(request.provider_extras)

        out["model"] = request.model
        out["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            out["temperature"] = request.temperature

        out["stream"] = request.stream

        # System
        if request.system:
            out["system"] = [self._block_to_anth(b) for b in request.system]

        # Messages
        out["messages"] = [self._message_to_anth(m) for m in request.messages]

        # Tools
        if request.tools:
            out["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in request.tools
            ]

        # Thinking
        if request.thinking is not None:
            out["thinking"] = {"type": request.thinking.type}

        return out

    def _block_to_anth(self, block: Any) -> dict:
        if isinstance(block, CanonicalTextBlock):
            d: dict[str, Any] = {"type": "text", "text": block.text}
            if block.cache_control is not None:
                d["cache_control"] = block.cache_control
            return d
        elif isinstance(block, CanonicalToolUseBlock):
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        elif isinstance(block, CanonicalToolResultBlock):
            d = {"type": "tool_result", "tool_use_id": block.tool_use_id, "content": block.content}
            if block.is_error:
                d["is_error"] = block.is_error
            if block.cache_control is not None:
                d["cache_control"] = block.cache_control
            return d
        elif isinstance(block, CanonicalThinkingBlock):
            return {"type": "thinking", "text": block.text}
        # dict (from model_validate passthrough)
        return dict(block) if not isinstance(block, dict) else block

    def _message_to_anth(self, msg: CanonicalMessage) -> dict:
        return {
            "role": msg.role,
            "content": [self._block_to_anth(b) for b in msg.content],
        }
