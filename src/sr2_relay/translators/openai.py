from __future__ import annotations

import json
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

# Keys that map to CanonicalRequest fields (handled explicitly)
_CANONICAL_FIELDS = {"model", "max_tokens", "temperature", "stream", "messages", "tools"}

# Keys that go to provider_extras
_PROVIDER_EXTRAS_KEYS = {"stream_options", "store", "chat_template_kwargs"}


class OpenAITranslator(RequestTranslator):
    call_types = ("completion", "acompletion")

    def to_canonical(self, data: dict, *, call_type: str | None = None) -> CanonicalRequest:
        effective_call_type = call_type if call_type is not None else self.call_types[0]

        raw_messages: list[dict] = list(data.get("messages", []))

        # Extract system messages
        system_msgs = [m for m in raw_messages if m["role"] == "system"]
        non_system = [m for m in raw_messages if m["role"] != "system"]

        system: list[CanonicalTextBlock] | None = None
        if system_msgs:
            system = [CanonicalTextBlock(text=m["content"]) for m in system_msgs]

        # Convert remaining messages, merging consecutive tool messages
        messages = self._convert_messages(non_system)

        # Tools
        tools: list[CanonicalToolDef] | None = None
        if data.get("tools"):
            tools = [
                CanonicalToolDef(
                    name=t["function"]["name"],
                    description=t["function"].get("description", ""),
                    input_schema=t["function"].get("parameters", {}),
                )
                for t in data["tools"]
            ]

        # max_tokens with default tracking
        provider_extras: dict[str, Any] = {}
        if "max_tokens" in data:
            max_tokens = data["max_tokens"]
        else:
            max_tokens = 16384
            provider_extras["max_tokens_was_default"] = True

        # Provider-specific extras
        for key in _PROVIDER_EXTRAS_KEYS:
            if key in data:
                provider_extras[key] = data[key]

        # thinking config from chat_template_kwargs
        thinking: CanonicalThinkingConfig | None = None
        ctk = data.get("chat_template_kwargs", {})
        if ctk.get("enable_thinking"):
            thinking = CanonicalThinkingConfig(type="enabled")

        return CanonicalRequest(
            call_type=effective_call_type,
            model=data["model"],
            max_tokens=max_tokens,
            temperature=data.get("temperature"),
            stream=data.get("stream", False),
            system=system,
            messages=messages,
            tools=tools,
            thinking=thinking,
            provider_extras=provider_extras,
        )

    def _convert_messages(self, msgs: list[dict]) -> list[CanonicalMessage]:
        result: list[CanonicalMessage] = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            role = msg["role"]

            if role == "assistant":
                content: list[Any] = []

                # reasoning_content → thinking block (prepend)
                if msg.get("reasoning_content"):
                    content.append(CanonicalThinkingBlock(text=msg["reasoning_content"]))

                # text content
                raw_content = msg.get("content")
                if raw_content:
                    if isinstance(raw_content, str):
                        content.append(CanonicalTextBlock(text=raw_content))
                    elif isinstance(raw_content, list):
                        for block in raw_content:
                            if block.get("type") == "text":
                                content.append(CanonicalTextBlock(text=block["text"]))

                # tool_calls → CanonicalToolUseBlock
                for tc in msg.get("tool_calls") or []:
                    fn = tc["function"]
                    arguments_raw = fn.get("arguments", "{}")
                    parsed_input = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
                    content.append(
                        CanonicalToolUseBlock(
                            id=tc["id"],
                            name=fn["name"],
                            input=parsed_input,
                        )
                    )

                result.append(CanonicalMessage(role="assistant", content=content))
                i += 1

            elif role == "tool":
                # Collect consecutive tool messages into a single user message
                tool_blocks: list[CanonicalToolResultBlock] = []
                while i < len(msgs) and msgs[i]["role"] == "tool":
                    tm = msgs[i]
                    tool_blocks.append(
                        CanonicalToolResultBlock(
                            tool_use_id=tm["tool_call_id"],
                            content=tm["content"],
                        )
                    )
                    i += 1
                result.append(CanonicalMessage(role="user", content=tool_blocks))

            elif role == "user":
                raw_content = msg.get("content", "")
                if isinstance(raw_content, str):
                    blocks: list[Any] = [CanonicalTextBlock(text=raw_content)]
                else:
                    blocks = []
                    for block in raw_content:
                        if block.get("type") == "text":
                            blocks.append(CanonicalTextBlock(text=block["text"]))
                result.append(CanonicalMessage(role="user", content=blocks))
                i += 1
            else:
                i += 1

        return result

    def from_canonical(self, request: CanonicalRequest) -> dict:
        out: dict[str, Any] = {}
        out["model"] = request.model

        # max_tokens — omit if it was defaulted
        if not request.provider_extras.get("max_tokens_was_default"):
            out["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            out["temperature"] = request.temperature

        if request.stream:
            out["stream"] = request.stream

        messages: list[dict] = []

        # System blocks → single system role message with joined text
        if request.system:
            joined = " ".join(b.text for b in request.system)
            messages.append({"role": "system", "content": joined})

        # Convert canonical messages
        for msg in request.messages:
            converted = self._message_to_oai(msg)
            messages.extend(converted)

        out["messages"] = messages

        # Tools
        if request.tools:
            out["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]

        # Re-emit provider extras (except internal flags)
        for key, value in request.provider_extras.items():
            if key == "max_tokens_was_default":
                continue
            out[key] = value

        return out

    def _message_to_oai(self, msg: CanonicalMessage) -> list[dict]:
        # Check if message contains tool result blocks → emit as individual tool messages
        tool_result_blocks = [b for b in msg.content if isinstance(b, CanonicalToolResultBlock)]
        if tool_result_blocks:
            return [
                {
                    "role": "tool",
                    "tool_call_id": b.tool_use_id,
                    "content": b.content if isinstance(b.content, str) else " ".join(t.text for t in b.content),
                }
                for b in tool_result_blocks
            ]

        # Check if message contains tool use blocks → assistant with tool_calls
        tool_use_blocks = [b for b in msg.content if isinstance(b, CanonicalToolUseBlock)]
        text_blocks = [b for b in msg.content if isinstance(b, CanonicalTextBlock)]

        if tool_use_blocks:
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.input),
                    },
                }
                for b in tool_use_blocks
            ]
            text_content = " ".join(b.text for b in text_blocks) if text_blocks else None
            return [{"role": "assistant", "content": text_content, "tool_calls": tool_calls}]

        # Plain message
        if msg.role == "user":
            # Flatten to string if single text block, else keep as list
            if len(msg.content) == 1 and isinstance(msg.content[0], CanonicalTextBlock):
                return [{"role": "user", "content": msg.content[0].text}]
            else:
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": b.text}
                            for b in msg.content
                            if isinstance(b, CanonicalTextBlock)
                        ],
                    }
                ]
        else:
            # Assistant text
            thinking_blocks = [b for b in msg.content if isinstance(b, CanonicalThinkingBlock)]
            parts = []
            for b in text_blocks:
                parts.append(b.text)
            content_str = " ".join(parts) if parts else None
            result: dict[str, Any] = {"role": "assistant", "content": content_str}
            if thinking_blocks:
                result["reasoning_content"] = thinking_blocks[0].text
            return [result]
