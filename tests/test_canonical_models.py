from __future__ import annotations

import pytest
from pydantic import ValidationError

from sr2_relay.models.canonical import (
  CanonicalContentBlock,
  CanonicalMessage,
  CanonicalRequest,
  CanonicalTextBlock,
  CanonicalThinkingBlock,
  CanonicalThinkingConfig,
  CanonicalToolDef,
  CanonicalToolResultBlock,
  CanonicalToolUseBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def minimal_request(**overrides) -> dict:
  base = {
    "call_type": "acompletion",
    "model": "gpt-4o",
    "messages": [],
    "max_tokens": 1024,
  }
  base.update(overrides)
  return base


# ---------------------------------------------------------------------------
# 1. Default values
# ---------------------------------------------------------------------------


class TestDefaults:
  def test_max_tokens_defaults_to_16384(self):
    req = CanonicalRequest.model_validate(
      {"call_type": "acompletion", "model": "gpt-4o", "messages": []}
    )
    assert req.max_tokens == 16384

  def test_stream_defaults_to_false(self):
    req = CanonicalRequest.model_validate(minimal_request())
    assert req.stream is False

  def test_temperature_defaults_to_none(self):
    req = CanonicalRequest.model_validate(minimal_request())
    assert req.temperature is None

  def test_system_defaults_to_none(self):
    req = CanonicalRequest.model_validate(minimal_request())
    assert req.system is None

  def test_tools_defaults_to_none(self):
    req = CanonicalRequest.model_validate(minimal_request())
    assert req.tools is None

  def test_thinking_defaults_to_none(self):
    req = CanonicalRequest.model_validate(minimal_request())
    assert req.thinking is None

  def test_provider_extras_defaults_to_empty_dict(self):
    req = CanonicalRequest.model_validate(minimal_request())
    assert req.provider_extras == {}


# ---------------------------------------------------------------------------
# 2. Discriminated union — each block type parses correctly
# ---------------------------------------------------------------------------


class TestDiscriminatedUnion:
  def test_text_block_from_dict(self):
    data = {"type": "text", "text": "hello"}
    block = CanonicalTextBlock.model_validate(data)
    assert block.type == "text"
    assert block.text == "hello"

  def test_tool_use_block_from_dict(self):
    data = {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "test"}}
    block = CanonicalToolUseBlock.model_validate(data)
    assert block.type == "tool_use"
    assert block.id == "tu_1"
    assert block.name == "search"
    assert block.input == {"q": "test"}

  def test_tool_result_block_from_dict(self):
    data = {"type": "tool_result", "tool_use_id": "tu_1", "content": "found it"}
    block = CanonicalToolResultBlock.model_validate(data)
    assert block.type == "tool_result"
    assert block.tool_use_id == "tu_1"
    assert block.content == "found it"

  def test_thinking_block_from_dict(self):
    data = {"type": "thinking", "text": "let me reason..."}
    block = CanonicalThinkingBlock.model_validate(data)
    assert block.type == "thinking"
    assert block.text == "let me reason..."

  def test_canonical_content_block_routes_text(self):
    from pydantic import TypeAdapter
    adapter = TypeAdapter(CanonicalContentBlock)
    block = adapter.validate_python({"type": "text", "text": "hi"})
    assert isinstance(block, CanonicalTextBlock)

  def test_canonical_content_block_routes_tool_use(self):
    from pydantic import TypeAdapter
    adapter = TypeAdapter(CanonicalContentBlock)
    block = adapter.validate_python(
      {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}
    )
    assert isinstance(block, CanonicalToolUseBlock)

  def test_canonical_content_block_routes_tool_result(self):
    from pydantic import TypeAdapter
    adapter = TypeAdapter(CanonicalContentBlock)
    block = adapter.validate_python(
      {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}
    )
    assert isinstance(block, CanonicalToolResultBlock)

  def test_canonical_content_block_routes_thinking(self):
    from pydantic import TypeAdapter
    adapter = TypeAdapter(CanonicalContentBlock)
    block = adapter.validate_python({"type": "thinking", "text": "hmm"})
    assert isinstance(block, CanonicalThinkingBlock)


# ---------------------------------------------------------------------------
# 3. Invalid discriminator raises ValidationError
# ---------------------------------------------------------------------------


class TestInvalidDiscriminator:
  def test_unknown_type_raises(self):
    from pydantic import TypeAdapter
    adapter = TypeAdapter(CanonicalContentBlock)
    with pytest.raises(ValidationError):
      adapter.validate_python({"type": "image_url", "url": "http://example.com/img.png"})

  def test_missing_type_raises(self):
    from pydantic import TypeAdapter
    adapter = TypeAdapter(CanonicalContentBlock)
    with pytest.raises(ValidationError):
      adapter.validate_python({"text": "no type field here"})


# ---------------------------------------------------------------------------
# 4. extra="allow" — LiteLLM-injected keys pass through
# ---------------------------------------------------------------------------


class TestExtraAllow:
  def test_unknown_top_level_key_is_accepted(self):
    data = minimal_request(litellm_call_id="abc-123")
    req = CanonicalRequest.model_validate(data)
    # Field is accessible (extra="allow" stores it on the model)
    assert req.litellm_call_id == "abc-123"  # type: ignore[attr-defined]

  def test_multiple_litellm_keys_accepted(self):
    data = minimal_request(
      litellm_call_id="id1",
      litellm_session_id="sess1",
      litellm_trace_id="trace1",
      secret_fields=["api_key"],
    )
    req = CanonicalRequest.model_validate(data)
    assert req.litellm_call_id == "id1"  # type: ignore[attr-defined]
    assert req.litellm_session_id == "sess1"  # type: ignore[attr-defined]

  def test_known_fields_still_validated(self):
    with pytest.raises(ValidationError):
      CanonicalRequest.model_validate(
        {"call_type": "x", "model": "x", "messages": [], "max_tokens": "not-an-int"}
      )


# ---------------------------------------------------------------------------
# 5. System blocks — list of CanonicalTextBlock
# ---------------------------------------------------------------------------


class TestSystemBlocks:
  def test_system_with_plain_text_block(self):
    data = minimal_request(
      system=[{"type": "text", "text": "You are a helpful assistant."}]
    )
    req = CanonicalRequest.model_validate(data)
    assert req.system is not None
    assert len(req.system) == 1
    assert req.system[0].text == "You are a helpful assistant."
    assert req.system[0].cache_control is None

  def test_system_with_cache_control(self):
    data = minimal_request(
      system=[
        {
          "type": "text",
          "text": "Cached system prompt.",
          "cache_control": {"type": "ephemeral"},
        }
      ]
    )
    req = CanonicalRequest.model_validate(data)
    assert req.system[0].cache_control == {"type": "ephemeral"}

  def test_system_multiple_blocks(self):
    data = minimal_request(
      system=[
        {"type": "text", "text": "Block one."},
        {"type": "text", "text": "Block two.", "cache_control": {"type": "ephemeral"}},
      ]
    )
    req = CanonicalRequest.model_validate(data)
    assert len(req.system) == 2
    assert req.system[1].cache_control is not None


# ---------------------------------------------------------------------------
# 6. Tool use blocks
# ---------------------------------------------------------------------------


class TestToolUseBlocks:
  def test_tool_use_block_input_is_dict(self):
    block = CanonicalToolUseBlock.model_validate(
      {
        "type": "tool_use",
        "id": "tu_abc",
        "name": "code_interpreter",
        "input": {"code": "print('hi')", "language": "python"},
      }
    )
    assert block.input == {"code": "print('hi')", "language": "python"}

  def test_tool_use_block_empty_input(self):
    block = CanonicalToolUseBlock.model_validate(
      {"type": "tool_use", "id": "tu_x", "name": "no_args", "input": {}}
    )
    assert block.input == {}

  def test_tool_use_in_message_content(self):
    msg = CanonicalMessage.model_validate(
      {
        "role": "assistant",
        "content": [
          {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "test"}}
        ],
      }
    )
    assert isinstance(msg.content[0], CanonicalToolUseBlock)


# ---------------------------------------------------------------------------
# 7. Tool result blocks
# ---------------------------------------------------------------------------


class TestToolResultBlocks:
  def test_is_error_defaults_to_false(self):
    block = CanonicalToolResultBlock.model_validate(
      {"type": "tool_result", "tool_use_id": "tu_1", "content": "result text"}
    )
    assert block.is_error is False

  def test_is_error_explicit_true(self):
    block = CanonicalToolResultBlock.model_validate(
      {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "error occurred",
        "is_error": True,
      }
    )
    assert block.is_error is True

  def test_cache_control_on_tool_result(self):
    block = CanonicalToolResultBlock.model_validate(
      {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "data",
        "cache_control": {"type": "ephemeral"},
      }
    )
    assert block.cache_control == {"type": "ephemeral"}

  def test_cache_control_defaults_to_none(self):
    block = CanonicalToolResultBlock.model_validate(
      {"type": "tool_result", "tool_use_id": "tu_1", "content": "data"}
    )
    assert block.cache_control is None


# ---------------------------------------------------------------------------
# 8. Thinking config — valid and invalid types
# ---------------------------------------------------------------------------


class TestThinkingConfig:
  @pytest.mark.parametrize("thinking_type", ["adaptive", "enabled", "disabled"])
  def test_valid_thinking_types(self, thinking_type: str):
    config = CanonicalThinkingConfig.model_validate({"type": thinking_type})
    assert config.type == thinking_type

  def test_invalid_thinking_type_raises(self):
    with pytest.raises(ValidationError):
      CanonicalThinkingConfig.model_validate({"type": "auto"})

  def test_thinking_on_request(self):
    data = minimal_request(thinking={"type": "enabled"})
    req = CanonicalRequest.model_validate(data)
    assert req.thinking is not None
    assert req.thinking.type == "enabled"


# ---------------------------------------------------------------------------
# 9. Roundtrip — full request serialize and reconstruct
# ---------------------------------------------------------------------------


class TestRoundtrip:
  def test_full_roundtrip(self):
    data = {
      "call_type": "anthropic_messages",
      "model": "claude-sonnet-4-5",
      "system": [
        {"type": "text", "text": "System prompt.", "cache_control": {"type": "ephemeral"}}
      ],
      "messages": [
        {
          "role": "user",
          "content": [{"type": "text", "text": "Hello, assistant."}],
        },
        {
          "role": "assistant",
          "content": [
            {"type": "thinking", "text": "The user greeted me."},
            {"type": "text", "text": "Hello!"},
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "hi"}},
          ],
        },
        {
          "role": "user",
          "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "results here"}
          ],
        },
      ],
      "tools": [
        {
          "name": "search",
          "description": "Search the web.",
          "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
      ],
      "max_tokens": 2048,
      "temperature": 0.7,
      "stream": True,
      "thinking": {"type": "enabled"},
      "provider_extras": {"litellm_call_id": "xyz"},
    }

    original = CanonicalRequest.model_validate(data)
    dumped = original.model_dump()
    reconstructed = CanonicalRequest.model_validate(dumped)

    assert reconstructed == original
    assert reconstructed.model == "claude-sonnet-4-5"
    assert reconstructed.max_tokens == 2048
    assert reconstructed.temperature == 0.7
    assert reconstructed.stream is True
    assert reconstructed.thinking.type == "enabled"
    assert reconstructed.provider_extras == {"litellm_call_id": "xyz"}
    assert reconstructed.system[0].text == "System prompt."
    assert len(reconstructed.messages) == 3


# ---------------------------------------------------------------------------
# 10. Role validation — only "user" and "assistant" are valid
# ---------------------------------------------------------------------------


class TestRoleValidation:
  def test_user_role_valid(self):
    msg = CanonicalMessage.model_validate(
      {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    )
    assert msg.role == "user"

  def test_assistant_role_valid(self):
    msg = CanonicalMessage.model_validate(
      {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
    )
    assert msg.role == "assistant"

  def test_system_role_raises(self):
    with pytest.raises(ValidationError):
      CanonicalMessage.model_validate(
        {"role": "system", "content": [{"type": "text", "text": "system"}]}
      )

  def test_tool_role_raises(self):
    with pytest.raises(ValidationError):
      CanonicalMessage.model_validate(
        {"role": "tool", "content": [{"type": "text", "text": "tool output"}]}
      )

  def test_empty_string_role_raises(self):
    with pytest.raises(ValidationError):
      CanonicalMessage.model_validate(
        {"role": "", "content": [{"type": "text", "text": "hi"}]}
      )


# ---------------------------------------------------------------------------
# 11. Mixed content — multiple block types in one message
# ---------------------------------------------------------------------------


class TestMixedContent:
  def test_text_and_tool_use_in_assistant_message(self):
    msg = CanonicalMessage.model_validate(
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "I will search for that."},
          {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "sr2"}},
        ],
      }
    )
    assert isinstance(msg.content[0], CanonicalTextBlock)
    assert isinstance(msg.content[1], CanonicalToolUseBlock)

  def test_text_and_tool_result_in_user_message(self):
    msg = CanonicalMessage.model_validate(
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Here are the results:"},
          {"type": "tool_result", "tool_use_id": "tu_1", "content": "data"},
        ],
      }
    )
    assert isinstance(msg.content[0], CanonicalTextBlock)
    assert isinstance(msg.content[1], CanonicalToolResultBlock)

  def test_thinking_text_and_tool_use_in_assistant_message(self):
    msg = CanonicalMessage.model_validate(
      {
        "role": "assistant",
        "content": [
          {"type": "thinking", "text": "Let me reason about this."},
          {"type": "text", "text": "Based on my reasoning..."},
          {"type": "tool_use", "id": "tu_2", "name": "calc", "input": {"expr": "2+2"}},
        ],
      }
    )
    assert isinstance(msg.content[0], CanonicalThinkingBlock)
    assert isinstance(msg.content[1], CanonicalTextBlock)
    assert isinstance(msg.content[2], CanonicalToolUseBlock)

  def test_multiple_tool_results_in_user_message(self):
    msg = CanonicalMessage.model_validate(
      {
        "role": "user",
        "content": [
          {"type": "tool_result", "tool_use_id": "tu_1", "content": "result A"},
          {
            "type": "tool_result",
            "tool_use_id": "tu_2",
            "content": "error B",
            "is_error": True,
          },
        ],
      }
    )
    assert isinstance(msg.content[0], CanonicalToolResultBlock)
    assert isinstance(msg.content[1], CanonicalToolResultBlock)
    assert msg.content[1].is_error is True

  def test_tool_def_fields(self):
    tool = CanonicalToolDef.model_validate(
      {
        "name": "read_file",
        "description": "Reads a file from disk.",
        "input_schema": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"],
        },
      }
    )
    assert tool.name == "read_file"
    assert tool.description == "Reads a file from disk."
    assert tool.input_schema["required"] == ["path"]


# ---------------------------------------------------------------------------
# 12. Required fields — missing fields raise ValidationError
# ---------------------------------------------------------------------------


class TestRequiredFields:
  def test_missing_model_raises(self):
    with pytest.raises(ValidationError):
      CanonicalRequest.model_validate({"call_type": "acompletion", "messages": []})

  def test_missing_call_type_raises(self):
    with pytest.raises(ValidationError):
      CanonicalRequest.model_validate({"model": "gpt-4o", "messages": []})

  def test_missing_messages_raises(self):
    with pytest.raises(ValidationError):
      CanonicalRequest.model_validate({"call_type": "acompletion", "model": "gpt-4o"})


# ---------------------------------------------------------------------------
# 13. CanonicalMessage.content as bare string raises
# ---------------------------------------------------------------------------


class TestMessageContentValidation:
  def test_content_as_bare_string_raises(self):
    with pytest.raises(ValidationError):
      CanonicalMessage.model_validate({"role": "user", "content": "hello"})

  def test_empty_content_list_is_valid(self):
    # No min_length constraint — empty list must be accepted
    msg = CanonicalMessage.model_validate({"role": "user", "content": []})
    assert msg.content == []


# ---------------------------------------------------------------------------
# 14. CanonicalToolResultBlock with list[CanonicalTextBlock] content
# ---------------------------------------------------------------------------


class TestToolResultBlockListContent:
  def test_content_as_list_of_text_blocks_parses(self):
    block = CanonicalToolResultBlock.model_validate(
      {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": [{"type": "text", "text": "structured result"}],
      }
    )
    assert isinstance(block.content, list)
    assert len(block.content) == 1

  def test_content_list_contains_text_block_with_correct_text(self):
    block = CanonicalToolResultBlock.model_validate(
      {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": [{"type": "text", "text": "structured result"}],
      }
    )
    assert isinstance(block.content[0], CanonicalTextBlock)
    assert block.content[0].text == "structured result"
