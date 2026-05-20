from __future__ import annotations

import json

import pytest

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
from sr2_relay.translators import get_translator
from sr2_relay.translators.anthropic import AnthropicTranslator
from sr2_relay.translators.base import RequestTranslator
from sr2_relay.translators.openai import OpenAITranslator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def anth_request(**overrides) -> dict:
  """Minimal Anthropic-format request dict."""
  base = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [],
  }
  base.update(overrides)
  return base


def oai_request(**overrides) -> dict:
  """Minimal OpenAI-format request dict."""
  base = {
    "model": "gpt-4o",
    "messages": [],
  }
  base.update(overrides)
  return base


# ---------------------------------------------------------------------------
# 1–4. get_translator — registry
# ---------------------------------------------------------------------------


class TestGetTranslator:
  def test_anthropic_messages_returns_anthropic_translator(self):
    t = get_translator("anthropic_messages")
    assert isinstance(t, AnthropicTranslator)

  def test_completion_returns_openai_translator(self):
    t = get_translator("completion")
    assert isinstance(t, OpenAITranslator)

  def test_acompletion_returns_openai_translator(self):
    t = get_translator("acompletion")
    assert isinstance(t, OpenAITranslator)

  def test_embeddings_returns_none(self):
    assert get_translator("embeddings") is None

  def test_image_generation_returns_none(self):
    assert get_translator("image_generation") is None

  def test_unknown_type_returns_none(self):
    assert get_translator("rerank") is None

  def test_returned_instance_is_request_translator(self):
    t = get_translator("anthropic_messages")
    assert isinstance(t, RequestTranslator)

  def test_openai_returned_instance_is_request_translator(self):
    t = get_translator("acompletion")
    assert isinstance(t, RequestTranslator)


# ---------------------------------------------------------------------------
# 5–13. AnthropicTranslator.to_canonical
# ---------------------------------------------------------------------------


class TestAnthropicToCanonical:
  def setup_method(self):
    self.translator = AnthropicTranslator()

  def test_system_string_becomes_single_text_block(self):
    data = anth_request(system="You are helpful.")
    result = self.translator.to_canonical(data)
    assert result.system is not None
    assert len(result.system) == 1
    assert isinstance(result.system[0], CanonicalTextBlock)
    assert result.system[0].text == "You are helpful."

  def test_system_list_of_dicts_becomes_text_blocks(self):
    data = anth_request(
      system=[
        {"type": "text", "text": "Block one."},
        {"type": "text", "text": "Block two.", "cache_control": {"type": "ephemeral"}},
      ]
    )
    result = self.translator.to_canonical(data)
    assert result.system is not None
    assert len(result.system) == 2
    assert isinstance(result.system[0], CanonicalTextBlock)
    assert result.system[0].text == "Block one."
    assert isinstance(result.system[1], CanonicalTextBlock)
    assert result.system[1].text == "Block two."
    assert result.system[1].cache_control == {"type": "ephemeral"}

  def test_simple_user_message_to_canonical_message(self):
    data = anth_request(
      messages=[
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
      ]
    )
    result = self.translator.to_canonical(data)
    assert len(result.messages) == 1
    msg = result.messages[0]
    assert isinstance(msg, CanonicalMessage)
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], CanonicalTextBlock)
    assert msg.content[0].text == "Hello!"

  def test_assistant_message_with_text(self):
    data = anth_request(
      messages=[
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
      ]
    )
    result = self.translator.to_canonical(data)
    msg = result.messages[0]
    assert msg.role == "assistant"
    assert isinstance(msg.content[0], CanonicalTextBlock)
    assert msg.content[0].text == "Hi there!"

  def test_tool_use_block_in_assistant_message(self):
    data = anth_request(
      messages=[
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_use",
              "id": "tu_abc",
              "name": "search",
              "input": {"query": "python"},
            }
          ],
        }
      ]
    )
    result = self.translator.to_canonical(data)
    block = result.messages[0].content[0]
    assert isinstance(block, CanonicalToolUseBlock)
    assert block.id == "tu_abc"
    assert block.name == "search"
    assert block.input == {"query": "python"}

  def test_tool_result_block_in_user_message(self):
    data = anth_request(
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "tool_result",
              "tool_use_id": "tu_abc",
              "content": "Search results here.",
            }
          ],
        }
      ]
    )
    result = self.translator.to_canonical(data)
    block = result.messages[0].content[0]
    assert isinstance(block, CanonicalToolResultBlock)
    assert block.tool_use_id == "tu_abc"
    assert block.content == "Search results here."

  def test_provider_specific_fields_go_to_provider_extras(self):
    data = anth_request(
      metadata={"user_id": "u123"},
      output_config={"format": None},
      provider_specific_header={"custom_llm_provider": "anthropic"},
    )
    result = self.translator.to_canonical(data)
    # These should not appear as top-level CanonicalRequest fields
    assert not hasattr(result, "metadata") or result.provider_extras.get("metadata") is not None or "metadata" not in result.model_fields
    # The canonical model fields don't have metadata/output_config/provider_specific_header
    assert result.provider_extras is not None

  def test_provider_extras_contains_metadata(self):
    data = anth_request(metadata={"user_id": "u999"})
    result = self.translator.to_canonical(data)
    assert "metadata" in result.provider_extras
    assert result.provider_extras["metadata"]["user_id"] == "u999"

  def test_provider_extras_contains_output_config(self):
    data = anth_request(output_config={"format": None})
    result = self.translator.to_canonical(data)
    assert "output_config" in result.provider_extras

  def test_thinking_config_to_canonical(self):
    data = anth_request(thinking={"type": "enabled"})
    result = self.translator.to_canonical(data)
    assert result.thinking is not None
    assert isinstance(result.thinking, CanonicalThinkingConfig)
    assert result.thinking.type == "enabled"

  def test_thinking_config_adaptive(self):
    data = anth_request(thinking={"type": "adaptive"})
    result = self.translator.to_canonical(data)
    assert result.thinking.type == "adaptive"

  def test_call_type_set_on_result(self):
    data = anth_request()
    result = self.translator.to_canonical(data)
    assert result.call_type == "anthropic_messages"

  def test_model_preserved(self):
    data = anth_request(model="claude-opus-4-5")
    result = self.translator.to_canonical(data)
    assert result.model == "claude-opus-4-5"

  def test_max_tokens_preserved(self):
    data = anth_request(max_tokens=2048)
    result = self.translator.to_canonical(data)
    assert result.max_tokens == 2048

  def test_temperature_preserved(self):
    data = anth_request(temperature=0.5)
    result = self.translator.to_canonical(data)
    assert result.temperature == 0.5

  def test_stream_preserved(self):
    data = anth_request(stream=True)
    result = self.translator.to_canonical(data)
    assert result.stream is True

  def test_tools_to_canonical_tool_defs(self):
    data = anth_request(
      tools=[
        {
          "name": "read_file",
          "description": "Reads a file.",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
      ]
    )
    result = self.translator.to_canonical(data)
    assert result.tools is not None
    assert len(result.tools) == 1
    tool = result.tools[0]
    assert isinstance(tool, CanonicalToolDef)
    assert tool.name == "read_file"
    assert tool.description == "Reads a file."
    assert "properties" in tool.input_schema

  def test_no_system_becomes_none(self):
    data = anth_request()
    result = self.translator.to_canonical(data)
    assert result.system is None

  def test_returns_canonical_request(self):
    data = anth_request()
    result = self.translator.to_canonical(data)
    assert isinstance(result, CanonicalRequest)


# ---------------------------------------------------------------------------
# 14–15. AnthropicTranslator.from_canonical
# ---------------------------------------------------------------------------


class TestAnthropicFromCanonical:
  def setup_method(self):
    self.translator = AnthropicTranslator()

  def _make_canonical(self, **overrides) -> CanonicalRequest:
    base = {
      "call_type": "anthropic_messages",
      "model": "claude-sonnet-4-5",
      "max_tokens": 1024,
      "messages": [],
    }
    base.update(overrides)
    return CanonicalRequest.model_validate(base)

  def test_roundtrip_preserves_system_prompt(self):
    data = anth_request(system="You are a coding assistant.")
    canonical = self.translator.to_canonical(data)
    output = self.translator.from_canonical(canonical)
    system = output.get("system")
    # Could be list of dicts or string — check content is present
    assert system is not None
    if isinstance(system, list):
      texts = [b["text"] if isinstance(b, dict) else b.text for b in system]
      assert any("coding assistant" in t for t in texts)
    else:
      assert "coding assistant" in system

  def test_roundtrip_preserves_messages(self):
    data = anth_request(
      messages=[
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
      ]
    )
    canonical = self.translator.to_canonical(data)
    output = self.translator.from_canonical(canonical)
    msgs = output["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"

  def test_roundtrip_preserves_tools(self):
    data = anth_request(
      tools=[
        {
          "name": "calc",
          "description": "Calculator.",
          "input_schema": {"type": "object", "properties": {}},
        }
      ]
    )
    canonical = self.translator.to_canonical(data)
    output = self.translator.from_canonical(canonical)
    tools = output.get("tools")
    assert tools is not None
    assert len(tools) == 1
    tool = tools[0]
    name = tool["name"] if isinstance(tool, dict) else tool.name
    assert name == "calc"

  def test_provider_extras_unpacked_to_top_level(self):
    canonical = self._make_canonical(
      provider_extras={"metadata": {"user_id": "u42"}}
    )
    output = self.translator.from_canonical(canonical)
    assert "metadata" in output
    assert output["metadata"]["user_id"] == "u42"

  def test_from_canonical_returns_dict(self):
    canonical = self._make_canonical()
    output = self.translator.from_canonical(canonical)
    assert isinstance(output, dict)

  def test_roundtrip_thinking_config(self):
    data = anth_request(thinking={"type": "enabled"})
    canonical = self.translator.to_canonical(data)
    output = self.translator.from_canonical(canonical)
    thinking = output.get("thinking")
    assert thinking is not None
    t_type = thinking["type"] if isinstance(thinking, dict) else thinking.type
    assert t_type == "enabled"

  def test_roundtrip_tool_use_block(self):
    data = anth_request(
      messages=[
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_use",
              "id": "tu_xyz",
              "name": "search",
              "input": {"q": "sr2"},
            }
          ],
        }
      ]
    )
    canonical = self.translator.to_canonical(data)
    output = self.translator.from_canonical(canonical)
    msg = output["messages"][0]
    content = msg["content"] if isinstance(msg, dict) else msg.content
    block = content[0] if isinstance(content[0], dict) else content[0]
    if isinstance(block, dict):
      assert block["type"] == "tool_use"
      assert block["id"] == "tu_xyz"
      assert block["name"] == "search"
    else:
      assert block.type == "tool_use"
      assert block.id == "tu_xyz"


# ---------------------------------------------------------------------------
# 16–25. OpenAITranslator.to_canonical
# ---------------------------------------------------------------------------


class TestOpenAIToCanonical:
  def setup_method(self):
    self.translator = OpenAITranslator()

  def test_system_message_extracted_to_system_field(self):
    data = oai_request(
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
      ]
    )
    result = self.translator.to_canonical(data)
    assert result.system is not None
    assert len(result.system) == 1
    assert isinstance(result.system[0], CanonicalTextBlock)
    assert result.system[0].text == "You are a helpful assistant."

  def test_system_message_not_in_messages(self):
    data = oai_request(
      messages=[
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Hi."},
      ]
    )
    result = self.translator.to_canonical(data)
    for msg in result.messages:
      assert msg.role != "system"

  def test_user_string_content_becomes_text_block(self):
    data = oai_request(
      messages=[{"role": "user", "content": "Hello, world!"}]
    )
    result = self.translator.to_canonical(data)
    assert len(result.messages) == 1
    msg = result.messages[0]
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], CanonicalTextBlock)
    assert msg.content[0].text == "Hello, world!"

  def test_user_list_content_becomes_text_blocks(self):
    data = oai_request(
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
          ],
        }
      ]
    )
    result = self.translator.to_canonical(data)
    msg = result.messages[0]
    assert len(msg.content) == 2
    assert all(isinstance(b, CanonicalTextBlock) for b in msg.content)
    assert msg.content[0].text == "Part one."
    assert msg.content[1].text == "Part two."

  def test_assistant_tool_calls_become_tool_use_blocks(self):
    data = oai_request(
      messages=[
        {
          "role": "assistant",
          "content": None,
          "tool_calls": [
            {
              "id": "call_1",
              "type": "function",
              "function": {
                "name": "search",
                "arguments": '{"query": "sr2 relay"}',
              },
            }
          ],
        }
      ]
    )
    result = self.translator.to_canonical(data)
    msg = result.messages[0]
    assert msg.role == "assistant"
    block = next(b for b in msg.content if isinstance(b, CanonicalToolUseBlock))
    assert block.id == "call_1"
    assert block.name == "search"
    assert block.input == {"query": "sr2 relay"}

  def test_tool_call_arguments_parsed_from_json_string(self):
    data = oai_request(
      messages=[
        {
          "role": "assistant",
          "tool_calls": [
            {
              "id": "call_2",
              "type": "function",
              "function": {
                "name": "calc",
                "arguments": '{"expr": "2+2", "precision": 4}',
              },
            }
          ],
        }
      ]
    )
    result = self.translator.to_canonical(data)
    block = result.messages[0].content[0]
    assert isinstance(block, CanonicalToolUseBlock)
    assert isinstance(block.input, dict)
    assert block.input["expr"] == "2+2"
    assert block.input["precision"] == 4

  def test_tool_messages_merged_into_user_message_with_tool_result_blocks(self):
    data = oai_request(
      messages=[
        {
          "role": "assistant",
          "tool_calls": [
            {
              "id": "call_1",
              "type": "function",
              "function": {"name": "search", "arguments": "{}"},
            }
          ],
        },
        {
          "role": "tool",
          "tool_call_id": "call_1",
          "content": "Search results here.",
        },
      ]
    )
    result = self.translator.to_canonical(data)
    # Last message should be a user message containing a CanonicalToolResultBlock
    tool_msg = result.messages[-1]
    assert tool_msg.role == "user"
    assert any(isinstance(b, CanonicalToolResultBlock) for b in tool_msg.content)
    block = next(b for b in tool_msg.content if isinstance(b, CanonicalToolResultBlock))
    assert block.tool_use_id == "call_1"
    assert block.content == "Search results here."

  def test_multiple_consecutive_tool_messages_merged_into_one(self):
    data = oai_request(
      messages=[
        {
          "role": "assistant",
          "tool_calls": [
            {
              "id": "call_1",
              "type": "function",
              "function": {"name": "search", "arguments": "{}"},
            },
            {
              "id": "call_2",
              "type": "function",
              "function": {"name": "calc", "arguments": "{}"},
            },
          ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Result A."},
        {"role": "tool", "tool_call_id": "call_2", "content": "Result B."},
      ]
    )
    result = self.translator.to_canonical(data)
    # Should have exactly 2 messages: assistant + merged user
    assert len(result.messages) == 2
    tool_msg = result.messages[-1]
    assert tool_msg.role == "user"
    tool_result_blocks = [b for b in tool_msg.content if isinstance(b, CanonicalToolResultBlock)]
    assert len(tool_result_blocks) == 2
    ids = {b.tool_use_id for b in tool_result_blocks}
    assert ids == {"call_1", "call_2"}

  def test_max_tokens_absent_defaults_to_16384(self):
    data = oai_request(messages=[{"role": "user", "content": "Hi."}])
    result = self.translator.to_canonical(data)
    assert result.max_tokens == 16384

  def test_max_tokens_absent_sets_provider_extras_flag(self):
    data = oai_request(messages=[{"role": "user", "content": "Hi."}])
    result = self.translator.to_canonical(data)
    assert result.provider_extras.get("max_tokens_was_default") is True

  def test_max_tokens_present_used_as_is(self):
    data = oai_request(max_tokens=512, messages=[{"role": "user", "content": "Hi."}])
    result = self.translator.to_canonical(data)
    assert result.max_tokens == 512

  def test_max_tokens_present_no_default_flag(self):
    data = oai_request(max_tokens=512, messages=[{"role": "user", "content": "Hi."}])
    result = self.translator.to_canonical(data)
    assert "max_tokens_was_default" not in result.provider_extras

  def test_tools_array_becomes_canonical_tool_defs(self):
    data = oai_request(
      tools=[
        {
          "type": "function",
          "function": {
            "name": "read_file",
            "description": "Reads a file.",
            "parameters": {
              "type": "object",
              "properties": {"path": {"type": "string"}},
              "required": ["path"],
            },
          },
        }
      ]
    )
    result = self.translator.to_canonical(data)
    assert result.tools is not None
    assert len(result.tools) == 1
    tool = result.tools[0]
    assert isinstance(tool, CanonicalToolDef)
    assert tool.name == "read_file"
    assert tool.description == "Reads a file."
    assert tool.input_schema["type"] == "object"
    assert "path" in tool.input_schema["properties"]

  def test_tools_function_parameters_becomes_input_schema(self):
    data = oai_request(
      tools=[
        {
          "type": "function",
          "function": {
            "name": "fn",
            "description": "",
            "parameters": {"type": "object", "required": ["x"], "properties": {"x": {"type": "int"}}},
          },
        }
      ]
    )
    result = self.translator.to_canonical(data)
    assert result.tools[0].input_schema == {
      "type": "object",
      "required": ["x"],
      "properties": {"x": {"type": "int"}},
    }

  def test_call_type_set_to_input_call_type_acompletion(self):
    data = oai_request(messages=[])
    result = self.translator.to_canonical(data, call_type="acompletion")
    assert result.call_type == "acompletion"

  def test_call_type_set_to_input_call_type_completion(self):
    data = oai_request(messages=[])
    result = self.translator.to_canonical(data, call_type="completion")
    assert result.call_type == "completion"

  def test_assistant_reasoning_content_becomes_thinking_block(self):
    data = oai_request(
      messages=[
        {
          "role": "assistant",
          "content": "My answer.",
          "reasoning_content": "Let me think about this.",
        }
      ]
    )
    result = self.translator.to_canonical(data)
    msg = result.messages[0]
    thinking_blocks = [b for b in msg.content if isinstance(b, CanonicalThinkingBlock)]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].text == "Let me think about this."

  def test_stream_options_goes_to_provider_extras(self):
    data = oai_request(
      stream_options={"include_usage": True},
      messages=[{"role": "user", "content": "Hi."}],
    )
    result = self.translator.to_canonical(data)
    assert "stream_options" in result.provider_extras

  def test_store_goes_to_provider_extras(self):
    data = oai_request(
      store=True,
      messages=[{"role": "user", "content": "Hi."}],
    )
    result = self.translator.to_canonical(data)
    assert "store" in result.provider_extras

  def test_chat_template_kwargs_goes_to_provider_extras(self):
    data = oai_request(
      chat_template_kwargs={"enable_thinking": True, "preserve_thinking": False},
      messages=[{"role": "user", "content": "Hi."}],
    )
    result = self.translator.to_canonical(data)
    assert "chat_template_kwargs" in result.provider_extras

  def test_enable_thinking_in_chat_template_kwargs_sets_thinking_config(self):
    data = oai_request(
      chat_template_kwargs={"enable_thinking": True},
      messages=[{"role": "user", "content": "Hi."}],
    )
    result = self.translator.to_canonical(data)
    assert result.thinking is not None
    assert result.thinking.type == "enabled"

  def test_returns_canonical_request(self):
    data = oai_request(messages=[{"role": "user", "content": "Hi."}])
    result = self.translator.to_canonical(data)
    assert isinstance(result, CanonicalRequest)


# ---------------------------------------------------------------------------
# 26–30. OpenAITranslator.from_canonical
# ---------------------------------------------------------------------------


class TestOpenAIFromCanonical:
  def setup_method(self):
    self.translator = OpenAITranslator()

  def _make_canonical(self, **overrides) -> CanonicalRequest:
    base = {
      "call_type": "acompletion",
      "model": "gpt-4o",
      "max_tokens": 1024,
      "messages": [],
    }
    base.update(overrides)
    return CanonicalRequest.model_validate(base)

  def test_system_blocks_prepended_as_system_role_message(self):
    canonical = self._make_canonical(
      system=[{"type": "text", "text": "You are helpful."}],
      messages=[{"role": "user", "content": [{"type": "text", "text": "Hi."}]}],
    )
    output = self.translator.from_canonical(canonical)
    messages = output["messages"]
    assert messages[0]["role"] == "system"
    content = messages[0]["content"]
    assert "helpful" in content

  def test_system_blocks_not_in_remaining_messages(self):
    canonical = self._make_canonical(
      system=[{"type": "text", "text": "System."}],
      messages=[{"role": "user", "content": [{"type": "text", "text": "Hi."}]}],
    )
    output = self.translator.from_canonical(canonical)
    non_system = [m for m in output["messages"] if m["role"] != "system"]
    assert all(m["role"] != "system" for m in non_system)

  def test_tool_result_blocks_become_individual_tool_messages(self):
    canonical = self._make_canonical(
      messages=[
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_use",
              "id": "call_1",
              "name": "search",
              "input": {"q": "test"},
            }
          ],
        },
        {
          "role": "user",
          "content": [
            {
              "type": "tool_result",
              "tool_use_id": "call_1",
              "content": "Results here.",
            }
          ],
        },
      ]
    )
    output = self.translator.from_canonical(canonical)
    messages = output["messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert tool_messages[0]["content"] == "Results here."

  def test_multiple_tool_result_blocks_become_individual_tool_messages(self):
    canonical = self._make_canonical(
      messages=[
        {
          "role": "assistant",
          "content": [
            {"type": "tool_use", "id": "call_1", "name": "fn1", "input": {}},
            {"type": "tool_use", "id": "call_2", "name": "fn2", "input": {}},
          ],
        },
        {
          "role": "user",
          "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "A."},
            {"type": "tool_result", "tool_use_id": "call_2", "content": "B."},
          ],
        },
      ]
    )
    output = self.translator.from_canonical(canonical)
    tool_messages = [m for m in output["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2
    ids = {m["tool_call_id"] for m in tool_messages}
    assert ids == {"call_1", "call_2"}

  def test_tool_use_block_input_dict_becomes_json_string_arguments(self):
    canonical = self._make_canonical(
      messages=[
        {
          "role": "assistant",
          "content": [
            {
              "type": "tool_use",
              "id": "call_1",
              "name": "search",
              "input": {"query": "hello", "top_k": 5},
            }
          ],
        }
      ]
    )
    output = self.translator.from_canonical(canonical)
    assistant_msg = next(m for m in output["messages"] if m["role"] == "assistant")
    tool_calls = assistant_msg["tool_calls"]
    assert len(tool_calls) == 1
    arguments = tool_calls[0]["function"]["arguments"]
    assert isinstance(arguments, str)
    parsed = json.loads(arguments)
    assert parsed["query"] == "hello"
    assert parsed["top_k"] == 5

  def test_max_tokens_was_default_omits_max_tokens(self):
    canonical = self._make_canonical(
      provider_extras={"max_tokens_was_default": True}
    )
    output = self.translator.from_canonical(canonical)
    assert "max_tokens" not in output

  def test_max_tokens_not_default_includes_max_tokens(self):
    canonical = self._make_canonical(max_tokens=512)
    output = self.translator.from_canonical(canonical)
    assert output.get("max_tokens") == 512

  def test_roundtrip_preserves_model(self):
    data = oai_request(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": "Hello!"}],
    )
    canonical = self.translator.to_canonical(data, call_type="acompletion")
    output = self.translator.from_canonical(canonical)
    assert output["model"] == "gpt-4o-mini"

  def test_roundtrip_preserves_user_message_text(self):
    data = oai_request(messages=[{"role": "user", "content": "Hello, world!"}])
    canonical = self.translator.to_canonical(data, call_type="acompletion")
    output = self.translator.from_canonical(canonical)
    user_msgs = [m for m in output["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1
    content = user_msgs[0]["content"]
    if isinstance(content, str):
      assert "Hello, world!" in content
    else:
      texts = [b["text"] if isinstance(b, dict) else b.text for b in content]
      assert any("Hello, world!" in t for t in texts)

  def test_roundtrip_preserves_stream_flag(self):
    data = oai_request(
      stream=True,
      messages=[{"role": "user", "content": "Hi."}],
    )
    canonical = self.translator.to_canonical(data, call_type="acompletion")
    output = self.translator.from_canonical(canonical)
    assert output.get("stream") is True

  def test_from_canonical_returns_dict(self):
    canonical = self._make_canonical(
      messages=[{"role": "user", "content": [{"type": "text", "text": "Hi."}]}]
    )
    output = self.translator.from_canonical(canonical)
    assert isinstance(output, dict)
