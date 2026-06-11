from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request — content blocks
# ---------------------------------------------------------------------------


class OAITextContent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["text"]
  text: str


# ---------------------------------------------------------------------------
# Request — tool / function definitions
# ---------------------------------------------------------------------------


class OAIFunctionParameters(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["object"] = "object"
  required: list[str] = []
  properties: dict[str, Any] = {}


class OAIFunctionDef(BaseModel):
  model_config = ConfigDict(extra="allow")
  name: str
  description: str = ""
  parameters: OAIFunctionParameters | None = None
  strict: bool | None = None


class OAIToolDef(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["function"]
  function: OAIFunctionDef


# ---------------------------------------------------------------------------
# Request — tool calls (in assistant messages and response deltas)
# ---------------------------------------------------------------------------


class OAIFunctionCall(BaseModel):
  model_config = ConfigDict(extra="allow")
  name: str
  arguments: str


class OAIToolCall(BaseModel):
  model_config = ConfigDict(extra="allow")
  id: str
  type: Literal["function"]
  function: OAIFunctionCall


# ---------------------------------------------------------------------------
# Request — messages
# ---------------------------------------------------------------------------


class OAISystemMessage(BaseModel):
  model_config = ConfigDict(extra="allow")
  role: Literal["system"]
  content: str


class OAIUserMessage(BaseModel):
  model_config = ConfigDict(extra="allow")
  role: Literal["user"]
  content: str | list[OAITextContent]


class OAIAssistantMessage(BaseModel):
  model_config = ConfigDict(extra="allow")
  role: Literal["assistant"]
  content: str | None = None
  reasoning_content: str | None = None
  tool_calls: list[OAIToolCall] | None = None


class OAIToolMessage(BaseModel):
  model_config = ConfigDict(extra="allow")
  role: Literal["tool"]
  content: str
  tool_call_id: str


OAIMessage = Annotated[
  OAISystemMessage | OAIUserMessage | OAIAssistantMessage | OAIToolMessage,
  Field(discriminator="role"),
]


# ---------------------------------------------------------------------------
# Request — top-level
# ---------------------------------------------------------------------------


class OAIStreamOptions(BaseModel):
  model_config = ConfigDict(extra="allow")
  include_usage: bool = False


class OAIChatTemplateKwargs(BaseModel):
  model_config = ConfigDict(extra="allow")
  enable_thinking: bool = False
  preserve_thinking: bool = False


class OAIChatCompletionRequest(BaseModel):
  model_config = ConfigDict(extra="allow")
  model: str
  messages: list[OAIMessage]
  tools: list[OAIToolDef] | None = None
  stream: bool = False
  stream_options: OAIStreamOptions | None = None
  store: bool | None = None
  max_tokens: int | None = None
  chat_template_kwargs: OAIChatTemplateKwargs | None = None


# ---------------------------------------------------------------------------
# Response — streaming chunks
# ---------------------------------------------------------------------------


class OAIDeltaFunctionCall(BaseModel):
  model_config = ConfigDict(extra="allow")
  name: str | None = None
  arguments: str | None = None


class OAIDeltaToolCall(BaseModel):
  model_config = ConfigDict(extra="allow")
  index: int
  id: str | None = None
  type: Literal["function"] | None = None
  function: OAIDeltaFunctionCall | None = None


class OAIDelta(BaseModel):
  model_config = ConfigDict(extra="allow")
  role: str | None = None
  content: str | None = None
  reasoning_content: str | None = None
  refusal: str | None = None
  tool_calls: list[OAIDeltaToolCall] | None = None
  function_call: dict[str, Any] | None = None
  audio: dict[str, Any] | None = None


class OAIChunkChoice(BaseModel):
  model_config = ConfigDict(extra="allow")
  index: int
  finish_reason: Literal["stop", "tool_calls", "length", "content_filter"] | None = None
  delta: OAIDelta
  logprobs: dict[str, Any] | None = None


class OAICompletionTokensDetails(BaseModel):
  model_config = ConfigDict(extra="allow")
  reasoning_tokens: int | None = None
  accepted_prediction_tokens: int | None = None
  audio_tokens: int | None = None
  rejected_prediction_tokens: int | None = None
  text_tokens: int | None = None
  image_tokens: int | None = None
  video_tokens: int | None = None


class OAIPromptTokensDetails(BaseModel):
  model_config = ConfigDict(extra="allow")
  cached_tokens: int | None = None
  audio_tokens: int | None = None
  text_tokens: int | None = None
  image_tokens: int | None = None
  video_tokens: int | None = None


class OAIUsage(BaseModel):
  model_config = ConfigDict(extra="allow")
  completion_tokens: int
  prompt_tokens: int
  total_tokens: int
  completion_tokens_details: OAICompletionTokensDetails | None = None
  prompt_tokens_details: OAIPromptTokensDetails | None = None


class OAIChatCompletionChunk(BaseModel):
  model_config = ConfigDict(extra="allow")
  id: str
  created: int
  model: str
  object: Literal["chat.completion.chunk"]
  system_fingerprint: str | None = None
  choices: list[OAIChunkChoice]
  usage: OAIUsage | None = None
  provider_specific_fields: dict[str, Any] | None = None
  citations: list[Any] | None = None
  service_tier: str | None = None
