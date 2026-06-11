from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request — content blocks
# ---------------------------------------------------------------------------


class AnthCacheControl(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["ephemeral"]


class AnthTextBlock(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["text"]
  text: str
  cache_control: AnthCacheControl | None = None


class AnthToolUseBlock(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["tool_use"]
  id: str
  name: str
  input: dict[str, Any]


class AnthToolResultBlock(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["tool_result"]
  tool_use_id: str
  content: str
  is_error: bool = False
  cache_control: AnthCacheControl | None = None


AnthContentBlock = Annotated[
  AnthTextBlock | AnthToolUseBlock | AnthToolResultBlock,
  Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Request — messages
# ---------------------------------------------------------------------------


class AnthMessage(BaseModel):
  model_config = ConfigDict(extra="allow")
  role: Literal["user", "assistant"]
  content: list[AnthContentBlock]


# ---------------------------------------------------------------------------
# Request — tool definitions
# ---------------------------------------------------------------------------


class AnthToolDef(BaseModel):
  model_config = ConfigDict(extra="allow")
  name: str
  description: str = ""
  input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# Request — config sub-models
# ---------------------------------------------------------------------------


class AnthMetadata(BaseModel):
  model_config = ConfigDict(extra="allow")
  user_id: str | None = None


class AnthJsonSchema(BaseModel):
  model_config = ConfigDict(extra="allow", populate_by_name=True)
  type: Literal["json_schema"]
  schema_: dict[str, Any] = Field(alias="schema")


class AnthOutputFormat(BaseModel):
  model_config = ConfigDict(extra="allow")
  format: AnthJsonSchema | None = None


class AnthThinkingConfig(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["adaptive", "enabled", "disabled"]


class AnthProviderSpecificHeader(BaseModel):
  model_config = ConfigDict(extra="allow")
  custom_llm_provider: str | None = None
  extra_headers: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Request — top-level
# ---------------------------------------------------------------------------


class AnthMessagesRequest(BaseModel):
  model_config = ConfigDict(extra="allow")
  model: str
  messages: list[AnthMessage]
  system: list[AnthTextBlock] | None = None
  tools: list[AnthToolDef] | None = None
  max_tokens: int
  temperature: float | None = None
  stream: bool = False
  metadata: AnthMetadata | None = None
  output_config: AnthOutputFormat | None = None
  thinking: AnthThinkingConfig | None = None
  provider_specific_header: AnthProviderSpecificHeader | None = None


# ---------------------------------------------------------------------------
# Response — usage
# ---------------------------------------------------------------------------


class AnthCacheCreationDetails(BaseModel):
  model_config = ConfigDict(extra="allow")
  ephemeral_5m_input_tokens: int = 0
  ephemeral_1h_input_tokens: int = 0


class AnthStartUsage(BaseModel):
  model_config = ConfigDict(extra="allow")
  input_tokens: int
  output_tokens: int
  cache_creation_input_tokens: int = 0
  cache_read_input_tokens: int = 0
  cache_creation: AnthCacheCreationDetails | None = None


class AnthDeltaUsage(BaseModel):
  model_config = ConfigDict(extra="allow")
  input_tokens: int | None = None
  output_tokens: int | None = None
  cache_creation_input_tokens: int | None = None
  cache_read_input_tokens: int | None = None


# ---------------------------------------------------------------------------
# Response — message_start
# ---------------------------------------------------------------------------


class AnthStartMessage(BaseModel):
  model_config = ConfigDict(extra="allow")
  id: str
  type: Literal["message"]
  role: Literal["assistant"]
  model: str
  content: list[Any]
  stop_reason: str | None = None
  stop_sequence: str | None = None
  stop_details: dict[str, Any] | None = None
  usage: AnthStartUsage


class AnthMessageStartEvent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["message_start"]
  message: AnthStartMessage


# ---------------------------------------------------------------------------
# Response — content_block_start
# ---------------------------------------------------------------------------


class AnthTextBlockStart(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["text"]
  text: str = ""


class AnthToolUseBlockStart(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["tool_use"]
  id: str
  name: str
  input: dict[str, Any] = {}


class AnthThinkingBlockStart(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["thinking"]
  text: str = ""


AnthResponseContentBlock = Annotated[
  AnthTextBlockStart | AnthToolUseBlockStart | AnthThinkingBlockStart,
  Field(discriminator="type"),
]


class AnthContentBlockStartEvent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["content_block_start"]
  index: int
  content_block: AnthResponseContentBlock


# ---------------------------------------------------------------------------
# Response — content_block_delta
# ---------------------------------------------------------------------------


class AnthTextDelta(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["text_delta"]
  text: str


class AnthThinkingDelta(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["thinking_delta"]
  thinking: str


class AnthInputJsonDelta(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["input_json_delta"]
  partial_json: str


AnthDelta = Annotated[
  AnthTextDelta | AnthThinkingDelta | AnthInputJsonDelta,
  Field(discriminator="type"),
]


class AnthContentBlockDeltaEvent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["content_block_delta"]
  index: int
  delta: AnthDelta


# ---------------------------------------------------------------------------
# Response — content_block_stop, message_delta, message_stop, ping
# ---------------------------------------------------------------------------


class AnthContentBlockStopEvent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["content_block_stop"]
  index: int


class AnthMessageDeltaBody(BaseModel):
  model_config = ConfigDict(extra="allow")
  stop_reason: str | None = None
  stop_sequence: str | None = None
  stop_details: dict[str, Any] | None = None


class AnthMessageDeltaEvent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["message_delta"]
  delta: AnthMessageDeltaBody
  usage: AnthDeltaUsage


class AnthMessageStopEvent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["message_stop"]


class AnthPingEvent(BaseModel):
  model_config = ConfigDict(extra="allow")
  type: Literal["ping"]


# ---------------------------------------------------------------------------
# Response — discriminated union of all SSE event types
# ---------------------------------------------------------------------------

AnthStreamEvent = Annotated[
  AnthMessageStartEvent
  | AnthContentBlockStartEvent
  | AnthContentBlockDeltaEvent
  | AnthContentBlockStopEvent
  | AnthMessageDeltaEvent
  | AnthMessageStopEvent
  | AnthPingEvent,
  Field(discriminator="type"),
]
