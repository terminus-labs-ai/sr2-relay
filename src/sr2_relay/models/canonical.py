from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class CanonicalTextBlock(BaseModel):
  type: Literal["text"] = "text"
  text: str
  cache_control: dict | None = None


class CanonicalToolUseBlock(BaseModel):
  type: Literal["tool_use"] = "tool_use"
  id: str
  name: str
  input: dict


class CanonicalToolResultBlock(BaseModel):
  type: Literal["tool_result"] = "tool_result"
  tool_use_id: str
  content: str | list[CanonicalTextBlock]
  is_error: bool = False
  cache_control: dict | None = None


class CanonicalThinkingBlock(BaseModel):
  type: Literal["thinking"] = "thinking"
  text: str


CanonicalContentBlock = Annotated[
  CanonicalTextBlock | CanonicalToolUseBlock | CanonicalToolResultBlock | CanonicalThinkingBlock,
  Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class CanonicalMessage(BaseModel):
  role: Literal["user", "assistant"]
  content: list[CanonicalContentBlock]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class CanonicalToolDef(BaseModel):
  name: str
  description: str
  input_schema: dict


# ---------------------------------------------------------------------------
# Thinking config
# ---------------------------------------------------------------------------


class CanonicalThinkingConfig(BaseModel):
  type: Literal["adaptive", "enabled", "disabled"]


# ---------------------------------------------------------------------------
# Top-level request
# ---------------------------------------------------------------------------


class CanonicalRequest(BaseModel):
  model_config = ConfigDict(extra="allow")

  call_type: str
  model: str
  system: list[CanonicalTextBlock] | None = None
  messages: list[CanonicalMessage]
  tools: list[CanonicalToolDef] | None = None
  max_tokens: int = 16384
  temperature: float | None = None
  stream: bool = False
  thinking: CanonicalThinkingConfig | None = None
  provider_extras: dict[str, Any] = Field(default_factory=dict)
