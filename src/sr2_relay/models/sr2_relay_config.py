from typing import Optional

from pydantic import BaseModel, ConfigDict


class ModelSlotConfig(BaseModel):
  model: str | None = None
  api_key: str | None = None
  api_base: str | None = None
  api_version: str | None = None
  model_config = ConfigDict(extra="allow")


class SR2RelayConfig(BaseModel):
  timeout: int = 60
  host: str = "127.0.0.1"
  port: int = 8340

  api_base: str
  api_key: str = ""

  model: Optional[ModelSlotConfig] = None
  fast_model: Optional[ModelSlotConfig] = None
  embedding_model: Optional[ModelSlotConfig] = None
