import logging
import os
from pathlib import Path
from typing import Any

import yaml

from sr2_relay.models import SR2RelayConfig

logger = logging.getLogger(__name__)

_config: SR2RelayConfig | None = None


def _resolve_env_vars(data: dict, env_base_prefix="SR2_RELAY"):
  if isinstance(data, dict):
    return {k: _resolve_env_vars(v) for k, v in data.items()}
  if isinstance(data, list):
    return [_resolve_env_vars(item) for item in data]
  if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
    var_name = data[2:-1]
    return os.environ[var_name]  # fail fast if missing
  return data


def build_litellm_config(config: SR2RelayConfig) -> dict:
  litellm_config: dict[str, Any] = {
    "router_settings": {"pass_through_all_models": True}
  }

  configured_models = []
  if config.model:
    configured_models.append(config.model)
  if config.fast_model:
    configured_models.append(config.fast_model)
  if config.embedding_model:
    configured_models.append(config.embedding_model)

  if len(configured_models) > 0:
    litellm_config["model_list"] = list(
      map(
        lambda model: {
          "model_name": model.model.split("/")[1]
          if "/" in model.model
          else model.model,
          "litellm_params": {
            k: v for k, v in model.model_dump().items() if v is not None
          },
        },
        configured_models,
      )
    )

  return litellm_config


def load_config(path: Path | None = None) -> SR2RelayConfig:
  """Load and merge config from YAML file + CLI overrides."""
  global _config
  if _config is None:
    file_content = Path(path or "config/sr2-relay-config.yaml").read_text()
    raw = _resolve_env_vars(yaml.safe_load(file_content) or {})

    _config = SR2RelayConfig(**raw)

  return _config
