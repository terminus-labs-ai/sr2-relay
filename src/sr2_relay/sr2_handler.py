import json
import logging
from typing import Any, AsyncGenerator, Literal

from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import LLMResponseTypes, ModelResponseStream

logger = logging.getLogger(__name__)

LOG_PATH = ".log/hooks.jsonl"


def _safe_serialize(obj: Any, _seen: set | None = None) -> Any:
  if _seen is None:
    _seen = set()
  obj_id = id(obj)
  if obj_id in _seen:
    return "<circular>"
  if isinstance(obj, (str, int, float, bool, type(None))):
    return obj
  _seen.add(obj_id)
  try:
    if isinstance(obj, dict):
      return {str(k): _safe_serialize(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return [_safe_serialize(v, _seen) for v in obj]
    if hasattr(obj, "model_dump"):
      return _safe_serialize(obj.model_dump(), _seen)
    if hasattr(obj, "__dict__"):
      return _safe_serialize(vars(obj), _seen)
    return repr(obj)
  finally:
    _seen.discard(obj_id)


def _dump_jsonl(record: dict) -> str:
  return json.dumps(_safe_serialize(record))


class SR2Handler(CustomLogger):
  def __init__(self):
    self._pass_through_set = False

  async def async_pre_call_hook(
    self,
    user_api_key_dict: UserAPIKeyAuth,
    cache: DualCache,
    data: dict,
    call_type: Literal[
      "completion",
      "text_completion",
      "embeddings",
      "image_generation",
      "moderation",
      "audio_transcription",
    ],
  ):

    if not self._pass_through_set:
      from litellm.proxy.proxy_server import llm_router

      if llm_router is not None:
        llm_router.router_general_settings.pass_through_all_models = True
        self._pass_through_set = True
        logger.info("Enabled pass_through_all_models on LiteLLM router")

    with open(LOG_PATH, "a") as f:
      d = {"hook": "async_pre_call_hook", "call_type": call_type, **data}
      f.write(_dump_jsonl(d) + "\n")
    return data

  async def async_post_call_success_hook(
    self,
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
    response: LLMResponseTypes,
  ):
    with open(LOG_PATH, "a") as f:
      d = {"hook": "async_post_call_success_hook", "data": data, "response": response}
      f.write(_dump_jsonl(d) + "\n")

  async def async_post_call_streaming_iterator_hook(
    self,
    user_api_key_dict: UserAPIKeyAuth,
    response: Any,
    request_data: dict,
  ) -> AsyncGenerator[ModelResponseStream, None]:

    chunks: list[ModelResponseStream] = []
    async for item in response:
      chunks.append(item)
      yield item

    with open(LOG_PATH, "a") as f:
      d = {
        "hook": "async_post_call_streaming_iterator_hook",
        "request_data": request_data,
        "response": chunks,
      }
      f.write(_dump_jsonl(d) + "\n")
