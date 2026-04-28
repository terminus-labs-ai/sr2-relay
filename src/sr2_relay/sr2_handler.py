import logging
from typing import Any, AsyncGenerator, Literal, Optional

from fastapi import HTTPException
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponseStream

logger = logging.getLogger(__name__)


class SR2Handler(CustomLogger):
  def __init__(self):
    pass

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

    logger.info(f"Got a async_pre_call_hook; call_type = {call_type}; {data}")
    return data

  async def async_post_call_failure_hook(
    self,
    request_data: dict,
    original_exception: Exception,
    user_api_key_dict: UserAPIKeyAuth,
    traceback_str: Optional[str] = None,
  ) -> Optional[HTTPException]:
    """
    Transform error responses sent to clients.

    Return an HTTPException to replace the original error with a user-friendly message.
    Return None to use the original exception.

    Example:
        if isinstance(original_exception, litellm.ContextWindowExceededError):
            return HTTPException(
                status_code=400,
                detail="Your prompt is too long. Please reduce the length and try again."
            )
        return None  # Use original exception
    """
    logger.info(
      f"Got a async_post_call_failure_hook; original_exception = {original_exception}"
    )
    pass

  async def async_post_call_success_hook(
    self,
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
    response,
  ):
    logger.info(f"Got a async_post_call_success_hook; response = {response}")

    pass

  async def async_moderation_hook(  # call made in parallel to llm api call
    self,
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
    call_type: Literal[
      "completion",
      "embeddings",
      "image_generation",
      "moderation",
      "audio_transcription",
    ],
  ):
    logger.info(f"Got a async_moderation_hook; call_type = {call_type}")

    pass

  async def async_post_call_streaming_hook(
    self,
    user_api_key_dict: UserAPIKeyAuth,
    response: str,
  ):
    logger.info(f"Got a async_post_call_streaming_hook; response = {response}")

    pass

  async def async_post_call_streaming_iterator_hook(
    self,
    user_api_key_dict: UserAPIKeyAuth,
    response: Any,
    request_data: dict,
  ) -> AsyncGenerator[ModelResponseStream, None]:
    """
    Passes the entire stream to the guardrail

    This is useful for plugins that need to see the entire stream.
    """
    async for item in response:
      logger.info(
        f"Got a async_post_call_streaming_iterator_hook; response = {response}"
      )

      yield item
