"""
Compatibility shim for generation helpers.

Historically, some modules imported ``python.model_generation`` while the
actual implementation lives in ``python.xai_0.model_generation``.

This module simply re-exports the public helpers so that both import paths
work:

    from python.model_generation import chat_completion, _simple_chat_prompt_to_ids
"""

from python.xai_0.model_generation import (  # type: ignore[F401]
    _simple_chat_prompt_to_ids,
    chat_completion,
    get_cache_token_count,
    get_text_token_count,
    text_completion,
)

__all__ = [
    "chat_completion",
    "text_completion",
    "get_text_token_count",
    "get_cache_token_count",
    "_simple_chat_prompt_to_ids",
]

