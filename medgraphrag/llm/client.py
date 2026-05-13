import asyncio

import numpy as np
from openai import AsyncOpenAI

from medgraphrag.llm.config import get_chat_model, openai_client_kwargs


LLM_MAX_RETRY_ATTEMPTS = 8
LLM_RETRY_BASE_SECONDS = 5
LLM_RETRY_MAX_SECONDS = 120

_NON_RETRYABLE_ERROR_NAMES = {
    "AuthenticationError",
    "BadRequestError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_ERROR_KEYWORDS = (
    "429",
    "rate limit",
    "速率限制",
    "timeout",
    "timed out",
    "connection",
    "connect",
    "connection reset",
    "connection aborted",
    "remote disconnected",
    "server disconnected",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "proxy",
    "try again",
)


def _retry_wait_seconds(attempt: int) -> int:
    return min(LLM_RETRY_MAX_SECONDS, LLM_RETRY_BASE_SECONDS * (2 ** attempt))


def _is_recoverable_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, str) and status_code.isdigit():
        status_code = int(status_code)
    if status_code in _NON_RETRYABLE_STATUS_CODES:
        return False
    if status_code in _RETRYABLE_STATUS_CODES:
        return True

    error_name = exc.__class__.__name__
    if error_name in _NON_RETRYABLE_ERROR_NAMES:
        return False
    if error_name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }:
        return True

    message = str(exc).lower()
    if any(keyword in message for keyword in _RETRYABLE_ERROR_KEYWORDS):
        return True

    if status_code is not None and 500 <= status_code < 600:
        return True

    return False


async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    for attempt in range(LLM_MAX_RETRY_ATTEMPTS):
        try:
            async with AsyncOpenAI(**openai_client_kwargs()) as client:
                response = await client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
            return response.choices[0].message.content
        except Exception as exc:
            if (
                not _is_recoverable_llm_error(exc)
                or attempt == LLM_MAX_RETRY_ATTEMPTS - 1
            ):
                raise
            wait_seconds = _retry_wait_seconds(attempt)
            print(
                f"[LLM] 调用失败，将在 {wait_seconds} 秒后重试 "
                f"({attempt + 1}/{LLM_MAX_RETRY_ATTEMPTS}): {exc}"
            )
            await asyncio.sleep(wait_seconds)


async def gpt_4o_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        get_chat_model("gpt-4o"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        get_chat_model("gpt-4o-mini"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def local_embedding(texts: list[str]) -> np.ndarray:
    from medgraphrag.embedding.local import embed as _embed
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, lambda: _embed(texts))
    return np.array(embeddings)
