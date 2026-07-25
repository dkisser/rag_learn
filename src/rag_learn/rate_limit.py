"""Thin wrapper composing pyrate-limiter + threading.Semaphore + tenacity.

Used by the batch RAG evaluation CLI to pace DeepSeek calls and retry on
429 RateLimitError. Pyrate-limiter v4 supplies token-bucket-style RPM
pacing; tenacity supplies exponential-backoff retry; the stdlib Semaphore
caps instantaneous concurrency. We do not wrap OpenAI's default SDK
retries — they remain as a defense-in-depth backstop below this layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Semaphore
from typing import Any, TypeVar

from openai import RateLimitError as OpenAIRateLimitError
from pyrate_limiter import Duration, Limiter, Rate
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RATE_LIMIT_BUCKET_NAME = "rag_eval"


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` represents an HTTP 429 from any layer."""
    if isinstance(exc, OpenAIRateLimitError):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


class RateLimiter:
    """Combine RPM pacing, in-flight concurrency cap, and 429 retry.

    Use :meth:`acquire` as a context manager around the protected section,
    or :meth:`call` to wrap a single callable. Both paths block on rate
    bucket exhaustion (pyrate-limiter's blocking ``try_acquire``) and on
    semaphore availability.

    :param max_concurrency: max in-flight protected calls at once.
    :param rate_per_minute: max permits issued per rolling minute.
    :param max_retries: tenacity ``stop_after_attempt`` ceiling on 429s.
    :param _rate_limiter: injection seam for tests; production callers
        leave this alone.
    """

    def __init__(
        self,
        max_concurrency: int = 3,
        rate_per_minute: float = 20.0,
        max_retries: int = 3,
        *,
        _rate_limiter: Limiter | None = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self._sem = Semaphore(max_concurrency)
        self._rate_limiter: Limiter = _rate_limiter or Limiter(
            Rate(int(rate_per_minute), Duration.MINUTE)
        )
        self._max_retries = max_retries
        self._retrying = Retrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=30.0),
            retry=retry_if_exception(is_rate_limit_error),
            reraise=True,
        )

    @contextmanager
    def acquire(self) -> Iterator[None]:
        """Block until a concurrency slot AND a rate token are available."""
        with self._sem:
            self._rate_limiter.try_acquire(_RATE_LIMIT_BUCKET_NAME, blocking=True, timeout=-1)
            yield

    def call(self, fn: Callable[..., Any], /, *args: object, **kwargs: object) -> Any:
        """Run ``fn`` under both limiters, retrying on 429 with backoff."""
        for attempt in self._retrying:
            with attempt:
                with self.acquire():
                    return fn(*args, **kwargs)
        # Unreachable: tenacity.reraise=True re-raises the last exception
        # before iteration ends, so we never fall out of the loop normally.
        raise RuntimeError("RateLimiter.call exhausted attempts without raising")
