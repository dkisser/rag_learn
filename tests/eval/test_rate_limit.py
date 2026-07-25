"""Tests for rag_learn.rate_limit.

Composes pyrate-limiter + threading.Semaphore + tenacity. Most tests pass a
fake Limiter via the _rate_limiter kwarg so rate behaviour does not depend
on wall-clock timing; concurrency is exercised via the real Semaphore.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from openai import RateLimitError
from pyrate_limiter import Limiter

from rag_learn.rate_limit import RateLimiter, is_rate_limit_error

# --------------------------------------------------------------------------- #
# is_rate_limit_error
# --------------------------------------------------------------------------- #


def test_is_rate_limit_error_detects_openai_rate_limit_error() -> None:
    exc = RateLimitError("rate limited", response=MagicMock(), body={})
    assert is_rate_limit_error(exc) is True


def test_is_rate_limit_error_detects_httpx_style_429() -> None:
    fake_response = MagicMock()
    fake_response.status_code = 429
    exc = Exception("boom")
    exc.response = fake_response  # type: ignore[attr-defined]
    assert is_rate_limit_error(exc) is True


def test_is_rate_limit_error_rejects_other_errors() -> None:
    assert is_rate_limit_error(ValueError("nope")) is False
    assert is_rate_limit_error(RuntimeError("boom")) is False
    fake_response = MagicMock()
    fake_response.status_code = 500
    exc = Exception("server")
    exc.response = fake_response  # type: ignore[attr-defined]
    assert is_rate_limit_error(exc) is False


# --------------------------------------------------------------------------- #
# Concurrency cap (real Semaphore)
# --------------------------------------------------------------------------- #


def _fake_limiter() -> Limiter:
    """A Limiter whose try_acquire is a no-op (does not block on rate)."""
    fake = MagicMock(spec=Limiter)
    fake.try_acquire.return_value = True
    return fake


def test_acquire_respects_concurrency_cap() -> None:
    """Two acquires inside the cap may overlap; the third blocks until release."""
    rl = RateLimiter(max_concurrency=2, rate_per_minute=600, _rate_limiter=_fake_limiter())

    inside: list[int] = []
    release = threading.Event()
    barrier = threading.Barrier(3)

    def hold(perm_id: int) -> None:
        with rl.acquire():
            inside.append(perm_id)
            barrier.wait(timeout=1.0)  # ensure all 3 reached this point
            release.wait(timeout=1.0)

    threads = [threading.Thread(target=hold, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()

    # Give threads a moment to enter; only 2 should be inside at once.
    time.sleep(0.05)
    assert len(inside) == 2, f"expected 2 inside, got {len(inside)}"

    release.set()
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "thread stuck after release"

    assert sorted(inside) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# call() — happy path
# --------------------------------------------------------------------------- #


def test_call_returns_value_on_success() -> None:
    rl = RateLimiter(max_concurrency=2, rate_per_minute=600, _rate_limiter=_fake_limiter())

    def add(a: int, b: int) -> int:
        return a + b

    assert rl.call(add, 2, 3) == 5


def test_call_propagates_non_rate_limit_error_immediately() -> None:
    rl = RateLimiter(
        max_concurrency=2, rate_per_minute=600, max_retries=5, _rate_limiter=_fake_limiter()
    )
    calls = {"n": 0}

    def boom() -> str:
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        rl.call(boom)

    # tenacity should NOT retry a non-rate-limit error
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# call() — retry on 429
# --------------------------------------------------------------------------- #


def test_call_retries_on_rate_limit_and_succeeds() -> None:
    """Two RateLimitErrors then success → final value returned, no exception."""
    rl = RateLimiter(
        max_concurrency=2, rate_per_minute=600, max_retries=3, _rate_limiter=_fake_limiter()
    )
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitError("429", response=MagicMock(), body={})
        return "ok"

    assert rl.call(flaky) == "ok"
    assert len(attempts) == 3


def test_call_gives_up_after_max_retries() -> None:
    rl = RateLimiter(
        max_concurrency=2, rate_per_minute=600, max_retries=2, _rate_limiter=_fake_limiter()
    )
    calls = {"n": 0}

    def always_fail() -> None:
        calls["n"] += 1
        raise RateLimitError("429", response=MagicMock(), body={})

    with pytest.raises(RateLimitError):
        rl.call(always_fail)

    # tenacity: max_attempt_number=2 means the function runs at most 2 times
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# call() — rate limiter is invoked per call
# --------------------------------------------------------------------------- #


def test_call_invokes_underlying_rate_limiter() -> None:
    fake = _fake_limiter()
    rl = RateLimiter(max_concurrency=2, rate_per_minute=600, _rate_limiter=fake)

    rl.call(lambda: 1)
    rl.call(lambda: 2)

    assert fake.try_acquire.call_count == 2
    args, _ = fake.try_acquire.call_args
    assert args[0] == "rag_eval"
