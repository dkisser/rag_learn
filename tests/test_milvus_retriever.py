from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from rag_learn.retriever.base import BaseRetriever
from rag_learn.retriever.milvus_impl import (
    MilvusRetriever,
    _run_isolated,
)

EMBED_DIM = 384


@pytest.fixture
def milvus_path(tmp_path: Path) -> Path:
    return tmp_path / "milvus.db"


# ---------------------------------------------------------------------------
# Subprocess targets for _run_isolated tests. Module-level so multiprocessing
# 'spawn' can pickle them on macOS.
# ---------------------------------------------------------------------------


def _target_returns_cleanly(*_args, **_kwargs) -> None:
    """Returns normally → exit code 0 → _run_isolated should return True."""


def _target_raises(*_args, **_kwargs) -> None:
    raise RuntimeError("boom from subprocess")


def _target_sleeps(seconds: float, *_args, **_kwargs) -> None:
    time.sleep(seconds)


def _target_kills_self_with_sigsegv(*_args, **_kwargs) -> None:
    """The exact regression mode for milvus-lite 2.6+ on macOS ARM:
    a C-level SIGSEGV inside the subprocess."""
    os.kill(os.getpid(), signal.SIGSEGV)


class TestRunIsolated:
    """The subprocess wrapper exists so a milvus-lite C-level crash doesn't
    bring down the host Python interpreter. These tests pin down the four
    exit-shape branches: clean exit, exception, timeout, signal kill."""

    def test_returns_true_when_subprocess_exits_cleanly(self):
        assert _run_isolated(_target_returns_cleanly, timeout=5.0) is True

    def test_returns_false_when_subprocess_raises_exception(self):
        assert _run_isolated(_target_raises, timeout=5.0) is False

    def test_returns_false_when_subprocess_exceeds_timeout(self):
        # Sleep for 10s but give 0.2s — wrapper must terminate and return False.
        assert _run_isolated(_target_sleeps, 10.0, timeout=0.2) is False

    def test_returns_false_when_subprocess_is_killed_by_sigsegv(self):
        # The case the wrapper was built for: a C-level SIGSEGV (milvus-lite
        # on macOS ARM) only kills the subprocess, not the test runner.
        assert _run_isolated(_target_kills_self_with_sigsegv, timeout=5.0) is False


def test_milvus_retriever_ensure_indexed_then_search(milvus_path: Path, fixtures_dir: Path):
    r = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r.ensure_indexed(str(fixtures_dir))
    hits = r.search("alpha", k=3)
    assert isinstance(hits, list)
    assert all(hasattr(h, "text") and hasattr(h, "score") for h in hits)
    assert all(h.score >= 0 for h in hits)
    assert len(hits) <= 3


def test_milvus_retriever_is_base_retriever(milvus_path: Path):
    r = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    assert isinstance(r, BaseRetriever)


def test_milvus_retriever_is_idempotent(milvus_path: Path, fixtures_dir: Path):
    r1 = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r1.ensure_indexed(str(fixtures_dir))
    a = r1.search("alpha", k=5)
    r2 = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r2.ensure_indexed(str(fixtures_dir))  # must not re-insert
    b = r2.search("alpha", k=5)
    assert len(a) == len(b) and len(a) > 0


def test_milvus_retriever_reloads_released_collection(milvus_path: Path, fixtures_dir: Path):
    """A collection left in 'released' state from a prior session must be
    reloaded by ensure_indexed so search() can return hits (regression for the
    'Collection in state released' error reported when re-launching the app
    against an existing data/milvus.db).
    """
    r1 = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r1.ensure_indexed(str(fixtures_dir))
    # Simulate the stale 'released' state.
    r1._client.release_collection(r1._collection_name)
    # New retriever on the same path: collection exists but is released.
    r2 = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r2.ensure_indexed(str(fixtures_dir))  # must reload, not re-insert
    hits = r2.search("alpha", k=3)
    assert hits
