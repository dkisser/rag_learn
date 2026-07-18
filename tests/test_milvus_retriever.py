from __future__ import annotations

from pathlib import Path

import pytest

from rag_learn.retriever.base import BaseRetriever
from rag_learn.retriever.milvus_impl import MilvusRetriever

EMBED_DIM = 384


@pytest.fixture
def milvus_path(tmp_path: Path) -> Path:
    return tmp_path / "milvus.db"


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
