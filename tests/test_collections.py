"""Tests for Collection dataclass and lazy retriever cache."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rag_learn.retriever.base import BaseRetriever, Hit


class FakeRetriever:
    """Minimal retriever that satisfies BaseRetriever for testing."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.constructed_with: tuple[Path, str] | None = None
        self.ensure_calls = 0

    def ensure_indexed(self, docs_dir: str) -> None:
        self.ensure_calls += 1
        self.constructed_with = (self.persist_dir, self.collection_name)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return []


def _fake_factory(persist_dir: Path, name: str) -> BaseRetriever:
    return FakeRetriever(persist_dir, name)


@pytest.fixture
def fake_docs(tmp_path: Path) -> Path:
    p = tmp_path / "docs"
    p.mkdir()
    (p / "a.md").write_text("# A\n\nhello world")
    return p


# ---- __post_init__ validation ----


def test_collection_rejects_name_too_short(fake_docs: Path):
    from rag_learn.collections import Collection

    with pytest.raises(ValueError, match="Invalid collection name"):
        Collection(name="x", display_name="x", docs_dir=fake_docs)


def test_collection_rejects_name_with_slash(fake_docs: Path):
    from rag_learn.collections import Collection

    with pytest.raises(ValueError, match="Invalid collection name"):
        Collection(name="bad/name", display_name="x", docs_dir=fake_docs)


def test_collection_rejects_missing_docs_dir(tmp_path: Path):
    from rag_learn.collections import Collection

    with pytest.raises(ValueError, match="docs_dir does not exist"):
        Collection(name="abc", display_name="x", docs_dir=tmp_path / "nope")


# ---- lazy retriever ----


def test_collection_retriever_is_lazy(fake_docs: Path):
    from rag_learn.collections import Collection

    c = Collection(
        name="abc",
        display_name="ABC",
        docs_dir=fake_docs,
        retriever_factory=_fake_factory,
    )
    # No construction yet
    assert getattr(c, "_retriever", None) is None


def test_collection_retriever_caches(fake_docs: Path):
    from rag_learn.collections import Collection

    c = Collection(
        name="abc",
        display_name="ABC",
        docs_dir=fake_docs,
        retriever_factory=_fake_factory,
    )
    r1 = c.retriever
    r2 = c.retriever
    assert r1 is r2
    assert isinstance(r1, FakeRetriever)
    assert r1.collection_name == "abc"
    assert r1.ensure_calls == 1  # ensure_indexed called exactly once


def test_collection_is_frozen(fake_docs: Path):
    from rag_learn.collections import Collection

    c = Collection(
        name="abc",
        display_name="ABC",
        docs_dir=fake_docs,
        retriever_factory=_fake_factory,
    )
    with pytest.raises(FrozenInstanceError):
        c.display_name = "other"  # type: ignore[misc]
