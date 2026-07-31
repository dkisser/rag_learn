"""Tests for Collection dataclass and lazy retriever cache."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rag_learn.collections import Catalog, Collection, CollectionNotFoundError
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


# ---- Catalog ----


def _make_collection(name: str, display: str, docs_dir: Path) -> Collection:
    return Collection(
        name=name,
        display_name=display,
        docs_dir=docs_dir,
        retriever_factory=_fake_factory,
    )


def test_catalog_rejects_duplicate_names(fake_docs: Path):
    a = _make_collection("dup", "甲", fake_docs)
    b = _make_collection("dup", "乙", fake_docs)
    with pytest.raises(ValueError, match="duplicate"):
        Catalog(collections=(a, b))


def test_catalog_names_returns_in_order(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    assert c.names() == ["aaa", "bbb"]


def test_catalog_display_choices(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    assert c.display_choices() == [("甲", "aaa"), ("乙", "bbb")]


def test_catalog_iter_collections_returns_in_order(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    assert list(c.iter_collections()) == [a, b]


def test_catalog_iter_collections_empty():
    c = Catalog(collections=())
    assert list(c.iter_collections()) == []


def test_catalog_get_returns_matching(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    assert c.get("bbb") is b


def test_catalog_get_unknown_raises_collection_not_found(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    c = Catalog(collections=(a,))
    with pytest.raises(CollectionNotFoundError):
        c.get("nope")
    # CollectionNotFoundError IS-A KeyError
    with pytest.raises(KeyError):
        c.get("nope")


def test_catalog_ensure_all_indexed_calls_each_retriever_once(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    warnings = c.ensure_all_indexed()
    assert warnings == []
    assert a.retriever.ensure_calls == 1  # type: ignore[attr-defined]
    assert b.retriever.ensure_calls == 1  # type: ignore[attr-defined]


def test_catalog_ensure_all_indexed_fail_open(fake_docs: Path):
    a = _make_collection("good", "Good", fake_docs)

    def boom(persist_dir: Path, name: str) -> BaseRetriever:
        raise RuntimeError("boom")

    bad = Collection(
        name="bad",
        display_name="Bad",
        docs_dir=fake_docs,
        retriever_factory=boom,
    )
    c = Catalog(collections=(a, bad))
    warnings = c.ensure_all_indexed()
    assert len(warnings) == 1
    name, msg = warnings[0]
    assert name == "bad"
    assert "boom" in msg
    # good one still got constructed
    assert a.retriever.ensure_calls == 1  # type: ignore[attr-defined]


# ---- BUILTIN_COLLECTIONS + build_catalog ----


def test_build_catalog_contains_rag_doc_and_shanzhongshi():
    from rag_learn.collections import build_catalog

    catalog = build_catalog()
    names = set(catalog.names())
    assert {"rag_doc", "shanzhongshi"}.issubset(names)


def test_builtin_collections_point_at_real_docs_dirs():
    from rag_learn.collections import BUILTIN_COLLECTIONS

    for c in BUILTIN_COLLECTIONS:
        assert c.docs_dir.is_dir(), f"{c.name} docs missing at {c.docs_dir}"


def test_builtin_collection_names_have_chroma_compatible_slugs():
    import re

    from rag_learn.collections import BUILTIN_COLLECTIONS

    pat = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,61}[A-Za-z0-9]$")
    for c in BUILTIN_COLLECTIONS:
        assert pat.match(c.name), f"bad slug: {c.name}"
