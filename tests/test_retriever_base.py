from dataclasses import FrozenInstanceError

import pytest
from rag_learn.retriever.base import BaseRetriever, Hit


def test_hit_is_frozen():
    hit = Hit(text="x", source_file="a.md", chunk_index=0, score=0.1)
    with pytest.raises(FrozenInstanceError):
        hit.text = "y"  # type: ignore[misc]


def test_hit_equality():
    a = Hit(text="x", source_file="a.md", chunk_index=0, score=0.1)
    b = Hit(text="x", source_file="a.md", chunk_index=0, score=0.1)
    assert a == b


def test_protocol_recognises_conforming_class():
    class Fake:
        def search(self, query: str, k: int = 5) -> list[Hit]:
            return [Hit(text=query, source_file="x.md", chunk_index=0, score=0.0)]

        def ensure_indexed(self, docs_dir: str) -> None:
            return None

    assert isinstance(Fake(), BaseRetriever)


def test_protocol_rejects_non_conforming():
    class NotARetriever:
        pass

    assert not isinstance(NotARetriever(), BaseRetriever)
