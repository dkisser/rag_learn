"""Hybrid retriever: Chroma (vector) + BM25 (keyword), fused via RRF."""

from __future__ import annotations

from pathlib import Path

from rag_learn.loader import load_documents
from rag_learn.retriever.base import Hit
from rag_learn.retriever.bm25_index import BM25Index
from rag_learn.retriever.chroma_impl import ChromaRetriever


class HybridRetriever:
    """Run vector and keyword retrieval side-by-side; fuse with RRF.

    Implements ``BaseRetriever`` via duck-typing (``search`` + ``ensure_indexed``)
    so the pipeline treats it identically to ``ChromaRetriever``.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "shanzhongshi",
        *,
        rrf_k: int = 60,
        max_distance: float | None = None,
        vector_retriever: ChromaRetriever | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._rrf_k = rrf_k
        self._vector = vector_retriever or ChromaRetriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            max_distance=max_distance,
        )
        self._bm25 = bm25_index or BM25Index()

    def ensure_indexed(self, docs_dir: str) -> None:
        """Index once for the vector store and once for the BM25 keyword index."""
        # Vector side: delegates to ChromaRetriever (idempotent via count()).
        self._vector.ensure_indexed(docs_dir)
        # BM25 side: rebuild from the same source chunks so both indexes
        # cover the same corpus. Cheap for our small docs and only happens
        # once because HybridRetriever is cached per collection.
        if not self._bm25.is_built:
            self._bm25.build(load_documents(docs_dir))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Fetch top-k from each retriever and fuse with Reciprocal Rank Fusion."""
        vector_hits = self._vector.search(query, k=k)
        bm25_hits = self._bm25.search(query, k=k)

        rrf: dict[tuple[str, int], tuple[Hit, float]] = {}

        for rank, hit in enumerate(vector_hits):
            key = (hit.source_file, hit.chunk_index)
            prev_hit, prev_score = rrf.get(key, (hit, 0.0))
            rrf[key] = (prev_hit, prev_score + 1.0 / (self._rrf_k + rank + 1))

        for rank, hit in enumerate(bm25_hits):
            key = (hit.source_file, hit.chunk_index)
            prev_hit, prev_score = rrf.get(key, (hit, 0.0))
            rrf[key] = (prev_hit, prev_score + 1.0 / (self._rrf_k + rank + 1))

        fused = sorted(rrf.values(), key=lambda item: item[1], reverse=True)
        return [
            Hit(
                text=h.text,
                source_file=h.source_file,
                chunk_index=h.chunk_index,
                score=float(score),
            )
            for h, score in fused[:k]
        ]
