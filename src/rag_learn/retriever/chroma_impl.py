"""Chroma adapter implementing BaseRetriever via PersistentClient + default embedder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from rag_learn.loader import load_documents
from rag_learn.retriever.base import Hit

_DISTANCE_FILTER_OVERSAMPLE = 4


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "rag_doc",
        *,
        max_distance: float | None = None,
    ) -> None:
        if max_distance is not None and (not math.isfinite(max_distance) or max_distance < 0):
            raise ValueError("max_distance must be a finite non-negative number")
        # Local import keeps the module import-cheap when running other tests
        # that don't touch Chroma (which downloads models on first call).
        import chromadb

        self._collection_name = collection_name
        self._max_distance = max_distance
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_indexed(self, docs_dir: str) -> None:
        if self._collection.count() > 0:
            return
        chunks = load_documents(docs_dir)
        if not chunks:
            return
        ids = [f"{c.source_file}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        # Chroma's Metadata type is Mapping[str, str | int | float], so widen
        # chunk_index to float (chunks store int, but the wire format is wider).
        metadatas: list[dict[str, str | int | float]] = [
            {"source_file": c.source_file, "chunk_index": c.chunk_index} for c in chunks
        ]
        # Insert in one call; chromadb batches internally.
        self._collection.add(ids=ids, documents=documents, metadatas=cast(Any, metadatas))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        # distance 由 Chroma 在查询时计算，因此必须在查询后过滤。
        # 超量召回为 max_distance 淘汰候选留出空间；阈值严格时结果可以少于 k。
        n_results = k
        if self._max_distance is not None:
            n_results = max(k, k * _DISTANCE_FILTER_OVERSAMPLE)
        result = self._collection.query(query_texts=[query], n_results=n_results)
        # result.get may return None per the stub; default to [] and treat any
        # None as "no results" (Chroma only returns None for empty collections).
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        hits: list[Hit] = []
        for text, meta, dist in zip(documents[0], metadatas[0], distances[0], strict=False):
            distance = float(dist)
            if self._max_distance is not None and distance > self._max_distance:
                continue
            hits.append(
                Hit(
                    text=str(text),
                    source_file=str(meta["source_file"]),
                    chunk_index=int(meta["chunk_index"]),
                    score=distance,
                )
            )
            if len(hits) >= k:
                break
        return hits
