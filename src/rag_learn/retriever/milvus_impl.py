"""Milvus Lite (embedded) adapter implementing BaseRetriever.

Uses pymilvus.model.dense.SentenceTransformerEmbeddingFunction with
all-MiniLM-L6-v2 to pre-compute embeddings at ingest and at query time.
The collection has a single `vector` field (384-dim L2) plus three scalar
fields: text, source_file, chunk_index.

NOTE: The original brief wrote `data=[query]` and relied on pymilvus's
implicit default embedder. pymilvus 2.6 removed that implicit path, so
we pre-compute via SentenceTransformerEmbeddingFunction explicitly. This
keeps the spec §6.4 contract (same embedder model class as Chroma) intact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag_learn.loader import load_documents
from rag_learn.retriever.base import Hit

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class MilvusRetriever:
    def __init__(
        self,
        db_path: Path,
        collection_name: str = "rag_doc",
        dim: int = 384,
    ) -> None:
        # Local imports keep this module cheap to import in unrelated tests.
        from pymilvus import MilvusClient
        from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

        self._db_path = Path(db_path)
        self._collection_name = collection_name
        self._dim = dim
        self._client = MilvusClient(uri=str(self._db_path))
        self._embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    def _collection_exists(self) -> bool:
        return self._client.has_collection(self._collection_name)

    def ensure_indexed(self, docs_dir: str) -> None:
        if self._collection_exists():
            # A collection from a previous run may be in 'released' state
            # (e.g. milvus-lite released it after a prior search, or it never
            # finished loading after a failed insert during the known
            # deadlock window). Loading is idempotent and required before
            # search() can return hits.
            self._client.load_collection(self._collection_name)
            return
        chunks = load_documents(docs_dir)
        if not chunks:
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            dimension=self._dim,
            metric_type="L2",
            auto_id=True,
        )
        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode_documents(texts)
        rows: list[dict[str, Any]] = []
        for c, emb in zip(chunks, embeddings, strict=False):
            rows.append(
                {
                    "text": c.text,
                    "source_file": c.source_file,
                    "chunk_index": int(c.chunk_index),
                    "vector": list(emb),
                }
            )
        self._client.insert(collection_name=self._collection_name, data=rows)
        self._client.flush(self._collection_name)
        self._client.load_collection(self._collection_name)
        logger.info("Milvus indexed %d chunks into %s", len(rows), self._collection_name)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self._collection_exists():
            return []
        query_vectors = self._embedder.encode_queries([query])
        raw = self._client.search(
            collection_name=self._collection_name,
            data=query_vectors,
            limit=k,
            output_fields=["text", "source_file", "chunk_index"],
        )
        results = raw[0] if raw else []
        hits: list[Hit] = []
        for r in results:
            entity = r.get("entity", {})
            hits.append(
                Hit(
                    text=str(entity.get("text", "")),
                    source_file=str(entity.get("source_file", "")),
                    chunk_index=int(entity.get("chunk_index", -1)),
                    score=float(r.get("distance", 0.0)),
                )
            )
        return hits
