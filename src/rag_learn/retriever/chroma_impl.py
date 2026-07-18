"""Chroma adapter implementing BaseRetriever via PersistentClient + default embedder."""

from __future__ import annotations

from pathlib import Path

from rag_learn.loader import load_documents
from rag_learn.retriever.base import Hit


class ChromaRetriever:
    def __init__(self, persist_dir: Path, collection_name: str = "rag_doc") -> None:
        # Local import keeps the module import-cheap when running other tests
        # that don't touch Chroma (which downloads models on first call).
        import chromadb

        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"},
        )

    def ensure_indexed(self, docs_dir: str) -> None:
        if self._collection.count() > 0:
            return
        chunks = load_documents(docs_dir)
        if not chunks:
            return
        ids = [f"{c.source_file}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"source_file": c.source_file, "chunk_index": c.chunk_index} for c in chunks]
        # Insert in one call; chromadb batches internally.
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        result = self._collection.query(query_texts=[query], n_results=k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[Hit] = []
        for text, meta, dist in zip(documents, metadatas, distances):
            hits.append(
                Hit(
                    text=text,
                    source_file=meta["source_file"],
                    chunk_index=int(meta["chunk_index"]),
                    score=float(dist),
                )
            )
        return hits
