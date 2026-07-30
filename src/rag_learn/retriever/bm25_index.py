"""BM25 keyword index built on top of jieba + rank-bm25.

Used by ``HybridRetriever`` as the lexical half of a hybrid (vector + BM25)
retrieval pipeline. Index is held in memory and rebuilt on first call to
``build()``; the wrapped ``BaseRetriever`` caches the ``HybridRetriever``
instance per collection, so the rebuild only happens once per process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_learn.loader import Chunk
from rag_learn.retriever.base import Hit

_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class _IndexedChunk:
    """Internal record: original chunk + its tokenized text."""

    chunk: Chunk
    tokens: list[str]


def _tokenize(text: str) -> list[str]:
    """Tokenize text with jieba, dropping whitespace and pure-punctuation tokens.

    Lazy-import jieba so unrelated tests don't pay the import cost.
    """
    import jieba

    raw = jieba.lcut(text, cut_all=False)
    return [tok for tok in raw if tok.strip() and not _PUNCT_ONLY_RE.match(tok)]


class BM25Index:
    """In-memory BM25 keyword index."""

    def __init__(self) -> None:
        self._docs: list[_IndexedChunk] = []
        self._bm25 = None  # lazily set by ``build()``

    def build(self, chunks: list[Chunk]) -> None:
        """(Re)build the index from a fresh chunk list."""
        # Lazy import keeps the module import-cheap for non-BM25 tests.
        from rank_bm25 import BM25Okapi

        self._docs = [_IndexedChunk(chunk=chunk, tokens=_tokenize(chunk.text)) for chunk in chunks]
        corpus = [d.tokens for d in self._docs]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to ``k`` hits sorted by descending BM25 score."""
        if self._bm25 is None or not self._docs:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        # ``get_scores`` returns a 1-D ndarray aligned with ``self._docs``.
        if max(scores) <= 0:
            # No term overlap, or every term has zero IDF (appears in >50% of
            # docs). Either way the corpus gives us no useful signal.
            return []
        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        hits: list[Hit] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            if len(hits) >= k:
                break
            doc = self._docs[idx]
            hits.append(
                Hit(
                    text=doc.chunk.text,
                    source_file=doc.chunk.source_file,
                    chunk_index=doc.chunk.chunk_index,
                    score=float(score),
                )
            )
        return hits

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None
