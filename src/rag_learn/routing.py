"""Intent classification and query decomposition for catalog-coverage queries.

Two small LLM-asked decisions that gate the catalog-coverage branch in
``pipeline.answer_stream``:

  * ``classify_intent`` — labels a user question as ``"specific"`` (point
    query) or ``"all"`` (compare-all / recommend / list-everything).
  * ``decompose_query`` — enumerates 3-6 sub-queries per a small catalog
    summary so the retriever can fan out for coverage.

Both functions drain ``llm.stream(...)`` into a string with a hard
timeout and fall back to safe defaults (``"specific"`` for classify,
``[question]`` for decompose) on any failure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Literal, Protocol

INTENT_LABELS = Literal["specific", "all"]
DECOMPOSE_MAX: Final = 8

INTENT_SYSTEM_PROMPT = (
    "You classify a user question for a RAG system. "
    "Reply with EXACTLY ONE WORD: `all` if the user wants comprehensive coverage "
    '(e.g. "推荐几款", "对比所有", "list everything"), '
    "otherwise `specific`. No punctuation, no explanation."
)

DECOMPOSE_SYSTEM_PROMPT = (
    "You help a RAG system that searches a small catalog. "
    "Decompose the user's question into 3-6 short sub-queries, each focused on one facet.\n\n"
    "Catalog summary:\n{catalog_summary}\n\n"
    'Output format: a JSON array of strings, e.g. ["sub1", "sub2"]. '
    "Reply with ONLY the JSON array, no prose."
)

_INTENT_LABELS_SET: Final[frozenset[str]] = frozenset({"all", "catalog", "compare-all", "coverage"})
_INTENT_SPECIFIC_SET: Final[frozenset[str]] = frozenset({"specific", "narrow", "single", "no"})


class _LLMStream(Protocol):
    """Duck-typed LLM interface: only ``stream`` is needed."""

    def stream(self, system: str, user: str) -> Iterator[str]: ...


def _drain_stream(llm: _LLMStream, system: str, user: str, timeout_s: float) -> str:
    """Drain ``llm.stream`` to a string with a hard timeout.

    Returns ``""`` on any failure (timeout, exception, empty stream). The
    caller is responsible for parsing the string and falling back to a
    safe default when empty.

    Implementation note: the executor is shut down with ``wait=False`` so
    that a timed-out worker thread does not block the caller waiting for
    the still-running ``time.sleep`` (or any LLM hang) to finish.
    """
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(lambda: "".join(llm.stream(system, user)))
        return fut.result(timeout=timeout_s)
    except Exception:
        return ""
    finally:
        ex.shutdown(wait=False)


def _parse_intent(text: str) -> INTENT_LABELS | None:
    """Best-effort parser for ``classify_intent`` LLM reply.

    Accepts:
      * the word alone on the first line: ``"all"`` / ``"specific"``
      * with surrounding punctuation / quotes
      * a chatty prefix on the first line (e.g. ``"Sure! all"``)
    """
    if not text:
        return None
    head = text.strip().splitlines()[0].strip().strip(" .,:;`'\"")
    head_lower = head.lower()
    if head_lower in _INTENT_LABELS_SET:
        return "all"
    if head_lower in _INTENT_SPECIFIC_SET:
        return "specific"
    # Last-ditch: grep the first line for the word "all".
    if re.search(r"\ball\b", head_lower):
        return "all"
    return None


def classify_intent(
    llm: _LLMStream,
    question: str,
    *,
    timeout_s: float = 8.0,
) -> INTENT_LABELS:
    """Return ``"all"`` or ``"specific"`` for the user's question.

    On any failure (timeout, garbled reply, empty stream) the function
    falls back to ``"specific"`` so the existing point-query path
    remains the safe default.
    """
    reply = _drain_stream(llm, INTENT_SYSTEM_PROMPT, question, timeout_s).strip()
    label = _parse_intent(reply)
    return label if label is not None else "specific"


def _parse_subqueries(text: str, cap: int) -> list[str]:
    """Extract a list of sub-queries from a (possibly chatty) LLM reply.

    Tries JSON array first, then falls back to bullet-style lines. Strips
    blanks, dedupes preserving order, and caps at ``cap`` items.
    """
    if not text:
        return []
    # 1. Prefer JSON array.
    m = re.search(r"\[.*?\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
        except Exception:
            arr = None
        if isinstance(arr, list):
            out = [str(x).strip() for x in arr if str(x).strip()]
            if out:
                return list(dict.fromkeys(out))[:cap]
    # 2. Bullet / list parser.
    out: list[str] = []
    for line in text.splitlines():
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if s and len(s) <= 200:
            out.append(s)
        if len(out) >= cap:
            break
    return list(dict.fromkeys(out))


def decompose_query(
    llm: _LLMStream,
    question: str,
    catalog_summary: str,
    *,
    max_sub_queries: int = DECOMPOSE_MAX,
    timeout_s: float = 15.0,
) -> list[str]:
    """Split ``question`` into a list of sub-queries for catalog coverage.

    On failure (timeout, garbled reply, empty stream) returns
    ``[question]`` so the caller still has a single sub-query to feed
    into the fan-out (degenerate but crash-free).
    """
    prompt = DECOMPOSE_SYSTEM_PROMPT.format(catalog_summary=catalog_summary)
    reply = _drain_stream(llm, prompt, question, timeout_s)
    out = _parse_subqueries(reply, cap=max_sub_queries)
    return out if out else [question]
