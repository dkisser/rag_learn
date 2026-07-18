"""RAG pipeline prompt construction and streaming performance data."""

from __future__ import annotations

from dataclasses import dataclass

from rag_learn.config import CHUNK_DISPLAY_CHARS
from rag_learn.retriever import Hit

SYSTEM_PROMPT = (
    "你是一个 RAG 助手。仅基于下方提供的「上下文」回答用户问题。"
    "如果上下文不足以回答，直接说「未找到相关上下文」。"
    "不要使用先验知识或编造内容。"
)

EMPTY_HITS_SYSTEM_PROMPT = (
    "你是一个 RAG 助手。当前没有检索到任何相关上下文，"
    "请直接告诉用户「未找到相关上下文」，不要使用先验知识或编造内容。"
)


@dataclass(frozen=True)
class StreamPerf:
    retrieve_ms: float
    first_token_ms: float
    total_ms: float
    finished_at: str  # HH:MM:SS.mmm


def build_prompt(chunks: list[Hit], question: str) -> tuple[str, str]:
    """Return ``(system_msg, user_msg)`` with display-safe chunk lengths."""
    if not chunks:
        return EMPTY_HITS_SYSTEM_PROMPT, f"问题：{question}\n回答："

    lines = ["上下文："]
    for i, hit in enumerate(chunks, start=1):
        text = hit.text
        if len(text) > CHUNK_DISPLAY_CHARS:
            text = text[:CHUNK_DISPLAY_CHARS]
        lines.append(f"[{i}] (来源: {hit.source_file}) {text}")
    user_msg = "\n".join(lines) + f"\n\n问题：{question}\n回答："
    return SYSTEM_PROMPT, user_msg
