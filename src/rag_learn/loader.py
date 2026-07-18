"""Markdown discovery + chunking for ingestion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Compile once at module load.
_H1_RE = re.compile(r"(?m)^# .+$")
_PARA_SPLIT_RE = re.compile(r"\n\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。.!！?？;；])\s*")


@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    chunk_index: int
    char_start: int
    char_end: int


def _chunk_size() -> tuple[int, int]:
    # Re-read every call to keep tests monkeypatch-friendly.
    size = int(os.environ.get("CHUNK_SIZE", "800"))
    overlap = int(os.environ.get("CHUNK_OVERLAP", "50"))
    return size, overlap


def iter_markdown(docs_dir: str | Path) -> list[tuple[str, str]]:
    """Return list of (filename, raw_text) sorted by filename, deterministic."""
    p = Path(docs_dir)
    files = sorted(p.glob("*.md"))
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def _split_into_pre_docs(raw: str) -> list[str]:
    """Split a markdown file by H1 headings.

    If no H1 is present, return the whole file as one document. The H1
    line itself is omitted (it's metadata, not body).
    """
    matches = list(_H1_RE.finditer(raw))
    if not matches:
        return [raw.strip()]
    docs: list[str] = []
    # Pre-H1 content (e.g., a title block without '# ') becomes the first doc.
    pre = raw[: matches[0].start()].strip()
    if pre:
        docs.append(pre)
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        docs.append(raw[start:end].strip())
    return [d for d in docs if d]


def _chunk_text(text: str, source_file: str, start_offset: int = 0) -> list[Chunk]:
    """Greedy paragraph-then-sentence packing up to CHUNK_SIZE with OVERLAP."""
    size, overlap = _chunk_size()
    if not text.strip():
        return []

    paragraphs = _PARA_SPLIT_RE.split(text)
    chunks: list[Chunk] = []
    buf = ""
    chunk_start = 0
    cursor = 0
    local_offset = 0  # offset within the original `text`

    def flush():
        nonlocal buf, chunk_start, local_offset
        if not buf.strip():
            return
        # Trim trailing whitespace from the buffered chunk before persisting.
        body = buf.rstrip()
        end_in_text = chunk_start + len(body)
        chunks.append(
            Chunk(
                text=body,
                source_file=source_file,
                chunk_index=len(chunks),
                char_start=start_offset + chunk_start,
                char_end=start_offset + end_in_text,
            )
        )
        # Compute the next chunk's start by backing up `overlap` characters
        # from the end of the just-emitted chunk, so adjacent chunks share
        # `overlap` characters of context.
        new_local = max(0, end_in_text - overlap)
        buf = text[new_local:end_in_text]
        chunk_start = new_local
        local_offset = end_in_text
        cursor = end_in_text

    for para in paragraphs:
        para = para.strip()
        if not para:
            cursor += 2  # account for the \n\n we split on
            continue
        # If a single paragraph already exceeds the size, split by sentence.
        if len(para) > size:
            sentences = _SENTENCE_SPLIT_RE.split(para)
            for sent in sentences:
                if not sent.strip():
                    continue
                if len(buf) + len(sent) + 1 > size:
                    flush()
                buf = (buf + " " + sent).strip() if buf else sent
        else:
            if len(buf) + len(para) + 2 > size:
                flush()
            buf = (buf + "\n\n" + para).strip() if buf else para
        cursor += len(para) + 2

    flush()
    return chunks


def split_into_chunks(filename: str, raw_text: str) -> list[Chunk]:
    """Top-level entry: chunk a single file's content."""
    pre_docs = _split_into_pre_docs(raw_text)
    out: list[Chunk] = []
    for doc in pre_docs:
        out.extend(_chunk_text(doc, filename))
    # Re-index so chunk_index is contiguous within the file.
    for i, c in enumerate(out):
        # frozen dataclass, so rebuild
        out[i] = Chunk(
            text=c.text,
            source_file=c.source_file,
            chunk_index=i,
            char_start=c.char_start,
            char_end=c.char_end,
        )
    return out


def load_documents(docs_dir: str | Path) -> list[Chunk]:
    """Read all *.md in docs_dir and return a flat list of chunks."""
    chunks: list[Chunk] = []
    for name, raw in iter_markdown(docs_dir):
        chunks.extend(split_into_chunks(name, raw))
    return chunks
