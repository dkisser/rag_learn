from pathlib import Path

from rag_learn.loader import load_documents

FIXTURES = Path(__file__).parent / "fixtures" / "sample_docs"


def test_doc_with_h1_splits_at_h1():
    chunks = load_documents(str(FIXTURES))
    section_chunks = [c for c in chunks if c.source_file == "doc_with_h1.md"]
    assert len(section_chunks) >= 2  # at least one chunk per H1
    assert all(c.chunk_index >= 0 for c in section_chunks)


def test_doc_no_h1_treated_as_single_document():
    chunks = [c for c in load_documents(str(FIXTURES)) if c.source_file == "doc_no_h1.md"]
    assert len(chunks) >= 1


def test_short_section_becomes_one_chunk():
    chunks = [c for c in load_documents(str(FIXTURES)) if c.source_file == "doc_short_section.md"]
    assert len(chunks) == 1


def test_chunk_length_respects_limit(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    # loader._chunk_size() re-reads env vars at every call, so no module
    # reload is needed. (Reloading config.py here would invalidate class
    # identity for downstream tests like test_load_config_missing_api_key_raises.)
    chunks = load_documents(str(FIXTURES))
    max_len = 200 + 20 + 2  # CHUNK_SIZE + CHUNK_OVERLAP + 1 for newline join
    for c in chunks:
        assert len(c.text) <= max_len, (
            f"chunk {c.source_file}#{c.chunk_index} is {len(c.text)} chars, exceeds bound {max_len}"
        )


def test_chunks_have_monotonic_index_per_file():
    chunks = load_documents(str(FIXTURES))
    by_file: dict[str, list[int]] = {}
    for c in chunks:
        by_file.setdefault(c.source_file, []).append(c.chunk_index)
    for indices in by_file.values():
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)
