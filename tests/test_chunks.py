from pathlib import Path

from rag_learn.loader import load_documents, split_into_chunks

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


def test_h1_title_preserved_in_chunk_text():
    """Each H1's text must appear inside the resulting chunk so the embedding
    can match the title (e.g. bean name from 咖啡豆风味信息.md)."""
    chunks = [c for c in load_documents(str(FIXTURES)) if c.source_file == "doc_with_h1.md"]
    titles = ["Section Alpha", "Section Beta"]
    by_title = {t: [c for c in chunks if t in c.text] for t in titles}
    for t, hits in by_title.items():
        assert hits, f"H1 '{t}' was stripped from all chunks — embedding cannot match on it"
        assert all(t in h.text for h in hits)


def test_each_bean_card_chunk_holds_its_own_bean_name():
    """Multi-H1 fixture (bean-card layout): every chunk must mention only its
    own bean's H1 title, not a sibling's. Guards against cross-contamination."""
    chunks = [c for c in load_documents(str(FIXTURES)) if c.source_file == "doc_beans.md"]
    # All three H1s in doc_beans.md have a body, so we expect three chunks.
    assert len(chunks) == 3
    assert "苏帕摩" in chunks[0].text
    assert "Brisas de mayo" in chunks[0].text
    assert "耶加 TOH亚军地块" in chunks[1].text
    assert "Yirgacheffe Woreda" in chunks[1].text
    assert "达摩" in chunks[2].text
    # Cross-contamination check: bean names stay in their own chunk.
    assert "苏帕摩" not in chunks[1].text and "苏帕摩" not in chunks[2].text
    assert "耶加" not in chunks[0].text and "耶加" not in chunks[2].text


def test_h1_section_with_empty_body_is_dropped():
    """An H1 followed immediately by another H1 (no body in between) yields
    no chunk — no info to embed."""
    chunks = split_into_chunks(
        "inline.md",
        "# Has Body\n\nSome content here.\n\n# No Body\n\n# Real Body\n\nMore content.",
    )
    texts = [c.text for c in chunks]
    assert len(chunks) == 2
    assert any("Has Body" in t and "Some content" in t for t in texts)
    assert any("Real Body" in t and "More content" in t for t in texts)
    assert all("No Body" not in t for t in texts)
