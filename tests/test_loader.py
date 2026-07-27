from pathlib import Path

from rag_learn.loader import iter_markdown, load_documents, split_into_chunks

FIXTURES = Path(__file__).parent / "fixtures" / "sample_docs"


def test_iter_markdown_returns_sorted_files():
    items = iter_markdown(FIXTURES)
    names = [n for n, _ in items]
    assert names == sorted(names)
    assert all(name.endswith(".md") for name in names)


def test_load_documents_assigns_filenames():
    chunks = load_documents(FIXTURES)
    assert {c.source_file for c in chunks} == {
        "doc_with_h1.md",
        "doc_no_h1.md",
        "doc_short_section.md",
        "doc_beans.md",
    }


def test_split_into_chunks_raw_includes_only_requested_file():
    chunks = split_into_chunks("doc_short_section.md", "# Tiny\n\nJust a few lines.")
    assert len(chunks) == 1
    assert chunks[0].source_file == "doc_short_section.md"
    assert "Tiny" in chunks[0].text or "few lines" in chunks[0].text
