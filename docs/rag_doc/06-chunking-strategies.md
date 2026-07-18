# Chunking Strategies for LLM Applications

> **Source**: https://www.pinecone.io/learn/chunking-strategies/
> **抓取日期**: 2026-07-18
> **作者/机构**: Pinecone Learning Center

## Overview

**Chunking** is the process of breaking down large text into smaller segments called chunks. It optimizes relevance of content stored in vector databases by ensuring chunks are large enough to contain meaningful information while small enough for performant, low-latency responses in RAG and agentic workflows.

### Why Chunking is Necessary

Two main reasons:

1. **Context window limits** — Embedding models have context windows determining token capacity for vector creation. Exceeding this causes truncation of important context.
2. **Relevant search results** — Chunks must contain useful information; sentences without context may not surface during queries.

### Key Factors for Choosing Strategy

- Type of data (long documents vs. short content like tweets)
- Embedding model capabilities and token limits
- User query length and complexity
- Application use case (semantic search, RAG, agentic workflows)

---

## Chunking Methods

### 1. Fixed-Size Chunking

The most common approach: decide token count and break documents into fixed-size chunks. Typically uses the max context window of the embedding model (e.g., 1024 for `llama-text-embed-v2`, 8196 for `text-embedding-3-small`).

**Best practice:** Start here and iterate only after determining it is insufficient.

### 2. Content-Aware Chunking

#### Simple Sentence and Paragraph Splitting

Tools for sentence-level chunking:

- **Naive splitting** — Split by periods, newlines, or whitespace.
- **NLTK** — Provides a trained sentence tokenizer.
- **spaCy** — Offers sophisticated sentence segmentation.

#### Recursive Character Level Chunking

LangChain's `RecursiveCharacterTextSplitter` splits text using separators in order: `["\n\n", "\n", " ", ""]`. This balances structure awareness with fixed chunk sizes.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

chunks = text_splitter.split_text(document)
```

The splitter first tries to keep paragraphs together (`\n\n`); if a paragraph is too long, it falls back to sentences (`\n`, then ` `), and finally to characters.

### 3. Document Structure-Based Chunking

For PDFs, DOCX, HTML, Markdown, and LaTeX:

- **PDF** — Headers, text, and tables require preprocessing.
- **HTML** — Use tags like `<p>`, `<title>` to inform chunking.
- **Markdown** — Recognize headings, lists, and code blocks.
- **LaTeX** — Parse commands and environments.

### 4. Semantic Chunking

First introduced by Greg Kamradt. Uses embeddings to identify thematic shifts:

1. Break document into sentences.
2. Group each sentence with surrounding sentences into a window.
3. Generate embeddings for each group.
4. Compare semantic distance between consecutive groups to identify topic shifts.

When the cosine distance exceeds a threshold, a new chunk is started.

### 5. Contextual Chunking with LLMs

Anthropic's **Contextual Retrieval** prompts an LLM with the entire document and each chunk to generate a contextualized description that is appended before embedding.

---

## Best Practices

### Selecting Chunk Sizes

- Test smaller chunks (128–256 tokens) for granular semantics.
- Test larger chunks (512–1024 tokens) for more context.

### Evaluating Performance

- Use multiple indices or namespaces for testing.
- Run representative queries to evaluate quality.
- Iterate until finding the optimal chunk size.

---

## Comparison of Strategies

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| Fixed-size | Fastest | Baseline | Prototyping |
| Recursive char | Fast | Good | General-purpose text |
| Sentence/paragraph | Medium | Good | Articles, blogs |
| Document structure | Medium | High | PDFs, Markdown, HTML |
| Semantic | Slow | Higher | Long-form, mixed-topic docs |
| Contextual (LLM) | Slowest | Highest | High-stakes RAG |

---

## Key Takeaways

- No one-size-fits-all solution exists.
- Start with fixed-size chunking and iterate.
- For most production RAG systems, **Recursive Character Text Splitting** with `chunk_size ≈ 500–1000` and `chunk_overlap ≈ 10–20%` is the recommended default.
