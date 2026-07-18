# GraphRAG: Unlocking LLM Discovery on Narrative Private Data

> **来源**: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
> **抓取日期**: 2026-07-18
> **作者/机构**: Jonathan Larson (Partner Data Architect) and Steven Truitt (Principal Program Manager), Microsoft

## Problem Setting

Enterprises and researchers increasingly want LLMs to reason over **private datasets** that the LLM has never seen during training: internal reports, news archives, legal documents, scientific corpora. When these datasets are **large, narrative, and richly interconnected** — think tens of thousands of news articles spanning geopolitical events — naive chunk-and-embed RAG breaks down because:

- Pure semantic similarity cannot answer *"what are the main themes in this corpus?"* — that is a **global, holistic** question, not a localized lookup.
- Each chunk is isolated; cross-document relationships are lost.
- "Sense-making" questions that span the whole dataset require aggregating information across hundreds of chunks, exceeding the LLM context window.

Microsoft Research's **GraphRAG** addresses this class of questions with a graph-based retrieval pipeline.

## The GraphRAG Idea

GraphRAG augments standard RAG with an **LLM-built knowledge graph** over the entire dataset. At indexing time, the LLM is prompted to extract every entity and every relationship it can find, producing a graph where:

- **Nodes** represent entities (people, places, organizations, events, concepts).
- **Edges** represent relationships between entities.
- **Edge weights / node degrees** capture how strongly or how often entities co-occur.

This graph is then partitioned into **communities** of densely connected entities using a graph machine learning algorithm (Leiden, Louvain, or similar). For each community, the LLM generates a **summary** that is cached alongside the graph. At query time, both the local graph neighborhoods and the community summaries populate the context window.

## Indexing Pipeline

```
Raw documents (chunks)
        |
        v
LLM entity & relationship extraction  ->  Knowledge Graph (nodes + edges)
        |
        v
Community detection (Leiden)        ->  Hierarchical communities
        |
        v
LLM community summarization         ->  Per-community summaries (cached)
        |
        v
Vector index over community summaries + entity descriptions
```

### Step 1: Chunking
Documents are split into text chunks small enough to fit in the LLM prompt. A sliding-window approach with overlap is used to avoid losing cross-boundary relationships.

### Step 2: Entity and Relationship Extraction
For every chunk, the LLM is prompted to emit structured tuples of `(entity, relationship, entity)`. The prompts are carefully engineered to elicit rich, domain-aware extractions.

### Step 3: Graph Construction
The LLM-emitted tuples are merged into a single graph. Duplicate entities are merged using fuzzy matching and embedding similarity. Each entity gets a description (also LLM-generated) and each relationship gets a weight (co-occurrence count).

### Step 4: Community Detection
A Leiden-based bottom-up clustering algorithm partitions the graph into hierarchical communities of closely related entities. The hierarchy allows GraphRAG to answer questions at multiple granularities.

### Step 5: Community Summarization
For every community at every level, the LLM generates a **natural-language summary** describing what the community is about. These summaries are then embedded and indexed for vector retrieval. This is the key step that makes "global sense-making" possible: the LLM pre-reads the whole corpus through the lens of the graph and writes summaries that can later be retrieved and stitched together.

## Query-Time Pipeline

```
User Query
    |
    v
Map relevant communities to query (vector + graph traversal)
    |
    v
Load partial community summaries + relevant entity descriptions
    |
    v
"Map-Reduce" answer synthesis: partial answers -> final answer
    |
    v
Response (with provenance citations to source documents)
```

The system records **provenance**: each claim in the final answer can be traced back to specific source documents, enabling human verification.

## Evaluation

Microsoft tested GraphRAG on the **VIINA dataset** — thousands of Russian and Ukrainian news articles from June 2023 (translated to English). This dataset is far too large to fit in any LLM context window, making it an ideal stress test for RAG.

They evaluated four axes:

| Metric | Definition | GraphRAG vs. baseline RAG |
|---|---|---|
| **Comprehensiveness** | Does the answer cover all key aspects of the question? | GraphRAG wins |
| **Human enfranchisement** | Does the answer help a non-expert understand the topic? | GraphRAG wins |
| **Diversity** | Does the answer provide varied perspectives? | GraphRAG wins |
| **Faithfulness** | Is the answer faithful to source material (SelfCheckGPT)? | Comparable |

The qualitative improvement on the first three metrics was substantial, while faithfulness was on par with standard RAG.

## Why GraphRAG Outperforms

For *global* queries ("Summarize the main themes of the dataset"), standard RAG retrieves a handful of locally-similar chunks and cannot aggregate. GraphRAG's community summaries **pre-aggregate** the dataset at multiple granularities, so the LLM at query time can compose answers from already-summarized clusters rather than from raw chunks.

For *local* queries ("Who is X?"), GraphRAG falls back to entity-centric retrieval and remains competitive with standard RAG.

## Use Cases

GraphRAG is well-suited to:

- News and journalism: summarizing geopolitical developments across thousands of articles.
- Legal and compliance: traversing entity relationships across contracts and filings.
- Scientific literature: identifying emerging themes and key entities.
- Customer support and product feedback: clustering issues by root cause.
- Enterprise search: combining exact keyword search with relational context.

## Strengths

- **Global sense-making**: the only mainstream RAG variant that handles corpus-wide questions out of the box.
- **Structured provenance**: every answer is traceable to source documents through the graph.
- **Hierarchical granularity**: communities at different resolutions support queries of varying scope.
- **Dataset coverage**: the indexing pipeline forces the LLM to look at every chunk, reducing the "blind spots" of top-k retrieval.

## Limitations and Open Questions

- **Indexing cost**: the LLM-driven entity extraction and community summarization are expensive — typically orders of magnitude more expensive than standard RAG indexing.
- **Latency**: building the graph can take hours for million-token corpora.
- **Graph quality**: extraction errors propagate; a missed entity is a missed opportunity for retrieval.
- **Dynamic data**: graph updates on changing corpora require re-running extraction, which is expensive.
- **Prompt sensitivity**: the quality of entity extraction is highly prompt-dependent.

## Operational Considerations

- GraphRAG works best when the corpus is **rich in entities and relationships**. For pure free-form prose without many named entities, the graph will be sparse and the benefit diminishes.
- The community-detection step should use a graph ML library that supports large graphs (e.g., igraph, NetworkX with limitations, or graph-tool).
- Storage: the resulting graph plus summaries can be larger than the original corpus; plan accordingly.

## Relationship to Other RAG Variants

GraphRAG is orthogonal to most other RAG variants:

- **Self-RAG / CRAG** focus on retrieval quality and self-critique.
- **GraphRAG** focuses on **indexing structure** — it changes *what gets retrieved* (community summaries and entity neighborhoods) rather than *how to evaluate retrieval*.
- It is naturally **composable**: GraphRAG's community summaries can feed into an Adaptive RAG router or a Self-RAG-style critique loop.

## Status

As of publication (February 13, 2024), Microsoft released an open-source reference implementation at `github.com/microsoft/graphrag`. The blog post marks GraphRAG as part of Microsoft's broader investment in **graph-based reasoning for foundation models**.
