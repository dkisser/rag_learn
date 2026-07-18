# Self-Reflective RAG with LangGraph: Agentic Patterns for Retrieval

> **来源**: https://www.langchain.com/blog/agentic-rag-with-langgraph (redirected from blog.langchain.com/agentic-rag-with-langgraph/)
> **抓取日期**: 2026-07-18
> **作者/机构**: Ankush Gola (LangChain), published February 7, 2024

## From Basic RAG to Agentic RAG

The standard RAG pipeline is linear: embed the user query, retrieve top-k documents from a vector store, and pass them to an LLM with a "use these documents to answer" prompt. This works for simple cases, but lacks any mechanism for **self-correction** or **adaptation** when retrieval goes wrong.

**Agentic RAG** generalizes RAG by treating the retriever, the generator, and the tools around them as nodes in a **state machine** rather than a fixed pipeline. The LLM (acting as an "agent") decides at runtime:

- Whether to retrieve at all.
- Which retriever to use (vector store, web search, SQL, etc.).
- Whether to re-retrieve after grading documents.
- Whether to re-write the query.
- Whether to trust the generation.

LangGraph — LangChain's graph-based orchestration framework — is purpose-built for this. It provides nodes, conditional edges, and persistent state, which together enable **loops** and **feedback** that linear chains cannot express.

## Cognitive Architectures for RAG

The blog post identifies three architectural patterns of increasing capability:

### 1. Chains
Linear sequence: `Retrieve -> Generate`. Simple and predictable, but no self-correction.

### 2. Routing
The LLM inspects the query and chooses among multiple retrieval paths (e.g., one of several vector stores, or a SQL fallback). This handles **which retriever** but still cannot re-retrieve or re-write.

### 3. State Machines (the agentic approach)
Full graph with conditional edges and loops. Supports:
- Document grading and conditional regeneration.
- Query rewriting on retrieval failure.
- Web-search fallback when local retrieval is insufficient.
- Generation grading (hallucination detection).
- Multi-step retrieval for complex questions.

The blog post argues that **state machines are well-suited for self-reflective RAG** because loops are essential for self-correction.

## The Self-Reflective RAG Idea

> "Self-reflective RAG... captures the idea of using an LLM to self-correct poor quality retrieval and/or generations."

Concretely, the agent watches two failure modes:

1. **Poor retrieval** — none of the retrieved documents are relevant. Action: rewrite the query and try again, or fall back to web search.
2. **Poor generation** — the LLM hallucinates, or its answer doesn't address the question. Action: re-retrieve, re-prompt, or ask the user to clarify.

Both failure modes are detected by **grading** steps, which are themselves LLM calls with structured outputs.

## Implementing Corrective RAG (CRAG) in LangGraph

The blog walks through a LangGraph implementation of CRAG with these components:

### Nodes
- **Retrieve**: vector-store retrieval.
- **Grade Documents**: binary relevance grader per document.
- **Generate**: answer synthesis from filtered documents.
- **Transform Query**: query rewriting for retry.
- **Web Search**: Tavily Search API as a fallback.

### Conditional edges
- `Grade Documents -> Generate` if at least one document passes grading.
- `Grade Documents -> Transform Query -> Web Search -> Generate` otherwise.
- `Grade Generation -> END` if the answer is faithful and useful.
- `Grade Generation -> Transform Query -> Retrieve` if not.

### Grading with Pydantic + Function Calling
Each grader is a **Pydantic model** with `score: Literal["yes", "no"]`. The Pydantic model is bound to the LLM as an **OpenAI function/tool**, guaranteeing that the LLM returns structured output the graph can route on.

## Implementing Self-RAG in LangGraph

The blog also shows a simplified Self-RAG implementation. While the full Self-RAG paper trains reflection tokens into the LM, the LangGraph version implements the same logic at the **graph level**:

| Reflection step | Question asked of grader | Output |
|---|---|---|
| Document relevance | Are the retrieved documents relevant to the question? | yes / no |
| Hallucination check | Is the generation supported by the documents? | yes / no |
| Answer usefulness | Is the generation useful to the question? | yes / no |

### Reflection-token equivalents (from the Self-RAG paper)
For reference, the full Self-RAG paper defines four token types:

| Token | Decision | Values |
|---|---|---|
| **Retrieve** | When to retrieve | yes / no / continue |
| **ISREL** | Are passages relevant? | relevant / irrelevant |
| **ISSUP** | Is generation supported? | fully / partially / no |
| **ISUSE** | Is generation useful? | 5, 4, 3, 2, 1 |

In the LangGraph implementation, these become **separate grading nodes** instead of LM-emitted tokens, but the routing logic is identical.

### Self-RAG Flow
```
Retrieve
   |
   v
Grade Documents (binary relevance per doc)
   |
   +-- any "yes"  --> Generate --> Grade Generation
   |                                    |
   |                                    +-- supported & useful --> END
   |                                    +-- not useful         --> Transform Query --> Retrieve (loop)
   |
   +-- all "no"   --> Transform Query --> Retrieve (loop)
```

This graph re-enters the retrieval loop until either a satisfactory answer is produced or a step limit is hit.

## Key Technical Elements

- **LangGraph**: state machine with `StateGraph`, nodes as Python functions, conditional edges as router functions.
- **Tavily Search**: web search API used as a fallback when local retrieval fails.
- **Pydantic**: schema definition for graders; bound as OpenAI tools for structured output.
- **LangSmith tracing**: every graph run is traced, so engineers can see which nodes executed and where loops fired.

## Why State Machines Enable Self-Correction

Loops are the missing primitive in vanilla RAG. A linear chain `Retrieve -> Generate` cannot recover from bad retrieval. A state machine can:

1. **Re-enter the retriever** with a re-written query.
2. **Combine multiple retrieval sources** (vector store + web) on a second pass.
3. **Reject a hallucinated generation** and try again.

This is the engineering counterpart to Self-RAG's reflection tokens: where Self-RAG asks the LM to emit a token that triggers re-retrieval, LangGraph asks the same question of an LLM-based grader in a conditional edge.

## In-Domain vs. Out-of-Domain Behavior

The blog post demonstrates two trace patterns:

- **In-domain question**: retrieval returns relevant docs -> grader passes them -> generator produces a faithful answer -> END. Fast, one-shot path.
- **Out-of-domain question**: retrieval returns irrelevant docs -> grader rejects all -> query is rewritten -> web search supplements -> generator uses the augmented context -> END. Slower, but still correct.

LangSmith traces make this divergence visible, which is critical for debugging production RAG systems.

## Authoritative Takeaways

- State machines enable loops essential for self-correction; chains cannot.
- Document grading (relevance) is the most impactful single step.
- Generation grading catches hallucinations and useless responses.
- Query rewriting improves retrieval on retry.
- Web search is a powerful fallback when local retrieval is insufficient.
- LangGraph provides the primitives for "flow engineering" with specific decision points and loops.

## Reference Implementations

- **Self-RAG cookbook**: `github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb`
- **CRAG cookbook**: `github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb`

Both notebooks are runnable end-to-end and serve as canonical implementations of agentic / self-reflective RAG.

## Related Work

- **Self-RAG** (Asai et al., 2023, arXiv:2310.11511) — the LM emits reflection tokens directly.
- **CRAG** (Yan et al., 2024, arXiv:2401.15884) — an external retrieval evaluator with web-search fallback.
- **Adaptive RAG** (Jeong et al., 2024) — adds a query-complexity classifier at the top of the graph.
- **Modular RAG** (Gao et al., 2023, arXiv:2312.10997) — the broader paradigm that places agentic patterns in context.

Agentic RAG as described here is **the orchestration layer** that makes Self-RAG and CRAG practical to build and deploy.
