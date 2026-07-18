# TruLens Evaluation Framework

> **来源**:
> - https://www.trulens.org/
> - https://github.com/truera/trulens (README)
> - https://www.trulens.org/getting_started/core_concepts/rag_triad/
>
> **抓取日期**: 2026-07-18
> **作者/机构**: TruEra (shepherded by Snowflake)

TruLens is an open-source evaluation and tracing framework for AI agents, developed by TruEra and shepherded by Snowflake. It helps developers objectively measure the quality and effectiveness of AI agent applications. The platform provides "scalable, trusted evals" with benchmarked metrics used by companies including Equinix, tribble.ai, KBC Group, and CubeServ.

> "Don't just vibe-check your LLM app! Systematically evaluate and track your LLM experiments with TruLens."

---

## Core Concepts

The framework is built around several key concepts:
- **Feedback Functions** - Customizable evaluation metrics
- **The RAG Triad** - Framework for evaluating retrieval-augmented generation
- **Honest, Harmless and Helpful Evals** - Ethical evaluation criteria

## Evaluation Metrics

TruLens provides a comprehensive set of metrics to measure AI agent performance:

- **Context Relevance** - measures how relevant the retrieved context is
- **Groundedness** - verifies claims against provided context
- **Answer Relevance** - assesses if responses directly address questions
- **Comprehensiveness** - evaluates coverage of all aspects
- **Harmful/toxic language detection**
- **User sentiment analysis**
- **Language mismatch detection**
- **Fairness and bias assessment**

## Evaluation Workflow

The framework follows an Evaluate, Iterate, and Test workflow:

1. **Evaluate** — Measure performance across multiple metrics
2. **Iterate** — Use built-in or custom metrics to identify weaknesses and inform prompt/hyperparameter improvements
3. **Test** — Compare different LLM apps on a metrics leaderboard

## Architecture

TruLens can work with any AI Agent via:

1. **Python SDK** — Direct integration into Python-based agent code
2. **OpenTelemetry ingestion** — For agents already emitting OTel traces

---

## The RAG Triad

The RAG Triad is TruEra's innovative framework for evaluating hallucinations in Retrieval-Augmented Generation (RAG) applications. It consists of three interconnected evaluations that work together to verify an LLM app is hallucination-free.

### 1. Context Relevance
The first evaluation checks whether retrieved context chunks are actually relevant to the user's query. Since this context forms the basis for the LLM's answer, irrelevant information could easily slip into and contaminate the response. TruLens evaluates this using the structure of the serialized record.

### 2. Groundedness
After retrieval, the LLM generates an answer from the context — but LLMs often stray from the provided facts, inflating or elaborating into plausible-sounding but incorrect statements. Groundedness verification works by breaking the response into individual claims and checking each one has supporting evidence within the retrieved context.

### 3. Answer Relevance
Finally, the generated response must actually address the original question. This metric evaluates whether the final answer is relevant to the user input.

### Putting It All Together

When all three metrics pass satisfactorily, you can make a confident statement about your application: **it's verified hallucination-free up to the limits of its knowledge base**. If your vector database contains only accurate information, then the RAG's answers will be accurate too.

---

## Agentic Evaluations

Seven purpose-built evaluators for agentic systems:

| Evaluator | What it Measures |
|-----------|------------------|
| `LogicalConsistency` | Reasoning coherence; flags hallucinations and unsupported assertions |
| `ExecutionEfficiency` | Redundant steps, unnecessary retries, wasted computation |
| `PlanAdherence` | Whether execution followed the stated plan |
| `PlanQuality` | Intrinsic plan quality — strategy, not outcome |
| `ToolSelection` | Right tool chosen for each subtask |
| `ToolCalling` | Argument validity and output interpretation |
| `ToolQuality` | External tool/service reliability |

---

## OpenTelemetry-based Tracing

TruLens instrumentation is built on OpenTelemetry. Every function call, LLM generation, retrieval, and tool invocation is captured as a structured OTEL span. This makes it interoperable with existing observability infrastructure — export traces to Jaeger, Grafana Tempo, Datadog, or any OTLP-compatible backend.

```python
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

class MyRAG:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        ...
```

---

## Installation

Install the core package:

```bash
pip install trulens-core
```

Install with specific LLM providers:

```bash
pip install trulens trulens-providers-openai     # OpenAI / Azure OpenAI
pip install trulens trulens-providers-litellm    # LiteLLM (Anthropic, Cohere, Mistral, …)
pip install trulens trulens-providers-google     # Google Gemini
pip install trulens trulens-providers-bedrock    # AWS Bedrock
pip install trulens trulens-providers-cortex     # Snowflake Cortex
pip install trulens trulens-providers-huggingface # HuggingFace
pip install trulens trulens-providers-langchain   # LangChain models
```

Install with app framework integrations:

```bash
pip install trulens trulens-apps-langchain      # LangChain / LangGraph
pip install trulens trulens-apps-llamaindex     # LlamaIndex
```

---

## Batch and Inline Evaluation

Run evaluations alongside your app, on existing data, or in offline batch mode:

```python
# Inline — evaluate as the app runs
with tru_recorder as recording:
    response = my_app.query("What is TruLens?")

# Batch — evaluate a pre-collected dataset using the TruLens 2.8 Run API
from trulens.core.run import RunConfig

run_config = RunConfig(
    run_name="batch_eval_v1",
    dataset_name="eval_questions",
    source_type="TABLE",
    dataset_spec={"input": "QUESTION"},
    invocation_max_workers=8,
    metric_max_workers=4,
)
run = tru_app.add_run(run_config=run_config)
run.start()
run.compute_metrics([relevance, groundedness])
```

---

## MCP Support

Instrument Model Context Protocol tool calls with the `MCP` span type:

```python
@instrument(span_type=SpanAttributes.SpanType.MCP)
def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
    ...
```

---

## Selector API

Target any span attribute for evaluation using the flexible Selector API:

```python
from trulens.core import Metric, Selector

f_context_relevance = Metric(
    name="Context Relevance",
    implementation=provider.context_relevance,
    selectors={
        "input": Selector.select_record_input(),
        "context": Selector.select_context(),
    },
)
```

---

## Supported LLM Providers

| Provider | Package |
|----------|---------|
| OpenAI / Azure OpenAI | `trulens-providers-openai` |
| LiteLLM (Anthropic, Cohere, Mistral, and more) | `trulens-providers-litellm` |
| Google Gemini | `trulens-providers-google` |
| AWS Bedrock | `trulens-providers-bedrock` |
| Snowflake Cortex | `trulens-providers-cortex` |
| HuggingFace | `trulens-providers-huggingface` |
| LangChain models | `trulens-providers-langchain` |

---

## Use Cases

- Agents
- Retrieval Augmented Generation (RAG)
- Summarization
- Co-pilots

## Value Proposition

1. **Interoperability** — Works with existing observability tools via OpenTelemetry
2. **Trusted evals** — Benchmark-backed metrics for reliable measurement
3. **Comprehensive metrics** — Breadth of built-in metrics plus custom metric support

---

## Resources

- **Documentation**: https://www.trulens.org/getting_started/
- **GitHub**: https://www.github.com/truera/trulens
- **Community**: https://snowflake.discourse.group/c/ai-research-and-development-community/trulens/97
- **Quickstart Colab**: Available for LangChain quickstart

TruLens has approximately 3.4k stars on GitHub and is actively maintained by Truera.
