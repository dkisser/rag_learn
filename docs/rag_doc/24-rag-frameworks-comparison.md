# LangChain vs LlamaIndex: A Comparison of RAG Frameworks

> **来源**:
> - https://docs.langchain.com/oss/python/langchain/overview
> - https://www.langchain.com/
> - https://developers.llamaindex.ai/python/framework/getting_started/concepts/
> - https://www.llamaindex.ai/
>
> **抓取日期**: 2026-07-18
> **作者/机构**: LangChain (LangChain, Inc.) and LlamaIndex (LlamaIndex)

This document compares the two dominant open-source frameworks for building LLM applications: **LangChain** and **LlamaIndex**. Both are widely used for Retrieval-Augmented Generation (RAG) pipelines, but they originate from different design philosophies and target different primary use cases.

---

## LangChain: An Overview

LangChain is a framework for building AI agents with a minimal, highly configurable harness. It enables developers to compose exactly the agent their use case needs from model, tools, prompt, and middleware components.

### Core Description

LangChain is positioned as an open-source framework for **"Quick start agents with any model provider"** and is described as **"For building agents fast with templates."**

### Key Features

- Enables building agents with any model provider
- Part of a suite of open-source frameworks offering **"batteries included"** approach
- Provides templates for rapid agent development
- Framework-agnostic tracing capabilities through LangSmith

### Design Philosophy

The fundamental equation is: **Agent = Model + Harness**. The harness encompasses everything around the model loop — the prompt, the tools, and any middleware that shapes behavior. Developers start with primitives and compose exactly what their use case requires.

The page positions LangChain as the **"quick start"** option compared to:
- **LangGraph**: For "low-level control" and "production agents that require some determinism"
- **Deep Agents**: For "highly autonomous, long-running agents"

LangChain is presented as the **entry-level framework** for developers who want to build AI agents quickly using pre-built templates, while more advanced use cases can leverage LangGraph for granular control or Deep Agents for complex, autonomous tasks.

### Architecture Details

- **Multi-Provider Support**: Works with OpenAI, Anthropic, Google, and many other providers
- **Standard Model Interface**: One interface for chat models, embeddings, and more across providers, allowing easy model switching with minimal code changes
- **Highly Configurable Harness**: Start with `create_agent` as a minimal harness and add capabilities incrementally through middleware — from guardrails and retries to routing and custom tool policies
- **Built on LangGraph**: LangChain's agents leverage LangGraph's durable execution, human-in-the-loop support, persistence, and other advanced features
- **Observability**: Integration with LangSmith for tracing, debugging, and evaluating agent behavior

### Primary Use Cases

- Custom agents with specific tools
- General-purpose assistants
- Multi-step agentic workflows
- Tool-using LLM applications
- Quick prototyping with templates

---

## LlamaIndex: An Overview

LlamaIndex is a framework for building applications with Large Language Models (LLMs). The framework emphasizes a structured data ingestion and retrieval pipeline, with deep native support for RAG.

### Core Description

LlamaIndex is described as:
- **Open-source software** with "OSS repos trusted by millions of developers"
- **Scale**: 25M+ package downloads per month
- Part of the **"agentic stack"** for building document processing AI agents
- Related products: **LlamaParse** (commercial), **LiteParse** (open-source local parser), **Workflows**, and the core **LlamaIndex library**

### Design Philosophy

- "Build document agents that understand, reason, and act"
- Enables end-to-end document automation
- Heavy emphasis on **data ingestion, indexing, and querying** primitives

### Key Concepts

#### Large Language Models (LLMs)

LLMs are the fundamental innovation that launched LlamaIndex. They are an artificial intelligence (AI) computer system that can understand, generate, and manipulate natural language, including answering questions based on their training data or data provided to them at query time.

#### Agentic Applications

When an LLM is used within an application, it is often used to make decisions, take actions, and/or interact with the world. Key characteristics:
- **LLM Augmentation**: The LLM is augmented with tools, memory, and/or dynamic prompts.
- **Prompt Chaining**: Several LLM calls build on each other.
- **Routing**: The LLM routes the application to the next appropriate step or state.
- **Parallelism**: Multiple steps or actions execute in parallel.
- **Orchestration**: A hierarchical structure of LLMs orchestrates lower-level actions.
- **Reflection**: The LLM reflects and validates outputs of previous steps.

In LlamaIndex, you build agentic applications by using the **`Workflow`** class to orchestrate a sequence of steps and LLMs.

#### Agents

An agent is a piece of software that semi-autonomously performs tasks by combining LLMs with other tools and memory, orchestrated in a reasoning loop that decides which tool to use next (if any).

What this means in practice:
- An agent receives a user message
- The agent uses an LLM to determine the next appropriate action
- The agent may invoke one or more tools
- Once the agent stops taking actions, it returns the final output

#### Retrieval Augmented Generation (RAG)

RAG is a **core technique** for building data-backed LLM applications with LlamaIndex. It allows LLMs to answer questions about your private data by providing it to the LLM at query time, rather than training the LLM on your data. RAG indexes your data and selectively sends only the relevant parts along with your query.

### Primary Use Cases

1. **Agents**: Automated decision-makers powered by an LLM that interact with the world via a set of tools.
2. **Workflows**: Event-driven abstractions for orchestrating a sequence of steps and LLM calls. Workflows are a core component of LlamaIndex.
3. **Structured Data Extraction**: Pydantic extractors allow you to specify a precise data structure to extract from your data and use LLMs to fill in the missing pieces in a type-safe way.
4. **Query Engines**: An end-to-end flow that allows you to ask questions over your data. Returns a response with reference context.
5. **Chat Engines**: An end-to-end flow for having a conversation with your data (multi-turn).

### Core Capabilities Demonstrated

- **Agent Creation**: Building agents that can perform tasks by calling tools
- **RAG**: Enhancing agents with the ability to search through documents
- **Vector Storage**: Using `VectorStoreIndex` with text embeddings (like `text-embedding-ada-002`) for document retrieval
- **Chat History**: Maintaining conversation context within agent workflows
- **Index Persistence**: Storing RAG indexes to disk to avoid reprocessing documents
- **Tool Integration**: Agents can use custom functions/tools
- **Document Loading**: Using `SimpleDirectoryReader` to load documents
- **Multiple LLM Support**: Works with OpenAI models and supports local models
- **Async Programming**: Recommended for improved performance

---

## Side-by-Side Comparison

| Aspect | LangChain | LlamaIndex |
|--------|-----------|------------|
| **Primary Focus** | Agent building with composable primitives | Data ingestion, indexing, and querying (RAG-first) |
| **Design Philosophy** | "Agent = Model + Harness" — composable, minimal core | "Document understanding" — comprehensive data pipeline |
| **Entry Point** | `create_agent` — quick start with templates | `VectorStoreIndex`, `Workflow`, agents |
| **Strength** | Flexibility, multi-provider support, agent ecosystem | Data connectors, indexing strategies, RAG-specific abstractions |
| **Companion Tools** | LangGraph (low-level control), LangSmith (observability) | LlamaParse (PDF parsing), Workflows, LlamaCloud |
| **Code Style** | LCEL (LangChain Expression Language) chain composition | Index/Query/Agent pattern |
| **Multi-Provider** | First-class: OpenAI, Anthropic, Google, etc. | Supported via adapters |
| **Document Loading** | Good, broad | Excellent, very extensive |
| **RAG Primitives** | General, requires assembly | First-class: indexes, retrievers, query engines, routers |
| **Agent Primitives** | First-class, multiple agent types | Solid, with `Workflow` class |
| **Observability** | LangSmith (tight integration) | Built-in + third-party |
| **Learning Curve** | Moderate (many abstractions) | Moderate (RAG-specific) |
| **Downloads** | Massive | 25M+/month |

---

## When to Use Which

### Choose **LangChain** if:

- You're building **agent-heavy** applications with tool use, planning, and reasoning loops
- You want **flexibility to compose** many LLM components into custom pipelines
- You need **first-class support for LangGraph** for production-grade, deterministic agents
- You want **LangSmith** for tracing and evaluation
- You're building **complex multi-step workflows** with branching logic
- Your application is more about **agents and tool use** than about document retrieval

### Choose **LlamaIndex** if:

- You're building **RAG-heavy applications** with sophisticated document ingestion needs
- You need **many document connectors** (PDFs, databases, APIs, Notion, Slack, etc.)
- You want **specialized indexing strategies** (vector, summary, knowledge graph, tree, keyword)
- You need **structured data extraction** from unstructured sources at scale
- You want **LlamaParse** for production PDF parsing
- You're building **query engines** over private knowledge bases
- Your application is more about **data retrieval and Q&A** than about agents

### Hybrid Approach

Many production systems use **both frameworks together**:
- LlamaIndex for **ingestion, indexing, and retrieval**
- LangChain (or LangGraph) for **agent orchestration and tool use**

They are not mutually exclusive — both expose clean Python interfaces and can interoperate.

---

## Code Style Comparison

### LangChain (Conceptual)

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")
agent = create_agent(model, tools=[my_tool])
response = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

### LlamaIndex (Conceptual)

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("...")
```

---

## Summary

| Criterion | LangChain | LlamaIndex |
|-----------|-----------|------------|
| Agent building | ★★★★★ | ★★★★ |
| RAG primitives | ★★★★ | ★★★★★ |
| Document loading | ★★★★ | ★★★★★ |
| Flexibility | ★★★★★ | ★★★★ |
| Production readiness | ★★★★★ | ★★★★★ |
| Onboarding speed | ★★★★ (with templates) | ★★★★ |
| Community size | Very large | Very large |
| Best for | Agentic apps, custom pipelines | Document Q&A, knowledge bases |

Both frameworks are mature, production-ready, and actively maintained. The choice depends on whether your application's center of gravity is **agent orchestration** (LangChain) or **document retrieval** (LlamaIndex).
