# RAGAS Evaluation Metrics

> **来源**:
> - https://docs.ragas.io/en/latest/concepts/metrics/overview/
> - https://docs.ragas.io/en/latest/concepts/metrics/available%5Fmetrics/
> - https://docs.ragas.io/en/latest/concepts/metrics/available%5Fmetrics/faithfulness
> - https://docs.ragas.io/en/latest/concepts/metrics/available%5Fmetrics/context_precision
> - https://docs.ragas.io/en/latest/concepts/metrics/available%5Fmetrics/context_recall
> - https://docs.ragas.io/en/latest/concepts/metrics/available%5Fmetrics/answer_relevance
> - https://docs.ragas.io/en/latest/concepts/metrics/available%5Fmetrics/noise_sensitivity
>
> **抓取日期**: 2026-07-18
> **作者/机构**: Ragas (vibrantlabsai)

Ragas (Retrieval Augmented Generation Assessment) is an evaluation framework for your AI Application. Its metrics module provides a comprehensive set of tools to objectively measure the quality of LLM-based applications, with a strong focus on RAG pipelines.

---

## Why Metrics Matter

Metrics provide the feedback loop needed for iteration in AI systems. They quantify improvement, detect regressions, and align optimization with user impact and business value.

## Types of Metrics in AI Applications

1. **End-to-End Metrics**: Evaluate overall system performance from user perspective (e.g., answer correctness, citation accuracy)
2. **Component-Level Metrics**: Assess individual parts independently (e.g., retrieval accuracy)
3. **Business Metrics**: Align with organizational objectives (e.g., ticket deflection rate)

## Types of Metrics in Ragas

**By Mechanism:**
- **LLM-based metrics**: Use an LLM for evaluation; more accurate but non-deterministic; inherit from `MetricWithLLM` class
- **Non-LLM-based metrics**: Use traditional methods (string similarity, BLEU); deterministic; inherit from `Metric` class

**By Interaction:**
- **Single turn metrics**: Evaluate one interaction; inherit from `SingleTurnMetric`
- **Multi-turn metrics**: Evaluate multiple interactions; inherit from `MultiTurnMetric`

## Output Types

1. **Discrete Metrics**: Return categorical values (pass/fail)
2. **Numeric Metrics**: Return integer/float in range (0-1); support aggregation
3. **Ranking Metrics**: Return ranked list of outputs

## Metric Design Principles

1. **Single-Aspect Focus**: Target one specific aspect
2. **Intuitive and Interpretable**: Easy to understand
3. **Effective Prompt Flows**: Align with human evaluation
4. **Robustness**: Include sufficient few-shot examples
5. **Consistent Scoring Ranges**: Normalize to 0-1 range

## Choosing the Right Metrics

- Prioritize end-to-end metrics reflecting user satisfaction
- Ensure interpretability across the team
- Emphasize objective over subjective metrics (look for ≥80% inter-rater agreement)
- Use few strong signals over many weak signals

---

## Available Metric Categories

### Retrieval Augmented Generation
- Context Precision
- Context Recall
- Context Entities Recall
- Noise Sensitivity
- Response Relevancy
- Faithfulness
- Multimodal Faithfulness
- Multimodal Relevance

### Nvidia Metrics
- Answer Accuracy
- Context Relevance
- Response Groundedness

### Agents or Tool Use Cases
- Topic adherence
- Tool call Accuracy
- Tool Call F1
- Agent Goal Accuracy

### Natural Language Comparison
- Factual Correctness
- Semantic Similarity
- Non LLM String Similarity
- BLEU Score
- CHRF Score
- ROUGE Score
- String Presence
- Exact Match

### SQL
- Execution based Datacompy Score
- SQL query Equivalence

### General Purpose
- Aspect critic
- Simple Criteria Scoring
- Rubrics based scoring
- Instance specific rubrics scoring

---

## Faithfulness

**Definition**

Faithfulness measures how factually consistent a response is with the retrieved context. It ranges from 0 to 1, with higher scores indicating better consistency.

A response is considered faithful if all its claims can be supported by the retrieved context.

**Formula**

```
Faithfulness Score = (Number of claims in the response supported by the retrieved context) / (Total number of claims in the response)
```

**Calculation Steps**

1. **Identify claims**: Break the response into individual statements
2. **Verify each claim**: Check if each statement can be inferred from the context
3. **Compute score**: Divide supported claims by total claims

**Example: Einstein**

- **Context**: "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist..."
- **High faithfulness**: "Einstein was born in Germany on 14th March 1879." → Score: 1.0
- **Low faithfulness**: "Einstein was born in Germany on 20th March 1879." → Score: 0.5 (1/2 claims supported)

**Code Examples**

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
scorer = Faithfulness(llm=llm)

result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    retrieved_contexts=["The First AFL–NFL World Championship Game was played on January 15, 1967..."]
)
```

With HHEM-2.1-Open (Vectara's free open-source hallucination classifier):

```python
from ragas.metrics import FaithfulnesswithHHEM
scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
# Optional: custom device and batch size
scorer = FaithfulnesswithHHEM(device="cuda:0", batch_size=10)
```

---

## Context Precision

**Definition**

A metric evaluating the retriever's ability to rank relevant chunks higher than irrelevant ones for a given query. It measures how well relevant chunks are placed at the top of the ranking.

**Formulas**

- **Context Precision@K**: Mean of precision@k for each chunk
- **Precision@k**: (true positives@k) / (true positives@k + false positives@k)

**Key Variants**

1. **ContextPrecision** - Uses reference answer to evaluate retrieved contexts
2. **ContextUtilization** - Uses generated response instead of reference
3. **NonLLMContextPrecisionWithReference** - Uses Levenshtein distance (requires rapidfuzz package)
4. **IDBasedContextPrecision** - Compares context IDs directly (range 0-1)

**Behavior**

Irrelevant chunks at position 1 significantly reduce the score (to ~0.5), while irrelevant chunks at position 2 maintain high scores (~1.0). Position matters significantly for this metric.

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
scorer = ContextPrecision(llm=llm)

result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    reference="The Eiffel Tower is located in Paris.",
    retrieved_contexts=[
        "The Eiffel Tower is located in Paris.",
        "The Brandenburg Gate is located in Berlin."
    ]
)
print(f"Context Precision Score: {result.value}")
# Output: 0.9999999999
```

---

## Context Recall

**Definition**

Context Recall measures how many relevant documents or pieces of information were successfully retrieved. The focus is on not missing important results — higher recall means fewer relevant documents were left out.

**Formula**

```
Context Recall = (Number of claims in reference supported by retrieved context) / (Total number of claims in reference)
```

Since it requires a reference to compare against, the metric uses `reference` as a proxy. The reference is broken down into claims, and each claim is analyzed to determine whether it can be attributed to the retrieved context.

### LLM-Based Context Recall

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
scorer = ContextRecall(llm=llm)

result = await scorer.ascore(
    user_input="Where is the Eiffel Tower located?",
    retrieved_contexts=["Paris is the capital of France."],
    reference="The Eiffel Tower is located in Paris."
)
print(f"Context Recall Score: {result.value}")
# Output: Context Recall Score: 1.0
```

### Non-LLM Context Recall

Uses string comparison to identify if retrieved contexts are relevant. Range: 0 to 1.

```
context recall = |Number of relevant contexts retrieved| / |Total number of reference contexts|
```

### ID-Based Context Recall

Compares IDs of retrieved contexts with reference context IDs. Useful when documents have unique IDs.

```
ID-Based Context Recall = (Number of reference context IDs found in retrieved context IDs) / (Total number of reference context IDs)
```

---

## Response Relevancy (Answer Relevancy)

**Definition**

The Answer Relevancy metric measures how well a response addresses the user's question, scored from 0 to 1. Higher scores indicate better alignment with the original question's intent. The metric focuses on relevance without evaluating factual accuracy.

**Calculation Steps**

1. Generate artificial questions (default: 3) from the response using an LLM
2. Compute cosine similarity between the user input embedding and each generated question embedding
3. Average all similarity scores

**Formula**

```
Answer Relevancy = (1/N) × Σ cosine_similarity(E_g_i, E_o)
```

Where:
- `E_g_i` = embedding of the ith generated question
- `E_o` = embedding of the original user input
- N = number of generated questions (default 3, configurable via `strictness` parameter)

The underlying principle is that "if the answer correctly addresses the question, it is highly probable that the original question can be reconstructed solely from the answer."

The metric penalizes responses that are:
- Incomplete
- Include unnecessary details
- Don't directly address the question intent

**Code Example**

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerRelevancy

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)

result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967"
)
print(f"Answer Relevancy Score: {result.value}")
# Output: Answer Relevancy Score: 0.9165088378587264
```

While scores typically fall between 0 and 1, they aren't guaranteed due to cosine similarity's mathematical range of -1 to 1.

---

## Noise Sensitivity

**Definition**

NoiseSensitivity measures how often an AI system makes errors by providing incorrect responses when utilizing either relevant or irrelevant retrieved documents. The score ranges from 0 to 1, with **lower values indicating better performance**.

It is computed using: `user_input`, `reference`, `response`, and `retrieved_contexts`.

**Formula**

```
noise sensitivity (relevant) = |Total number of incorrect claims in response| / |Total number of claims in the response|
```

**Calculation Steps**

1. **Identify relevant contexts** from which the ground truth can be inferred
2. **Verify** if the claims in the generated answer can be inferred from the relevant context
3. **Identify incorrect claims** in the answer (statements not supported by ground truth)
4. **Calculate** noise sensitivity using the formula

**Code Example**

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import NoiseSensitivity

client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

scorer = NoiseSensitivity(llm=llm)

result = await scorer.ascore(
    user_input="What is the Life Insurance Corporation of India (LIC) known for?",
    response="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.",
    reference="The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.",
    retrieved_contexts=[
        "The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.",
        "LIC is the largest insurance company in India, with a vast network of policyholders and huge investments.",
        "As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.",
        "The Indian economy is one of the fastest-growing major economies in the world, thanks to sectors like finance, technology, manufacturing etc."
    ]
)

print(f"Noise Sensitivity Score: {result.value}")
# Output: Noise Sensitivity Score: 0.3333333333333333
```

To calculate noise sensitivity of irrelevant context, set the `mode` parameter to `"irrelevant"`:

```python
scorer = NoiseSensitivity(llm=llm, mode="irrelevant")
```

**Worked Example**

- **Question**: What is the Life Insurance Corporation of India (LIC) known for?
- **Ground Truth**: The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.
- **Answer**: The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.
- **Analysis**: 3 total claims in the answer; 1 incorrect claim ("LIC contributes to the financial stability of the country" — not mentioned in ground truth); Noise sensitivity = 1/3 = 0.333.

**Key Points**

- Lower scores are better — indicates fewer incorrect claims
- Mode parameter defaults to "relevant", can be set to "irrelevant"
- Origin: introduced in RAGChecker (Amazon Science)

---

## Installation & Quick Start

```bash
pip install ragas
# or from source
pip install git+https://github.com/vibrantlabsai/ragas
```

```python
import asyncio
from openai import AsyncOpenAI
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory

client = AsyncOpenAI()
llm = llm_factory("gpt-4o", client=client)

metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if the summary is accurate and captures key information.
Response: {response}
Answer with only 'accurate' or 'inaccurate'."""
)

async def main():
    score = await metric.ascore(
        llm=llm,
        response="The summary of the text is..."
    )
    print(f"Score: {score.value}")
    print(f"Reason: {score.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```
