# Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models

> **来源**: https://arxiv.org/abs/2310.06117
> **抓取日期**: 2026-07-18
> **作者/机构**: Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V Le, Denny Zhou — Google DeepMind

## Abstract

We present Step-Back Prompting, a simple prompting technique that enables LLMs to do abstractions to derive high-level concepts and first principles from instances containing specific details. Using the concepts and principles to guide reasoning, LLMs significantly improve their abilities in following a correct reasoning path towards the solution. We conduct experiments of Step-Back Prompting with PaLM-2L, GPT-4 and Llama2-70B models, and observe substantial performance gains on various challenging reasoning-intensive tasks including STEM, Knowledge QA, and Multi-Hop Reasoning. For instance, Step-Back Prompting improves PaLM-2L performance on MMLU (Physics and Chemistry) by 7% and 11% respectively, TimeQA by 27%, and MuSiQue by 7%.

> The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise. — Edsger W. Dijkstra

## 1 Introduction

The field of natural language processing (NLP) is witnessing a ground-breaking revolution because of Transformer-based large language models (LLMs). Scaling up model size and pre-training corpus has brought remarkable improvement in model capabilities and sample efficiency with insights from the scaling law, as well as emergent abilities such as multi-step reasoning and instruction following.

Despite the great advancements, complex multi-step reasoning remains challenging for even the state-of-the-art LLMs. Process-supervision with step-by-step verification is a promising remedy to improve the correctness of intermediate reasoning steps. Techniques such as Chain-of-Thought were introduced to produce a coherent series of intermediate reasoning steps to increase the success rate of following the right decoding path. Inspired by the fact that when faced with challenging tasks humans often step back and do abstractions to arrive at high-level principles to guide the process, we propose Step-Back Prompting to ground reasoning on abstractions to reduce the chance of making errors in the intermediate reasoning steps.

Among many of the cognitive skills, abstraction is ubiquitous to humans’ ability to process vast amounts of information and derive general principles. For example, Kepler compressed thousands of measurements into Kepler’s three laws of planetary motion, which precisely describe the orbits of planets around the Sun. In critical decision-making, humans find abstraction to be helpful since it provides a broader view of the environment. This work explores how LLMs can tackle complex tasks involving many low-level details through a two-step process of abstraction-and-reasoning.

The first step is to show LLMs how to step back through in-context learning—prompting them to derive high-level abstractions such as concepts and principles for a specific example. The second step is to leverage the reasoning ability to reason on top of the high-level concepts and principles. We use few-shot exemplar demonstrations to execute Step-Back Prompting on LLMs.

We experiment across a range of tasks involving domain-specific reasoning such as Physics and Chemistry, knowledge-intensive question answering requiring factual knowledge, and multi-hop commonsense reasoning. We observe significant performance improvements (up to 27%) in PaLM-2L, demonstrating the efficacy of Step-Back Prompting in tackling complex tasks, which are otherwise challenging due to the amount of details needed for reasoning. Some tasks are very challenging: both PaLM-2L and GPT-4 achieve only approximately 40% accuracy on TimeQA and MuSiQue. Chain-of-Thought prompting leads to a minor improvement on a few tasks, while Step-Back Prompting improves the performance of PaLM-2L across the board: 7% and 11% on MMLU Physics and Chemistry, 27% on TimeQA, and 7% on MuSiQue.

We conduct a variety of analyses and find that Step-Back Prompting leads to strong performance improvements (up to 36%) over chain-of-thought (CoT) prompting and “take-a-deep-breath” (TDB) prompting. We perform a qualitative evaluation where we find that Step-Back fixes a large portion of errors of the base model (up to approximately 40%) while introducing a small portion of new errors (max approximately 12%). We also conduct an error analysis and find that majority of the errors made by Step-Back Prompting is attributed to the intrinsic limitations of reasoning capabilities of LLMs while abstraction skills are relatively easy to demonstrate to LLMs, pointing out the direction for future improvements of methods alike Step-Back Prompting.

## Figure 2: Abstraction and reasoning

The figure illustrates Step-Back Prompting with two steps of Abstraction and Reasoning guided by concepts and principles.

For a physics example, the original question is:

> What happens to the pressure, P, of an ideal gas if the temperature is increased by a factor of 2 and the volume is increased by a factor of 8?

The step-back question is:

> What are the physics principles behind this question?

The step-back answer retrieves the ideal gas law:

> Ideal gas law: $PV=nRT$, where $P$ is the pressure, $V$ is the volume, $n$ is the number of moles, $R$ is the gas constant, and $T$ is the temperature.

The final answer applies the ideal gas law. If the temperature is increased by a factor of 2, then $T$ becomes $2T$. If the volume is increased by a factor of 8, then $V$ becomes $8V$:

$$
P(8V)=nR(2T).
$$

Dividing both sides by $8V$, the pressure has decreased by a factor of 4.

For a TimeQA example, the original question is:

> Estella Leopold went to which school between Aug 1954 and Nov 1954?

The step-back question is:

> What was Estella Leopold’s education history?

The step-back answer gives the relevant history:

- B.S. in Botany, University of Wisconsin, Madison, 1948
- M.S. in Botany, University of California, Berkeley, 1950
- Ph.D. in Botany, Yale University, 1955

She was enrolled in the Ph.D. program in Botany at Yale from 1951 to 1955. Therefore, Estella Leopold was most likely attending Yale University between August 1954 and November 1954.

## 2 Step-Back Prompting

Step-Back Prompting is motivated by the observation that many tasks contain a lot of details, and it is hard for LLMs to retrieve relevant facts to tackle the task. For a Physics question about what happens to the pressure of an ideal gas when temperature and volume change, the LLM can deviate from the first principle of Ideal Gas Law when reasoning directly on the question. Similarly, a question of “Estella Leopold went to which school between Aug 1954 and Nov 1954?” is very hard to address directly given the detailed time range constraint. In both cases, asking a step-back question helps the model to solve the problem effectively.

We define a step-back question as a derived question from the original question at a higher level of abstraction. For instance, instead of directly asking “which school Estella Leopold went to during a specific period”, a step-back question would ask about the “education history”, which is a high-level concept that encompasses the original question. Answering the step-back question of “Estella Leopold’s education history” in this case will provide all the necessary information to reason about “which school Estella Leopold went to during a specific period”. The premise is that the step-back question is typically much easier. Grounding the reasoning on top of such abstractions helps to avoid reasoning errors in the intermediate steps from Chain-of-Thought.

In short, Step-Back Prompting consists of two simple steps:

- **Abstraction:** Instead of addressing the question directly, we first prompt the LLM to ask a generic step-back question about a higher-level concept or principle, and retrieve relevant facts about the high-level concept or principle. The step-back question is unique for each task in order to retrieve the most relevant facts.
- **Reasoning:** Grounded on the facts regarding the high-level concept or principle, the LLM can reason about the solution to the original question. We term this as Abstraction-grounded Reasoning.

## 3 Experimental Setup

### Tasks

We experiment with the following diverse tasks:

- **STEM:** We evaluate MMLU and GSM8K for STEM tasks. MMLU contains a series of benchmarks across diverse domains to evaluate the model’s language understanding. We consider the high school physics and chemistry portions of MMLU because of the deep reasoning involved.
- **Knowledge QA:** We consider TimeQA since it contains complex queries that require challenging time-sensitive knowledge. We also experiment with SituatedQA, another challenging open-retrieval QA dataset requiring the model to answer questions given temporal or geographical contexts.
- **Multi-Hop Reasoning:** We experiment with MuSiQue, a hard multihop reasoning dataset created via composable pairs of single-hop questions, and StrategyQA with open-domain questions that demand some strategy to solve.

### Models

We use instruction-tuned PaLM-2L, GPT-4, and Llama2-70B.

### Evaluation

Conventional evaluation metrics such as accuracy and F1 score have limitations specifically for evaluating the generations of state-of-the-art LLMs since these models often generate long-form answers which are hard to capture. We instead conduct an evaluation using the PaLM-2L model where we few-shot prompt the model to identify equivalence between target answers and the model predictions.

### Baseline methods

- **PaLM-2L, PaLM-2L 1-shot:** PaLM-2L is either queried directly with the question or has a single demonstration exemplar of question-answer included in the prompt.
- **PaLM-2L + CoT, PaLM-2L + CoT 1-shot:** PaLM-2L model is queried with zero-shot CoT prompting: “Let’s think step by step” is appended to the question. For 1-shot, one demonstration example of a question and answer pair is provided in the prompt, where the answer is in the style of CoT.
- **PaLM-2L + TDB:** Zero-shot prompting with “Take a deep breath and work on this problem step-by-step.” prepended to the question.
- **PaLM-2L + RAG:** For Knowledge QA and Multi-Hop Reasoning, we use retrieval-augmented generation (RAG) where the retrieved passage is used as context by the LLM.
- **GPT-4 and Llama2-70B:** We run GPT-4 and Llama2-70B on MMLU tasks for all methods. In addition, we also run GPT-4 on all baselines for all tasks.

We do not use RAG for STEM tasks, because of the inherent reasoning nature of the tasks contrary to the other fact-seeking datasets. All inferences are done using greedy decoding.

## 4 STEM

Questions in the MMLU benchmarks require deeper reasoning. Furthermore, they also require understanding and application of formulae which are often physics and chemistry principles and concepts. In this case, we first demonstrate to the model abstraction skills in the form of concepts and first principles such as Newton’s first law of motion, Doppler effect, and Gibbs free energy. The implicit step-back question here is “what are the physics or chemistry principles and concepts involved in solving this task?”. We provide demonstrations to the model to recite the relevant principles for solving the task from its own knowledge.

### MMLU results

| Method | MMLU Physics | MMLU Chemistry |
|---|---:|---:|
| PaLM-2L | 66.4% | 70.9% |
| PaLM-2L 1-shot | 64.0% | 75.6% |
| PaLM-2L + CoT | 65.0% | 75.3% |
| PaLM-2L + CoT 1-shot | 61.5% | 76.6% |
| PaLM-2L + TDB | 65.7% | 73.8% |
| **PaLM-2L + Step-Back** | **73.2%** | **81.8%** |
| GPT-4 | 69.4% | 80.9% |
| GPT-4 + CoT | 82.9% | 85.3% |
| **GPT-4 + Step-Back** | **84.5%** | **85.6%** |
| Llama2-70B | 51.9% | 55.7% |
| Llama2-70B + CoT | 59.3% | 64.1% |
| **Llama2-70B + Step-Back** | **64.8%** | **66.7%** |

PaLM-2L baseline performance is 66.4% and 70.9% on Physics and Chemistry, respectively. We find that CoT and TDB zero-shot prompting do not significantly increase model performance, which could be due to the inherent difficulty and deep reasoning associated with these tasks. In contrast, Step-Back Prompting significantly improves model performance: +7% and +11% compared to PaLM-2L. Similarly, with GPT-4 and Llama2-70B models, Step-Back Prompting is very competitive among all the baseline methods we tested, showing that Step-Back Prompting is model-agnostic.

### Ablation and error analysis

Step-Back Prompting is robust to the number of few-shot exemplars of `(question, principles)` pairs used as demonstrations. Adding more demonstration examples beyond a single example does not lead to further improvements. This indicates that the task of retrieving the relevant principles and concepts is relatively easy through in-context learning and a single demonstration suffices.

Comparing the predictions of Step-Back Prompting to the baseline PaLM-2L model for MMLU high-school Physics, we find that Step-Back Prompting corrects 20.5% errors from the baseline while introducing 11.9% errors.

The wrong predictions are categorized into five classes:

- **Principle Error:** The error happens at the step of Abstraction, where the first principles generated by models are wrong or incomplete.
- **Factual Error:** There is at least one factual error when the model recites its own factual knowledge.
- **Math Error:** There is at least one math error in the intermediate steps when math calculations are involved in deriving the final answer.
- **Context Loss:** There is at least one error where the model response loses context from the question, and deviates from addressing the original question.
- **Reasoning Error:** The model makes at least one error in the intermediate Reasoning steps before arriving at the final answer.

All five types of errors are happening during the Reasoning step except Principle Error which points to the failure of the Abstraction step. Principle Error comprises only a small fraction of the errors the model makes: more than 90% of the errors happen at the Reasoning step. Reasoning Error and Math Error are the major error categories. Reasoning is still the bottleneck of how well Step-Back Prompting can perform tasks such as MMLU requiring complex reasoning.

## 5 Knowledge QA

We evaluate Step-Back Prompting on TimeQA and SituatedQA. We first show the LLMs how to do Abstraction through in-context demonstrations. Given the knowledge-intensive nature of these queries, we use retrieval augmentation (RAG) in combination with Step-Back Prompting. The step-back question is used to retrieve relevant facts, which work as additional context to ground the final reasoning step.

| Method | TimeQA | TQA Easy | TQA Hard | SituatedQA |
|---|---:|---:|---:|---:|
| PaLM-2L | 41.5% | 42.6% | 40.4% | 54.3% |
| PaLM-2L + CoT | 40.8% | 41.8% | 39.8% | 56.4% |
| PaLM-2L + TDB | 40.9% | 42.6% | 39.1% | 54.0% |
| PaLM-2L + RAG | 57.4% | 67.8% | 46.8% | 59.3% |
| PaLM-2L + Step-Back | 66.0% | 70.4% | 61.6% | 57.5% |
| **PaLM-2L + Step-Back + RAG** | **68.7%** | **75.2%** | **62.3%** | **61.0%** |
| GPT-4 | 45.6% | 48.9% | 42.6% | 63.2% |

The result of Step-Back + RAG shows the effectiveness of going back to a high-level concept, which enables much more reliable retrieval augmentation: the accuracy on TimeQA achieves a remarkable 68.7%. While RAG can improve the Easy accuracy from 42.6% to 67.8%, the improvement is much smaller on the Hard accuracy: 40.4% to 46.8%. This is where Step-Back Prompting shines by retrieving facts regarding high-level concepts to ground the final reasoning: Step-Back + RAG further improves the Hard accuracy to 62.3%.

## 6 Multi-Hop Reasoning

| Method | MuSiQue | StrategyQA |
|---|---:|---:|
| PaLM-2L | 35.5% | 82.8% |
| PaLM-2L + CoT | 38.7% | 83.6% |
| PaLM-2L + TDB | 39.0% | 82.7% |
| PaLM-2L + RAG | 39.6% | 84.2% |
| PaLM-2L + Step-Back | 42.6% | 82.7% |
| **PaLM-2L + Step-Back + RAG** | **42.8%** | **86.4%** |
| GPT-4 | 38.5% | 78.3% |

Step-Back Prompting with the power of abstraction produces the best performance of all methods: 42.8% in MuSiQue and 86.4% in StrategyQA, significantly outperforming GPT-4 on both tasks.

## 7 Discussion

Abstraction helps humans to solve complex tasks by removing irrelevant details and distilling high-level concepts and principles to guide the problem-solving process. Step-Back Prompting breaks complex tasks such as knowledge-intensive QA, multi-hop reasoning, and science questions into two separate steps of Abstraction and Reasoning. We demonstrate through empirical experiments that Abstraction is an easy skill for the LLMs such as PaLM-2L via sample-efficient in-context learning. Grounding on the high-level concepts and principles, LLMs can leverage their intrinsic Reasoning capabilities to derive the solution. This reduces the chance of reasoning failures in the intermediate steps and is shown to improve the performance on a wide range of complex reasoning tasks. Despite the success, through error analysis, we find that Reasoning is still one of the hardest skills for LLMs to acquire: it is still the dominant failure mode even after the large reduction of task complexity by Step-Back Prompting.

Nevertheless, Abstraction is neither necessary nor possible in all scenarios. For instance, the task can be as simple as “who was the president of the United States in 2000?”, in which case there is no such need to step back and ask a high-level question as the answer to such questions is readily available. Questions such as “what is the speed of light?” point to the first principles themselves. Doing Abstraction in this case would not make a difference either.

Step-Back Prompting is focused on the key idea of abstraction. This differs from decomposition, which is often a low-level breakdown of the original question. For instance, a generic question for “which employer did Steve Jobs work for in 1990?” could be “what is the employment history of Steve Jobs?”. Decomposition would instead lead to sub-questions such as “What was Steve Jobs doing in 1990?”, “Was Steve Jobs employed in 1990?”, and “If Steve Jobs was employed, who was his employer?”. Furthermore, abstract questions are often generic in nature and have a many-to-one mapping, in contrast to decomposition where there is often a one-to-many mapping since there are multiple decomposed sub-problems necessary to solve a given question.

## 9 Conclusion

We introduce Step-Back Prompting as a simple yet generic method to elicit deep reasoning via abstraction in large language models. Experimentation on LLMs across fact-seeking, commonsense reasoning and domain-specific reasoning benchmarks shows that Step-Back Prompting significantly improves model performance. We hypothesize that abstraction helps models to hallucinate less and reason better, probably reflecting the true nature of the model which are often hidden while responding to the original question without abstraction. We hope our work will inspire more human-inspired approaches to elicit the hidden potential of large language models.
