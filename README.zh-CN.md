# rag-learn

> 🌐 **语言**: [English](./README.md) · **中文（当前）**

> **用数据说话，不用技巧堆叠** —— 一个基于 Chroma 的渐进式 RAG 学习工程，每一次优化都从评估报告里的数字出发。
>
> Learn RAG by measuring, not by tricks — a progressive Chroma learning project, where every optimization starts from a number on an eval report.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Gradio 5](https://img.shields.io/badge/gradio-5-orange.svg)](https://gradio.app/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![uv-ready](https://img.shields.io/badge/uv-ready-purple.svg)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-make%20all-brightgreen.svg)](#tests)
[![Coverage 80%+](https://img.shields.io/badge/coverage-80%25%2B-success.svg)](#tests)

```
Question ─▶ Catalog ─▶ Hybrid (BM25 + 向量, RRF) ─▶ Reranker (cross-encoder)
                                                          │
                                                          ▼
Gradio UI ◀── DeepSeek 流式回答 ◀── top-k 过滤 ◀── Chroma 存储
   │
   └─▶ JSONL 评估事件  (rag_events_YYYY-MM-DD.jsonl)
```

## 为什么做这个项目

RAG 在生产里"突然不灵"的频率，远高于"加了 hybrid / reranker / routing 之后立刻变好"的频率。本项目不是又一个技巧清单，而是一个**以评估闭环为骨架**的学习工程：

- **RAG 不会因为你换了 retriever 就自动变聪明** —— 它会变的前提是：你有 ground truth、能跑指标、能看到差异。
- 每一段新增能力（hybrid、rerank、routing、阈值过滤……）都伴随一组评估指标；**没有数字的优化不进主干**。
- demo 之外还配一个 CSV 驱动的批量评估 CLI（`sample / run / evaluate`），supervised + LLM-judge 两路指标都齐。

适合这样的人：

- 想学 RAG 但被 "hybrid + reranker + routing + ..." 技巧清单淹没的人。
- 想看 LLM-judge 与 supervised 指标在真实闭环里怎么用的人。
- 想用 Chroma 起步、逐步加能力、每次都有数字依据的人。

## 功能特性

- **Chroma + 多集合目录（Catalog）**：在 Gradio 下拉切换知识库。
- **可选混合检索**：BM25（jieba 中文分词）+ 向量，RRF 融合（`HYBRID_ENABLED`）。
- **可选 Cross-encoder reranker**：默认 BGE，可配 score threshold。
- **Intent-aware 路由 + 子查询分解**：自动 fan-out 到多个集合（`INTENT_ENABLED` / `DECOMPOSE_ENABLED`）。
- **检索阈值过滤**：`CHROMA_MAX_DISTANCE` 与 `RERANK_MIN_SCORE`。
- **完整埋点**：每次问答写入 `data/rag_events_YYYY-MM-DD.jsonl`。
- **CSV 驱动批量评估 CLI**：`sample / run / evaluate` 三个子命令。
- **Supervised 指标**：`retrieval_recall@k / precision@k / MRR / NDCG@k / answer_f1`。
- **LLM-judge 指标**：`context_relevance / faithfulness / answer_relevance / overall_usefulness / answer_llm_correctness`。
- **Token-bucket 限流 + 断点续跑**（DeepSeek 免费档友好）。
- **DeepSeek 流式回答**（OpenAI 兼容 SDK）。
- **80%+ 单测覆盖**（`make all` 强制门禁）。

## 目录

- [快速开始](#快速开始)
- [它是怎么工作的](#它是怎么工作的)
- [演化路径：渐进式之旅](#演化路径渐进式之旅)
- [评估方法论](#评估方法论)
- [环境变量](#环境变量)
- [测试](#测试)
- [批量评估 CLI](#批量评估-cli)
- [历史 adapter：Milvus](#历史-adaptermilvus)
- [Roadmap 与已知限制](#roadmap-与已知限制)
- [致谢](#致谢)
- [License](#license)

## 快速开始

```bash
# 方案 A — uv（推荐；使用已 commit 的 uv.lock）
uv sync --extra dev
cp .env.example .env
# 编辑 .env 设置 DEEPSEEK_API_KEY
uv run python main.py
# 浏览器打开 http://127.0.0.1:7860

# 方案 B — pip
pip install -e ".[dev]"
cp .env.example .env
# 编辑 .env 设置 DEEPSEEK_API_KEY
python main.py
# 浏览器打开 http://127.0.0.1:7860
```

第一次启动会自动摄入 `docs/shanzhongshi/*.md` 到 `data/chroma/`，下拉框里会出现"山中事咖啡"集合。embedder 模型首次下载后缓存。

## 它是怎么工作的

1. **摄入**：首次启动时 `docs/shanzhongshi/*.md` 被切片写入 Chroma `PersistentClient`（bundled `all-MiniLM-L6-v2` 384-dim cosine）。
2. **选集合**：UI 顶部"知识库"下拉选当前 catalog（单集合 / 多集合扇出由 routing 决定）。
3. **检索（可选链路）**：catalog → hybrid (BM25 + 向量, RRF) → reranker (cross-encoder) → threshold filter。
4. **生成**：DeepSeek 流式回答；折叠面板展示 chunks（file + chunk-index + 距离/分数）+ perf 行（retrieve / first-token / total）。
5. **埋点**：每次问答同步写一行 `RAGEvent` 到 `data/rag_events_YYYY-MM-DD.jsonl`（question / hits / answer / perf / metadata），给后续离线评估消费。

任何步骤的开关都在 [环境变量](#环境变量) 里；默认全关，得到的就是最朴素的"向量召回 → DeepSeek"。

## 演化路径：渐进式之旅

本项目按"每加一项能力，先有评估"的顺序演进。时间线真实来源于 `git log` 与 `docs/superpowers/specs/`，不是事后整理的目录：

| 阶段 | 主题 | 关键能力 | 设计稿 |
|---|---|---|---|
| **v0.1** | Vanilla 检索 | Chroma 单集合、DeepSeek 流式、perf 埋点 | [`2026-07-18-rag-multiretriever-design`](docs/superpowers/specs/2026-07-18-rag-multiretriever-design.md) |
| **v0.2** | Evaluation 闭环 | `RAGEvent` 持久化、5 supervised + 5 LLM-judge 指标、`batch` 聚合 | [`2026-07-22-rag-metrics-design`](docs/superpowers/specs/2026-07-22-rag-metrics-design.md), [`2026-07-22-batch-evaluation-design`](docs/superpowers/specs/2026-07-22-batch-evaluation-design.md) |
| **v0.3** | 多集合目录 | `Collection`/`Catalog`、`shanzhongshi` 集合、UI 下拉替代"双侧对比" | [`2026-07-21-multi-collection-catalog-design`](docs/superpowers/specs/2026-07-21-multi-collection-catalog-design.md) |
| **v0.5** | Hybrid 检索 | BM25（jieba 中文分词）+ Chroma 向量，RRF 融合 | — |
| **v0.6** | Reranker | Cross-encoder 加权、可配 `RERANK_MIN_SCORE` | — |
| **v0.7** | Intent routing | 意图分类 + 子查询分解 + catalog 扇出 | — |
| **v0.8** | 阈值过滤 + 提示词打磨 | `CHROMA_MAX_DISTANCE`、按集合切 top-k、prompt 收紧 | — |

原则：**每一段新增能力，都伴随一组评估指标；没有数字的优化不进主干。**

## 评估方法论

> "没有 ground_truth，就只能算 LLM-judge；有了 ground_truth，supervised 指标才能上场 —— 而 supervised 指标是判断'这个技巧到底有没有用'的唯一可重复依据。"
>
> —— 改写自 [`2026-07-22-batch-evaluation-design`](docs/superpowers/specs/2026-07-22-batch-evaluation-design.md) §1

闭环长这样：

```
在线 UI  ──▶  JSONL (rag_events_YYYY-MM-DD.jsonl)
                       │
                       ▼
              sample（采样未标注问题）
                       │
                       ▼
              label   （人工或半自动标注 ground_truth）
                       │
                       ▼
              run     （批量送入 RAG，写带 ground_truth 的事件）
                       │
                       ▼
              evaluate（aggregates / by_collection / details）
```

`aggregates` 给出均值/中位/p95；`by_collection` 按集合切分；`details` 是逐条明细。**所有指标的判分都来自同一个事件流，避免"在线一种口径、离线另一种口径"。**

## 环境变量

应用启动时通过 `rag_learn.config.load_config()` 读取；键名严格区分大小写。`DEEPSEEK_API_KEY` 缺失会直接 `ConfigError` 退出。

| Var | Default | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | _(必填)_ | DeepSeek API key，缺失则启动失败。 |
| `LLM_MODEL` | `deepseek-v4-flash` | 任一 DeepSeek API 接受的模型。 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 用于走代理。 |
| `RETRIEVE_K` | `5` | 召回 top-k。 |
| `CHUNK_SIZE` | `800` | 切片字符上限；改后需 `rm -rf data/`。 |
| `CHUNK_OVERLAP` | `50` | 相邻切片重叠字符。 |
| `LOG_LEVEL` | `INFO` | 全局日志级别。 |
| `CHROMA_MAX_DISTANCE` | _(未设)_ | cosine 距离上限，超出直接丢弃。 |
| `HYBRID_ENABLED` | `false` | 打开 BM25 + 向量 RRF 融合。 |
| `HYBRID_RRF_K` | `60` | RRF 公式中的 k 常数。 |
| `RERANK_ENABLED` | `false` | 打开 cross-encoder rerank。 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 任意 `sentence-transformers` CrossEncoder。 |
| `RERANK_K` | _(未设)_ | rerank 前保留的候选数（默认 `RETRIEVE_K * RERANK_FACTOR`）。 |
| `RERANK_DEVICE` | `auto` | `cpu` / `cuda` / `mps` / `auto`。 |
| `RERANK_MIN_SCORE` | _(未设)_ | cross-encoder 分数下限，低于丢弃。 |
| `RERANK_FACTOR` | `4` | `RERANK_K = RETRIEVE_K * RERANK_FACTOR`。 |
| `RERANK_BATCH_SIZE` | `8` | 批量打分。 |
| `INTENT_ENABLED` | `false` | 打开意图分类。 |
| `INTENT_TIMEOUT_S` | `8.0` | 意图分类 LLM 超时。 |
| `DECOMPOSE_ENABLED` | `false` | 打开子查询分解（catalog 扇出）。 |
| `DECOMPOSE_TIMEOUT_S` | `15.0` | 分解 LLM 超时。 |
| `DECOMPOSE_MAX` | `8` | 最多生成的子查询数。 |
| `CATALOG_SUB_K` | `8` | 每个子查询在每个 retriever 上的召回数。 |
| `CATALOG_RECALL_K` | `20` | 合并后进入 prompt 的上限。 |

`.env.example` 提供全部键名的占位；只要保留未启用项为注释即可。

## 测试

```bash
make all   # ruff lint + ty + pytest --cov-fail-under=80
# 或（uv 环境下）：
uv run pytest
```

`pyproject.toml` 强制 `--cov-fail-under=80`，低于 80% 覆盖会让 `make all` 红。`tests/test_*_retriever.py` 与 `tests/test_eval*` 是改对应模块时必动的测试。

## 批量评估 CLI

`rag_learn.eval.cli` 三个子命令驱动同一份 CSV 模板（`question, answer, source_files, chunk_ids, collection`）；`source_files` / `chunk_ids` 以 `;` 分隔，可空。

```bash
# 1) 采样在线流量 → 待标注 CSV
uv run python -m rag_learn.eval.cli sample data \
    --samples-per-collection 5 --output samples.csv

# 2a) 把标注好的 Q&A 跑一遍 RAG，落事件 + 报告
uv run python -m rag_learn.eval.cli run qa.csv \
    --collection shanzhongshi \
    --output-events data/shanzhongshi_events.jsonl \
    --output-report data/shanzhongshi_report.json

# 2b) 不重新查询、直接对事件重算指标（适合调权重 / 调阈值）
uv run python -m rag_learn.eval.cli evaluate data \
    --output data/report.json --dry-run
```

`run` 默认 **crash-resume**：写到 `--output-events` 的 `(collection, question)` 在再次执行时被跳过。需要强制重跑传 `--no-resume`。

DeepSeek 免费档对突发流量敏感。`run` / `evaluate` 都走共享 `RateLimiter`（token-bucket RPM + 并发上限 + 429 指数退避）：

| Flag | Default | 作用 |
|---|---|---|
| `--max-concurrency` | `3` | 同时进行的 judge / 生成调用上限。 |
| `--rate` | `20.0` | 每分钟请求上限；免费档请降到 5。 |
| `--max-retries` | `3` | HTTP 429 后重试次数。 |
| `--no-resume` | off | 仅 `run`：重跑已写过事件的问题。 |

免费档示例：

```bash
uv run python -m rag_learn.eval.cli run docs/eval/shanzhongshi_qa.csv \
    --collection shanzhongshi \
    --output-events data/shanzhongshi_events.jsonl \
    --output-report data/shanzhongshi_report.json \
    --max-concurrency 1 --rate 5
```

`report` 聚合 supervised（`retrieval_recall@k`、`retrieval_precision@k`、`retrieval_mrr`、`retrieval_ndcg@k`、`answer_f1`）+ LLM-judge（`context_relevance`、`faithfulness`、`answer_relevance`、`overall_usefulness`、`answer_llm_correctness`）。

完整设计参见 [`2026-07-22-batch-evaluation-design`](docs/superpowers/specs/2026-07-22-batch-evaluation-design.md)。

## 历史 adapter：Milvus

`src/rag_learn/retriever/milvus_impl.py` 与 `tests/test_milvus_retriever.py` 仍保留在仓库中，作为历史 adapter，**但当前主路径不再实例化**：`factory.build_retriever` 只返回 Chroma / Hybrid，`app.launch` 不再 import `MilvusRetriever`。如果你想比较 Milvus Lite，可手动 import 该 adapter 并自行承担 `pymilvus` 2.6+ 在 macOS ARM 上的 SIGSEGV 风险（参见 `CLAUDE.md` Known gotchas）。本项目不把 Milvus 作为演示特性。

## Roadmap 与已知限制

- **Incremental UI streaming 还没做**。当前 `gr.Chatbot` 把整个回答聚成一次 frame update（一次 delta / 一次 click），不是按 token 增量 flush；`TODO` 在 `src/rag_learn/app.py` 标了改造点。
- **本项目不是生产级 RAG**：不做鉴权、不做监控、不做水平扩展，只是一个学习工程。
- **Milvus adapter 已退役但代码还在**：见上一节，不计划恢复为主路径。
- **chunking 改了要清索引**：修改 `CHUNK_SIZE` / `CHUNK_OVERLAP` 后请先 `rm -rf data/`，再启动 app。

## 致谢

构建于以下开源项目之上：

- [Chroma](https://www.trychroma.com/) — 向量存储
- [DeepSeek](https://api-docs.deepseek.com/) — 流式 LLM（OpenAI 兼容）
- [Gradio 5](https://gradio.app/) — UI
- [sentence-transformers](https://www.sbert.net/) — embedder + CrossEncoder
- [BAAI / BGE](https://huggingface.co/BAAI) — `bge-reranker-base` 等模型
- [jieba](https://github.com/fxsjy/jieba) — 中文分词
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25
- [pyrate-limiter](https://pypi.org/project/pyrate-limiter/) / [tenacity](https://tenacity.readthedocs.io/) — 限流与重试

中文领域语料：`docs/shanzhongshi/`（山中事咖啡——豆种 / 烘焙度 / 冲煮 / 公司信息）。

## License

MIT —— 见 [LICENSE](./LICENSE)。