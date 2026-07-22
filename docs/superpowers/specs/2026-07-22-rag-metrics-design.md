# RAG 埋点与效果评估设计

**日期**: 2026-07-22  
**作者**: Claude Code  
**状态**: 已批准，待实现  
**关联**: `rag-learn` 项目

## 1. 背景与目标

`rag-learn` 当前已实现 Chroma/Milvus 双检索器对比的 RAG 链路，并在 `pipeline.py` 中通过 `StreamPerf` 记录了基础延迟指标。然而，项目仍缺少结构化的 RAG 效果评估能力：

- 无法量化检索质量（召回是否准确）。
- 无法评估生成质量（答案是否忠实、相关）。
- 没有持久化的请求日志，难以离线分析或批量跑分。

本设计目标是在现有链路上增加一套**可插拔、低开销、本地优先**的埋点与评估机制，同时支持：

1. **在线运行监控**：记录每次用户请求的检索、生成、性能数据。
2. **离线效果评估**：对有 ground truth 的问题计算传统指标；对无 ground truth 的问题，通过后台批量 LLM-as-judge 打分。

## 2. 设计原则

- **不阻塞主链路**：评分、指标计算放在后台手动触发，不影响用户响应。
- **Fail-open**：埋点/emitter 失败不影响 RAG 返回回答。
- **可插拔存储**：先实现本地 JSONL emitter，后续可扩展为 SQLite、Prometheus、OpenTelemetry 等。
- **最小依赖**：优先使用标准库和项目已有依赖（`openai<2`）。
- **高可测性**：通过 `ListEmitter` 等测试替身，保证 TDD 和 80%+ 覆盖率。

## 3. 模块组织

所有评估相关代码收敛到 `src/rag_learn/eval/` 包下，避免源码根目录散落。

```
src/rag_learn/
├── eval/
│   ├── __init__.py          # 导出公共 API：RAGEvent, MetricsEmitter, JSONLEmitter, ListEmitter
│   ├── tracing.py           # 事件模型与 Emitter 实现
│   ├── metrics.py           # 指标计算函数（检索/生成/系统）
│   └── batch.py             # 批量评估 CLI 入口
├── pipeline.py              # 修改：answer_stream 接收可选 emitter
└── app.py                   # 修改：实例化 JSONLEmitter 并传入 pipeline
```

对应测试：

```
tests/
├── eval/
│   ├── test_tracing.py
│   ├── test_metrics.py
│   └── test_batch.py
├── test_pipeline.py         # 扩展 emitter 相关用例
└── test_app_launch.py       # 扩展 build_app 传 emitter 用例
```

## 4. 数据模型

### 4.1 `RAGEvent`

每个 RAG 请求产生一个事件，包含检索、生成、性能全量信息。

```python
@dataclass(frozen=True)
class GroundTruth:
    # 期望答案文本（可选）
    answer: str | None = None
    # 期望引用的来源文件列表，用于计算检索指标
    source_files: tuple[str, ...] = ()
    # 期望引用的 chunk 标识列表（可选，更细粒度）
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RAGEvent:
    trace_id: str                      # UUID，用于关联评估结果
    timestamp: str                     # ISO-8601
    collection: str                    # 知识库 slug
    question: str
    hits: list[Hit]                    # 检索命中的 chunks
    prompt: str                        # 最终送给 LLM 的完整 prompt
    answer: str                        # 生成的完整回答
    perf: StreamPerf                   # 性能指标
    ground_truth: GroundTruth | None   # 有标注时填写
    metadata: dict[str, Any]           # 扩展字段：模型名、retrieve_k 等
```

**字段说明**：

- `prompt` 显式存储，便于后续分析 prompt 工程效果。
- `ground_truth` 为 `None` 表示无标注，后续走 LLM-as-judge。
- `metadata` 保留运行期配置（`llm_model`、`retrieve_k` 等），便于 A/B 对比。

### 4.2 `MetricsEmitter` Protocol

```python
class MetricsEmitter(Protocol):
    def emit(self, event: RAGEvent) -> None: ...
```

首批实现：

- `JSONLEmitter(dir_path: Path)`：按天轮转写入 `<dir_path>/rag_events_YYYY-MM-DD.jsonl`。
- `ListEmitter()`：内存列表，专供测试使用。
- `NullEmitter()`：空实现，用于默认不回退行为。

## 5. 数据流

### 5.1 在线阶段

```
用户提交问题
    │
    ▼
app.on_submit()
    │
    ▼
answer_stream(retrievers, llm, question, k, emitter=JSONLEmitter)
    │
    ├── 并行 retrieve → hits_by_side
    ├── 拼 prompt
    ├── 返回 token iterator + hits + perf_fn
    │
    ▼
UI drain iterator → 得到完整 answer
    │
    ▼
调用 perf_fn() → StreamPerf
    │
    ▼
emitter.emit(RAGEvent(...))
    │
    ▼
追加写入 data/rag_events_YYYY-MM-DD.jsonl
```

### 5.2 离线批量评估阶段

```
python -m rag_learn.eval.batch data/ --output report.json
    │
    ▼
扫描 data/rag_events_*.jsonl
    │
    ▼
按 trace_id 去重
    │
    ├── 有 ground_truth
    │   ├── retrieval_recall@k
    │   ├── retrieval_precision@k
    │   ├── retrieval_mrr
    │   ├── retrieval_ndcg@k
    │   └── answer_f1 / exact_match（若提供标准答案文本）
    │
    └── 无 ground_truth
        ├── context_relevance
        ├── faithfulness
        ├── answer_relevance
        └── overall_usefulness
    │
    ▼
输出 report.json（聚合统计 + 逐条明细）
```

## 6. 评估指标清单

### 6.1 检索指标

| 指标 | 需要 ground truth | 说明 |
|------|------------------|------|
| `retrieval_recall@k` | ✅ | 相关文档是否出现在 Top-K 中。 |
| `retrieval_precision@k` | ✅ | Top-K 中相关文档的比例。 |
| `retrieval_mrr` | ✅ | 第一个相关命中的排名倒数。 |
| `retrieval_ndcg@k` | ✅ | 考虑排名位置的相关性加权。 |
| `avg_hit_score` | ❌ | 命中 chunks 的平均相似度分数（L2 距离，越小越好）。 |
| `num_hits_retrieved` | ❌ | 实际召回的 chunk 数量。 |
| `retrieved_sources` | ❌ | 召回来源文件列表，用于观察覆盖度。 |

### 6.2 生成指标

**有标准答案时：**

| 指标 | 说明 |
|------|------|
| `answer_exact_match` | 答案与标准答案是否完全匹配（偏严，仅供参考）。 |
| `answer_f1` | 答案与标准答案的字符/词 F1（使用简单分词，不引入额外 tokenizer 依赖）。 |
| `answer_llm_correctness` | LLM-as-judge 判断答案是否正确。 |

**无标准答案时（LLM-as-judge）：**

| 指标 | 说明 |
|------|------|
| `context_relevance` | 召回的上下文是否与问题相关。 |
| `faithfulness` | 答案是否忠实于上下文，没有编造。 |
| `answer_relevance` | 答案是否回答了问题。 |
| `overall_usefulness` | 综合有用性评分（如 1-5 分）。 |

### 6.3 系统/运营指标

| 指标 | 来源 |
|------|------|
| `retrieve_ms` | `StreamPerf.retrieve_ms` |
| `first_token_ms` | `StreamPerf.first_token_ms` |
| `total_ms` | `StreamPerf.total_ms` |
| `num_input_tokens` | prompt 字符数估算（按 4 字符 ≈ 1 token），避免新增依赖。 |
| `num_output_tokens` | 实际生成字符数估算（按 4 字符 ≈ 1 token）。 |
| `error_flag` | 请求是否发生异常。 |
| `empty_hits_flag` | 是否没有召回任何 chunk。 |

## 7. 接口变更

### 7.1 `pipeline.py`

`answer_stream` 增加可选 `emitter` 参数：

```python
def answer_stream(
    retrievers: dict[str, BaseRetriever],
    llm: DeepSeekLLM,
    question: str,
    k: int = 5,
    emitter: MetricsEmitter | None = None,
) -> dict[str, tuple[Iterator[str], list[Hit], Callable[[], StreamPerf]]]:
```

emit 时机：当 caller drain 完 iterator 并调用 `perf_fn()` 后，构造完整 `RAGEvent` 并调用 `emitter.emit()`。

### 7.2 `app.py`

- `build_app` 内部实例化 `JSONLEmitter(dir_path=config.data_dir)`。
- 调用 `answer_stream` 时传入 `emitter`。
- 保留原有 `logger.info` 性能日志，作为双重保障。

## 8. 批量评估 CLI

```bash
python -m rag_learn.eval.batch data/ --output report.json
```

参数：

- `events_dir`: JSONL 文件所在目录（默认 `data/`）。
- `--output`: 报告输出路径（默认 `data/eval_report_YYYY-MM-DD.json`）。
- `--judge-model`: 指定 judge 模型（默认读取 `LLM_MODEL` 环境变量）。
- `--dry-run`: 只统计事件数，不调用 LLM。

输出 `report.json` 结构：

```json
{
  "generated_at": "2026-07-22T12:00:00Z",
  "total_events": 150,
  "with_ground_truth": 50,
  "without_ground_truth": 100,
  "aggregates": {
    "retrieval_recall@5": { "mean": 0.78, "median": 0.80, "p95": 1.0 },
    "faithfulness": { "mean": 0.82, "median": 0.85, "p95": 1.0 }
  },
  "by_collection": {
    "rag_doc": { "retrieval_recall@5": 0.80, "faithfulness": 0.85 },
    "shanzhongshi": { "retrieval_recall@5": 0.70, "faithfulness": 0.75 }
  },
  "details": [
    { "trace_id": "...", "metrics": { ... } }
  ]
}
```

## 9. 错误处理

| 场景 | 策略 |
|------|------|
| `emitter.emit()` 失败（磁盘满、权限不足） | 捕获异常，`logger.warning` 记录，不影响主链路返回回答。 |
| JSONL 并发写入 | 当前单进程 Gradio 天然串行；未来多 worker 再引入文件锁。 |
| `batch.py` 读取损坏行 | 跳过并计数，报告 `skipped_corrupted_lines`。 |
| LLM-as-judge 单条失败 | 不中断批量跑分，该指标记为 `null`，记录失败原因。 |
| ground truth 格式错误 | 校验 `source_files` 非空且为字符串；无效则跳过该条。 |
| 敏感信息泄露 | JSONL 不记录 API key；prompt 按用户要求存储用于分析。 |

## 10. 实现计划概要

1. 创建 `src/rag_learn/eval/` 包与 `tests/eval/` 测试目录。
2. 实现 `tracing.py`：`RAGEvent`、`GroundTruth`、`MetricsEmitter`、`JSONLEmitter`、`ListEmitter`。
3. 实现 `metrics.py`：检索指标、LLM-as-judge prompt、评分解析。
4. 修改 `pipeline.py`：`answer_stream` 接入 emitter。
5. 修改 `app.py`：实例化 `JSONLEmitter` 并传入。
6. 实现 `eval/batch.py`：批量评估 CLI。
7. 补充测试：`test_tracing.py`、`test_metrics.py`、`test_batch.py`，扩展 `test_pipeline.py` 和 `test_app_launch.py`。
8. 运行 `make all` 确保 lint、typecheck、测试、覆盖率全部通过。

## 11. 未来扩展

- **SQLite emitter**：将事件写入本地 SQLite，便于 SQL 查询。
- **Prometheus metrics**：暴露 `rag_requests_total`、`rag_latency_seconds` 等。
- **UI 反馈按钮**：在 Gradio 界面增加 👍/👎，把人工反馈也写入 JSONL。
- **独立 judge 模型配置**：通过环境变量配置更便宜的 judge 模型，降低成本。
- **A/B 实验**：在 `metadata` 中记录配置版本，按版本聚合指标。

## 12. 决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 存储格式 | JSONL | 用户指定，便于解析和后续迁移。 |
| 评分时机 | 后台手动批量 | 不同步阻塞用户，成本可控。 |
| 代码组织 | `rag_learn/eval/` 包 | 用户要求，便于理解。 |
| Judge 模型 | 复用当前 DeepSeek | 实现最简单，后续可配置化。 |
| Prompt 存储 | 存入 JSONL | 用户要求，便于后续分析。 |
| Emitter 失败 | Fail-open | 不因为埋点失败影响用户体验。 |
