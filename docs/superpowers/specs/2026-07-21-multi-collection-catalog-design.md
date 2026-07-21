# Multi-Collection Catalog (山中事咖啡新集合 + Collection 注册抽象) — Design

- **Status:** Approved (post-brainstorming)
- **Date:** 2026-07-21
- **Owner:** dkisser
- **Supersedes:** §4.2 data layout & §5.5 startup of `2026-07-18-rag-multiretriever-design.md`（仅 data/chroma 目录结构 + 启动 ingest 流程部分）

## 1. Purpose

把 `docs/shanzhongshi/`（3 篇山中事咖啡相关 markdown）接入 RAG 系统，
作为**独立的 Chroma 集合**，与现有 `docs/rag_doc/` 集合共存。同时把
"知识库" 抽象成可注册的 `Collection` + `Catalog`，让加新集合从「改 Python
代码」变成「在 collections.py 加一行」，并让 UI 用下拉框选要查询的集合。

最终形态：用户启动应用 → 顶部下拉显示 `RAG 论文集` / `山中事咖啡` →
选一个 + 输入问题 → 流式回答。

## 2. In Scope / Out of Scope

**In scope**

- 新建 `src/rag_learn/collections.py`，定义 `Collection` dataclass + `Catalog` 类
- `collections.py` 内置两个 collection：`rag_doc`（现有）、`shanzhongshi`（新增）
- `app.py` UI 改造：顶部 `gr.Dropdown` 选 collection，删除并排对比
- `pipeline.answer_stream` 签名 + 行为**不变**（保持 `dict[str, BaseRetriever]`，新模式调用方传 N=1 的 dict）
- 老 `data/chroma/` 一次性自动迁移到 `data/chroma/rag_doc/`
- 测试覆盖率 ≥ 80%；TDD；ruff + ty

**Out of scope (YAGNI)**

- 多 collection 并行召回 + 合并（multi-collection ensemble）
- per-collection 自定义 system prompt
- 自动扫描 `docs/*` 子目录发现 collection
- 跨 collection 检索结果去重 / 重排
- UI 切回 Chroma vs Milvus 并排对比（历史 demo 形态）
- 持久化对话历史
- 认证 / 限流 / 云部署

## 3. Confirmed Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | 交互模型 | 顶部下拉单选 + 单流式回答（替代并排对比） |
| 2 | 发现机制 | 手动注册表 `collections.py` |
| 3 | 抽象层级 | `Collection` 领域对象 + `Catalog` 注册表 |
| 4 | retriever 加载时机 | **Eager**：启动期一次性 `ensure_all_indexed()`，与现有 launch() 行为一致 |
| 5 | 持久化目录布局 | 每个 collection 独立 `data/chroma/<name>/`，物理隔离 |
| 6 | shanzhongshi 粒度 | 单 collection（3 篇 md 一起入库），不细分 |
| 7 | 老 `data/chroma/` 迁移 | 启动期自动迁移到 `data/chroma/rag_doc/`，写 `.migrated` 标记防重复 |
| 8 | pipeline.answer_stream | **不改**，N=1 调用与 N=2 调用走同一路径 |
| 9 | 并排 Chroma vs Milvus UI | **删除**（不在 `app.launch()` 中构造 Milvus 实例，Milvus 适配器代码保留） |

## 4. Architecture

### 4.1 高层组件图

```
   ┌──────────────────────────────────────────────────┐
   │              Gradio UI (app.py)                  │
   │  ┌────────────────────────────────────────────┐  │
   │  │  [Collection Dropdown ▾]  [question    ]    │ │
   │  │              [Send] [Clear]                │ │
   │  └────────────────────────────────────────────┘  │
   │  ┌────────────────────────────────────────────┐  │
   │  │  ▼ 回答 (single stream)                    │ │
   │  │  ▼ chunks                                  │ │
   │  │  ▼ perf: retrieve/first_token/total        │ │
   │  └────────────────────────────────────────────┘  │
   └─────────────────────────┬────────────────────────┘
                             │ catalog.get(slug).retriever
                             ▼
   ┌──────────────────────────────────────────────────┐
   │  collections.py (new)                            │
   │   Collection (frozen dataclass)                  │
   │   Catalog (frozen dataclass)                     │
   │   BUILTIN_COLLECTIONS = (rag_doc, shanzhongshi)  │
   │   build_catalog() → Catalog                      │
   └─────────────────────────┬────────────────────────┘
                             │ uses
                             ▼
   ┌──────────────────────────────────────────────────┐
   │  retriever/* (不变)                              │
   │   BaseRetriever Protocol + Hit                   │
   │   ChromaRetriever(persist_dir, collection_name) │
   └─────────────────────────┬────────────────────────┘
                             │
                             ▼
   ┌──────────────────────────────────────────────────┐
   │  pipeline.answer_stream (不变)                   │
   │   answer_stream({slug: retriever}, llm, q)       │
   │   → build_prompt → DeepSeekLLM.stream            │
   └──────────────────────────────────────────────────┘
```

### 4.2 Directory layout（变更部分）

```
src/rag_learn/
├── collections.py                 # 新
├── config.py                      # 不变
├── loader.py                      # 不变
├── pipeline.py                    # 不变（顶部加一行注释说明两种调用模式）
├── app.py                         # 改：build_app 接 Catalog，删并排 UI
├── llm.py                         # 不变
└── retriever/
    ├── __init__.py
    ├── base.py                    # 不变
    ├── chroma_impl.py             # 不变（已支持 collection_name 参数）
    └── milvus_impl.py             # 不变（保留但 launch() 不再实例化）

data/chroma/
├── rag_doc/                       # 新位置（自动迁移自原 data/chroma/）
│   ├── chroma.sqlite3
│   └── <uuid>/...
└── shanzhongshi/                  # 启动期首次 ingest 时创建

tests/
├── test_collections.py            # 新
├── test_app_launch.py             # 改：build_app 接 Catalog 而非 dict
├── test_e2e.py                    # 略改：构造多 collection mock catalog
├── test_chroma_retriever.py       # 不变（collection_name 已支持）
└── ... (其余测试不变)
```

## 5. Data Flow

### 5.1 启动期

```
app.launch()
  ├─ load_config()                       # .env, paths
  ├─ DeepSeekLLM(api_key, model, ...)
  ├─ catalog = build_catalog()           # 读 BUILTIN_COLLECTIONS
  ├─ _migrate_legacy_chroma(config)      # 见 §5.4
  ├─ warnings = catalog.ensure_all_indexed()
  │     ├─ collection rag_doc.retriever
  │     │     ├─ ChromaRetriever(data/chroma/rag_doc, "rag_doc")
  │     │     └─ .ensure_indexed("docs/rag_doc")
  │     └─ collection shanzhongshi.retriever
  │           ├─ ChromaRetriever(data/chroma/shanzhongshi, "shanzhongshi")
  │           └─ .ensure_indexed("docs/shanzhongshi")
  └─ build_app(catalog=catalog, llm, config, warnings)
        └─ Gradio Blocks 渲染 Dropdown(choices=catalog.display_choices())
```

### 5.2 运行时（单问题）

```
用户：选择 "山中事咖啡" + 输入 "耶加雪菲怎么冲？" + 提交
       │
       ▼
build_app.on_submit(collection_slug="shanzhongshi", question="...")
       │
       ├─ collection = catalog.get("shanzhongshi")    # O(1) lookup
       ├─ retriever = collection.retriever           # 已缓存
       ├─ outputs = answer_stream(
       │     {"shanzhongshi": retriever},
       │     llm,
       │     question,
       │     k=config.retrieve_k
       │ )
       ├─ stream_iter, hits, perf_fn = outputs["shanzhongshi"]
       ├─ chunks_md = _format_chunks(hits)
       ├─ answer_text = "".join(drain(stream_iter))
       └─ perf = perf_fn()
       │
       ▼
Gradio Chatbot: [user, "耶加雪菲怎么冲？"] + [assistant, answer_text]
chunks_md:       "** [1] `咖啡冲煮技巧.md#3` (dist=0.42) ..."
perf_md:         "检索 28ms · 首个 token 410ms · 总 1.1s · 完成于 14:23:45.123"
```

### 5.3 answer_stream 两种调用模式（pipeline.py 顶部注释）

```python
# answer_stream 支持两种调用模式：
#   1. 单 collection 模式（新 collection picker UI 用法）：
#        answer_stream({slug: retriever}, llm, q)
#   2. 多 retriever 并排对比（历史 Chroma vs Milvus demo 用法）：
#        answer_stream({"chroma": c, "milvus": m}, llm, q)
# 两种模式底层完全相同：并行检索 → 各自 build_prompt → 各自流式生成。
```

底层 `_retrieve` 用 `ThreadPoolExecutor(max_workers=max(2, N))`：
N=1 时退化为单线程；N=2+ 时并行。`_TimedIter` / `_make_perf` 无变化。

### 5.4 老 `data/chroma/` 自动迁移

启动期 `app.launch()` 在 `build_catalog()` 之后调用 `_migrate_legacy_chroma(config)`：

```python
def _migrate_legacy_chroma(config: Config) -> None:
    """一次性：把 data/chroma/ 根下的遗留文件搬到 data/chroma/rag_doc/。
    
    触发条件：data/chroma/rag_doc/ 不存在 AND data/chroma/ 根下含
    chroma.sqlite3 或 形如 <uuid>/ 的子目录。
    
    幂等：迁移完成后写 data/chroma/.migrated 哨兵文件，下次启动跳过。
    """
    target = config.chroma_dir / "rag_doc"
    marker = config.chroma_dir / ".migrated"
    if target.exists() or marker.exists():
        return
    legacy_files = list(config.chroma_dir.glob("chroma.sqlite3")) + \
                   [p for p in config.chroma_dir.iterdir()
                    if p.is_dir() and _UUID_RE.match(p.name)]
    if not legacy_files:
        return
    target.mkdir(parents=True, exist_ok=True)
    for src in legacy_files:
        shutil.move(str(src), str(target / src.name))
    marker.write_text("migrated at " + _now_iso() + "\n")
```

`_migrate_legacy_chroma` 与 `catalog.ensure_all_indexed` 的顺序：先迁移，
再 ingest。迁移完成后 `ChromaRetriever("data/chroma/rag_doc", "rag_doc")` 会
读到已索引数据，`ensure_indexed` 的 `if self._collection.count() > 0` 短路。

### 5.5 Prompt 模板（不变）

```
system: 你是一个 RAG 助手。尽量基于下方提供的「上下文」回答用户问题。

user:   上下文：
        [1] (来源: {source_file}) {chunk_text_1}
        ...
        [k] (来源: {source_file}) {chunk_text_k}

        问题：{user_query}
        回答：
```

不变理由：coffee 领域问答同样适用"基于上下文回答"。per-collection prompt
定制属于 YAGNI（见 §2）。

## 6. Component Contracts

### 6.1 `Collection` (`collections.py`)

```python
@dataclass(frozen=True)
class Collection:
    """一个独立知识库：slug + 显示元数据 + 文档目录 + retriever 工厂。"""
    name: str                                              # Chroma 集合名 + 内部 slug
    display_name: str                                      # 中文（下拉显示）
    docs_dir: Path                                         # docs/rag_doc 等
    description: str = ""                                  # UI 副标题
    retriever_factory: Callable[[Path, str], BaseRetriever] = (
        lambda persist_dir, name: ChromaRetriever(persist_dir, name)
    )

    def __post_init__(self) -> None:
        # Chroma collection name 约束：3-63 字符，alnum + _-.
        if not (3 <= len(self.name) <= 63) or \
           not all(c.isalnum() or c in "_-." for c in self.name):
            raise ValueError(f"Invalid collection name: {self.name!r}")
        if not self.docs_dir.is_dir():
            raise ValueError(f"docs_dir does not exist: {self.docs_dir}")
        object.__setattr__(self, "_retriever", None)       # lazy cache

    @property
    def retriever(self) -> BaseRetriever:
        """懒加载：首次访问时建 retriever + ingest；同实例复用。"""
        if self._retriever is None:
            persist_dir = (
                self.docs_dir.parent.parent / "data" / "chroma" / self.name
            )
            self._retriever = self.retriever_factory(persist_dir, self.name)
            self._retriever.ensure_indexed(str(self.docs_dir))
        return self._retriever
```

**retriever 持久化目录推导**：`self.docs_dir.parent.parent / "data" / "chroma" / self.name`。
对 `docs/rag_doc` → `data/chroma/rag_doc`；对 `docs/shanzhongshi` →
`data/chroma/shanzhongshi`。

### 6.2 `Catalog` (`collections.py`)

```python
class CollectionNotFoundError(KeyError):
    """请求的 collection 不在注册表里。"""

@dataclass(frozen=True)
class Catalog:
    """不可变集合注册表：slug → Collection 双向索引。"""
    collections: tuple[Collection, ...]

    def __post_init__(self) -> None:
        names = [c.name for c in self.collections]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate collection names: {names}")
        object.__setattr__(self, "_by_name", {c.name: c for c in self.collections})

    def names(self) -> list[str]: ...
    def display_choices(self) -> list[tuple[str, str]]: ...   # [(display, slug), ...]
    def get(self, name: str) -> Collection: ...              # raises CollectionNotFoundError
    def ensure_all_indexed(self) -> list[tuple[str, str]]: ...  # fail-open，返回 warnings
```

### 6.3 `BUILTIN_COLLECTIONS` (`collections.py`)

```python
def _build_builtin() -> tuple[Collection, ...]:
    root = _repo_root() / "docs"
    return (
        Collection(
            name="rag_doc",
            display_name="RAG 论文集",
            docs_dir=root / "rag_doc",
            description="25 篇 RAG 相关论文 / 综述 / 实践文章",
            retriever_factory=_default_factory,
        ),
        Collection(
            name="shanzhongshi",
            display_name="山中事咖啡",
            docs_dir=root / "shanzhongshi",
            description="山中事咖啡（SHAN.IN COFFEE）的豆子参数、冲煮教程与公司信息",
            retriever_factory=_default_factory,
        ),
    )

BUILTIN_COLLECTIONS: tuple[Collection, ...] = _build_builtin()

def build_catalog() -> Catalog:
    return Catalog(collections=BUILTIN_COLLECTIONS)
```

### 6.4 `app.build_app` 新签名

```python
def build_app(
    catalog: Catalog,                          # 旧：retrievers: dict[str, BaseRetriever]
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
) -> gr.Blocks: ...
```

返回的 UI：
- 顶部一行：`gr.Dropdown(choices=catalog.display_choices(), value=choices[0][1])`
  + `gr.Textbox(label="问题", lines=2, scale=3)`
- 第二行：`gr.Button("发送")` + `gr.Button("清空")`
- 第三行（单列）：`gr.Chatbot(type="messages")` + `gr.Accordion("检索到的 chunks")`
  + `gr.Markdown` (perf)

`on_submit(collection_slug, q)` 调用 `catalog.get(slug).retriever`，构造
`answer_stream({slug: retriever}, llm, q)`。

### 6.5 `app.launch`

```python
def launch() -> None:
    config = load_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    llm = DeepSeekLLM(...)
    _migrate_legacy_chroma(config)               # 见 §5.4
    catalog = build_catalog()
    warnings = catalog.ensure_all_indexed()
    if not catalog.names():
        raise SystemExit("Catalog 为空，无法启动")
    app = build_app(catalog=catalog, llm=llm, config=config, warnings=warnings)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app.queue().launch(server_name="127.0.0.1", server_port=7860)
```

### 6.6 `pipeline.answer_stream` — 不变

```python
def answer_stream(
    retrievers: dict[str, BaseRetriever],
    llm: DeepSeekLLM,
    question: str,
    k: int = 5,
) -> dict[str, tuple[Iterator[str], list[Hit], Callable[[], StreamPerf]]]:
    ...
```

唯一改动：函数顶部加 §5.3 那段 docstring 注释。

## 7. Error Handling

| Failure | Trigger | Behavior |
|---------|---------|----------|
| `ConfigError` | `DEEPSEEK_API_KEY` unset | raise; abort boot |
| `Collection.__post_init__` ValueError | slug 不合规 / docs_dir 不存在 | 构造时 fail-fast（不能注册一个跑不起来的 collection） |
| `Catalog.__post_init__` ValueError | 重名 slug | 构造时 fail-fast |
| `CollectionNotFoundError` | UI 传未知 slug（理论上不会发生，dropdown 限定） | UI 显示 `⚠ 未知集合：{slug}` |
| 单 collection `ensure_indexed` 抛异常 | Chroma 故障 / 文件权限 / embed 模型下载失败 | log warning；该 collection 从 UI 中**隐藏**？**还是**仍展示但 search() 返回空？—— 见 §7 决策 1 |
| `answer_stream` 抛异常 | embed / search / LLM 网络故障 | UI Chatbot 显示 `⚠ 流水线失败：{exc}` |
| 流式生成中途异常 | LLM 连接中断 | Chatbot 显示 `⚠ 检索失败：{exc}`，perf 留空 |
| 老 data/chroma 迁移失败 | shutil.move IO 错误 | log warning；继续启动（用户可手动 `make clean`） |

**决策点（待 §6 写完后二次确认）**：单 collection ingest 失败时，
该 collection 应该从 Dropdown choices 中**剔除**（避免用户选了搜不到），
还是**保留**但 search() 返回 `[]`（保留选项可见，UI 显示"无召回"）？

当前倾向：**剔除**。在 `build_app` 构造 UI 前，从 catalog 中过滤掉 ingest
失败的 collection：

```python
warnings, ready = _split_warnings(catalog, raw_warnings)
working_catalog = Catalog(collections=tuple(c for c in catalog.collections
                                            if c.name not in dict(warnings)))
# build_app 接 working_catalog；UI Dropdown 只展示能用的
```

`warnings` 走顶部 banner（与现有行为一致）。

**保留原 §7 不变的项**：`_TimedIter` 内部 try/except、`on_submit` 的
`except Exception` fail-open、`EmptyHit` system prompt 分支。

## 8. Observability

### 8.1 Server logs（新增）

```
[<ts>] Catalog: 2 collections [rag_doc, shanzhongshi]
[<ts>] Migrated legacy Chroma data: 3 files → data/chroma/rag_doc/
[<ts>] Catalog ingest: rag_doc ready (124 chunks)
[<ts>] Catalog ingest: shanzhongshi ready (8 chunks)
[<ts>] Catalog ingest warnings: []
```

### 8.2 Stream perf（不变）

```
[<ts>] shanzhongshi  retrieve=28ms  first_token=410ms  total=1100ms
```

格式不变，仅 side 名从 `chroma` / `milvus` 换成实际 slug。

## 9. Testing

TDD mandatory，覆盖率门 `--cov-fail-under=80` 不变。

| Test | Type | 关键覆盖 |
|------|------|----------|
| `test_collections.py`（新） | unit | `Collection.__post_init__` 校验 / 懒加载缓存 / `Catalog` 去重 / `get` 抛 `CollectionNotFoundError` / `display_choices` 顺序 / `ensure_all_indexed` fail-open / `build_catalog` 含 rag_doc + shanzhongshi |
| `test_app_launch.py`（改） | smoke | `build_app(catalog, ...)` 接 Catalog 构造 UI；Dropdown choices 与 catalog 一致；on_submit 输入 dropdown 改变答案侧 |
| `test_e2e.py`（略改） | e2e | mock catalog 含 2 个 collection；mock retriever 返回固定 hits；断言 UI 选择不同 collection 时答案不同 |
| `test_chroma_retriever.py` | integration | 不变（collection_name 构造参数已支持） |
| `test_milvus_retriever.py` | integration | 不变 |
| `test_pipeline.py` / `test_pipeline_parallel.py` | unit | 不变 |

### 新增 fixture：`tests/fixtures/sample_docs_alt/`

第二个 fixture（3 篇不同语义的 markdown），让 `test_e2e` 能模拟两个
互不污染的 collection。fixture 内容不需精心——保证 embed 后两个 collection
检索 top-1 不同即可。

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| 老 `data/chroma/` 迁移到 `data/chroma/rag_doc/` 时碰到 IO 错误 | log warning；启动继续；用户提供 `make clean` 文档路径 |
| Chroma 集合名约束与子目录名不一致（如 docs 下出现 `Rag Doc/` 带空格） | `Collection.__post_init__` 校验 slug 字符；不强制 docs_dir 目录名 = slug |
| shanzhongshi 只有 3 个 md，可能 ingest 出极少 chunks（< 5），top-k=5 时返回少 | `RETRIEVE_K` 仍是 5；不多不少；如果用户嫌少是模型问题不是架构问题 |
| Eager ingest 启动期慢（2 个 collection × Chroma embed 下载） | Chroma 共享 `data/chroma/` 根目录的 embed 模型缓存；首次启动 ~5s，后续启动 < 1s |
| Dropdown 默认值写死为第一个 collection，未来加新 collection 时默认变了 | 显式 `default_slug` 字段？或依赖"第一个就是默认"约定？倾向约定（YAGNI） |
| 用户切换 collection 时 on_submit 拿到旧 retriever（线程安全） | Gradio handler 是同步的；`collection.retriever` 是不可变 + 缓存读，无并发问题 |

## 11. Glossary

- **Collection** — 一个独立知识库（slug + 显示元数据 + docs_dir + retriever 工厂）
- **Catalog** — Collection 的不可变集合 + 启动期 eager ingest 协调器
- **slug** — 内部唯一标识（与 Chroma collection name 一致），与 display_name 区分
- **Eager ingest** — 启动期一次性对所有 collection 调 `.retriever` 触发 ingest；与 fail-open 结合
- **Legacy migration** — 老 `data/chroma/` 根目录下的文件搬到 `data/chroma/rag_doc/`

## 12. Open Questions

设计阶段已收敛。本节列出**实施期**可能出现的小问题，不阻塞 spec 批准：

1. §7 决策点：单 collection ingest 失败时从 Dropdown 剔除 vs 保留——倾向剔除，待 §6 实施时确认 Gradio Dropdown 能否在构造后修改 choices。
2. `Collection.retriever` 的 lazy 属性 + `object.__setattr__` 在 dataclass(frozen=True) 上的兼容性——`pytest` + `ty` 跑过即知。
3. `_migrate_legacy_chroma` 的 UUID 匹配正则——具体字符范围实施期定。