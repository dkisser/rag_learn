# Contributing

> 本项目的哲学在 [README.md — Why this project](./README.md#why-this-project)：
> **每一段新增能力都伴随一组评估指标；没有数字的优化不进主干。**
>
> 提 PR 前请把这段原则在脑子里跑一遍：你的改动能不能用一组数字说明"比之前好"？如果不能，先想想是不是缺一个评估。

## 开发环境

```bash
# uv (推荐)
uv sync --extra dev

# pip
pip install -e ".[dev]"
```

## Pre-commit gate

本仓库的提交门禁是一行命令：

```bash
make all   # ruff lint + ty typecheck + pytest --cov-fail-under=80
```

- `ruff check src tests`（`line-length=100`，规则集 `E F I B UP`）
- `ty check src`（**注意是 `ty`，不是 `mypy`**，见 `Makefile`）
- `pytest`，`pyproject.toml` 强制覆盖率 ≥ 80%

任何一项红都不进主干；CI 还没接，这一步目前依赖本地执行。

## TDD 工作流

1. **先写一个会红的测试**——明确"成功后应该观察到什么"。
2. **跑测试**——确认它在没改实现时是红的（否则这个测试没意义）。
3. **最小实现让它变绿**——只写让测试过的那部分。
4. **重构**——保留测试为绿的同时清掉代码异味。
5. **验证覆盖率**——新增代码必须被覆盖；`make all` 会拦 80%。

修改已有模块时先看对应的测试文件：`tests/test_<module>.py`。

## Repository tooling (graphify)

本仓库用 `graphify`（由 `graphifyy` 包提供）做贡献者侧代码导航。**应用运行时不依赖 `graphify`**；它只是用来探索 `graphify-out/` 里已经构建好的代码知识图谱。

```bash
# 安装（推荐 uv tool）
uv tool install graphifyy

# 从零构建知识图谱
graphify .

# 在已有图上查询
graphify query "How does the retrieval pipeline work?"
graphify path "pipeline" "retriever"
graphify explain "RAGEvent"

# 改了代码后增量更新（仅 AST，无 LLM 成本）
graphify update .
```

`graphify-out/graph.json` 存在时优先用上述命令而不是全仓 grep；返回的是 scoped subgraph，比 `GRAPH_REPORT.md` 小很多。如果没装 `graphify`，应用照常启动，只是少了图导航能力。

## PR checklist

- [ ] `make all` 在本地全绿
- [ ] 新增 / 修改的代码有对应测试
- [ ] 如新增 env var，更新 `.env.example` 与 `README.md` Env vars 表
- [ ] 如新增 RAG 能力（retriever / reranker / routing …），同步更新 README 的 "Evolution" 时间线段
- [ ] 如改了 prompt 或 metric，更新 `docs/eval/` 或对应 spec 文档
- [ ] commit message 走 `<type>: <description>` 格式（`feat / fix / refactor / docs / test / chore / perf`）