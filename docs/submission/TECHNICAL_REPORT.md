# Evidence-Driven Deep Research Agent 技术报告

版本：2026-08-27　|　报告上限：6 页　|　实现语言：Python 3.11+

## 摘要

本项目实现了一个面向开放域研究任务的多轮 Agent 系统。系统以 LangGraph 管理有状态 DAG，以 Planner 生成任务合同和依赖关系，以多个 Research Worker 在私有、有界的 ReAct Skill Loop 中执行 Search、Fetch 与 Extraction，再由 Synthesizer 判断“可写、补证或部分回答”。所有可引用事实进入 `Source -> Snapshot -> Evidence -> Claim -> Citation` 账本，最终由确定性 Citation Gate 检查结构完整性。系统还实现了不依赖 Mem0 调包的分层记忆、Skill/MCP 授权 Harness、Checkpoint/Artifact 审计和成套 Memory 评测。

## 核心贡献

- **DAG + ReAct 混合编排**：全局任务依赖、预算和轮次由 DAG 管理；Worker 在任务内部自主决定工具序列，兼顾可控性与适应性。
- **证据优先输出**：搜索摘要不能直接成为证据，网页必须经过 Safe Fetch、结构化抽取、Claim 绑定与 Citation Gate。
- **自研分层 Memory**：区分热对话、结构化 Segment、RollingState、长期 Fact、Research Episode 与原文 Archive，支持 ADD、REINFORCE、SUPERSEDE、RETRACT。
- **Skill/MCP Harness**：角色、Profile、Schema、超时、回退链与工具白名单统一治理，默认 fail-closed。
- **可审计评测**：提供写入状态演化、混合检索、时序投影、压缩保真、故障注入和全项目回归。

## 适用范围

适合需要多来源检索、证据回溯、长对话约束保持和失败降级的研究任务。它不是无限自治 Agent，也不把“有引用”视为“事实必然正确”；语义充分性仍由模型判断，Python 负责权限、预算、状态和证据结构边界。

<!-- PAGEBREAK -->

## 1. 系统架构与核心流程

### 1.1 执行链路

```text
ResearchRequest
  -> Scope Normalizer
  -> Lead Planner: task contract + DAG + sufficiency targets
  -> Wave Scheduler: dependency-ready tasks + budget leases
  -> Research Worker (parallel)
       -> Strategy
       -> bounded ReAct Skill Loop
       -> Search -> Candidate Select -> Safe Fetch -> Extract
       -> deterministic admission -> Reflection
  -> Synthesizer
       -> ready / need_more / partial
  -> Writer -> Citation Gate -> Artifact + Checkpoint
```

Planner 不直接抓网页，而是生成目标、关键问题、查询、来源偏好、完成条件和预算。Graph 校验依赖、拓扑和可执行 Wave；Worker 只收到自身任务、依赖证据和角色可见 Skill。第一轮建立答案空间，后续轮次只针对 Synthesizer 返回的结构化高优先级缺口补证，避免固定重试和空转。

### 1.2 DAG 与 ReAct 的职责分工

| 层级 | 负责内容 | 强制边界 |
| --- | --- | --- |
| Graph/DAG | 任务依赖、Wave、轮次、Checkpoint、全局预算 | 最大 Worker、Search、Fetch、Deadline |
| Worker ReAct | 查询改写、工具选择、观察、继续或停止 | 私有状态、有界迭代、角色工具白名单 |
| Synthesizer | 证据充分性、冲突、缺口、可写性 | 只能引用账本中可见证据 |
| Python Gate | URL 安全、Schema、Scope、引用映射、Artifact | Fail-closed、原子写入、稳定 ID |

### 1.3 证据链

Search Hit 只是导航候选；只有 Safe Fetch 成功得到正文并完成 Extraction 后，内容才进入 Evidence Ledger。Claim 保存支持语段、来源快照和状态，Writer 只需输出 Claim ID，运行时再解析 Citation 并验证引用映射。该设计把模型擅长的语义判断与代码擅长的权限、结构和可追溯性分离。

<!-- PAGEBREAK -->

## 2. Memory、Skill 与 Harness 设计

### 2.1 分层记忆

| 层 | 载体 | 作用 |
| --- | --- | --- |
| L0/L1 | 当前消息与最近对话 | 保持即时上下文，按预算裁剪 |
| L2 | Conversation Segment + RollingState | Segment 保存不可变历史；RollingState 表示当前约束、决定、未决项 |
| L3 | User/Project Fact + Research Episode | 跨 Session 的偏好、项目决策和研究经验 |
| L4 | MemoryPacket 临时投影 | 按 Planner/Worker/Synthesizer 角色预算组装，不持久化新事实 |
| Archive | 原始消息与完整报告 | 用于恢复、审计和精确值回溯 |

写入由 LLM 提议结构化候选，确定性 WriteGate 再检查来源、敏感信息、Scope、类别和目标关系。Repository 以 Revision 和 Operation ID 实现乐观并发与幂等，支持 `ADD / REINFORCE / SUPERSEDE / RETRACT`。压缩采用“LLM 结构化 Segment + must_keep 原文 Span + RollingState 状态机”，避免单一摘要把旧约束继续留在当前状态。

检索先在 Scope 内过滤，再组合 FTS5、Dense、置信度、时间和强化次数。一次 MemoryPacket 内的 Fact 与 Document 检索共享一个 `PreparedMemoryQuery`，查询向量只编码一次，两路只读召回并行执行；写路径仍同步完成索引提交，保持 Read-Your-Writes。

### 2.2 Skill 与 MCP Harness

每个 Skill 由 `SKILL.md` 操作说明和 `skill.json` 机器契约组成，声明角色、Profile、输入输出 JSON Schema、入口、超时、标签和回退链。Registry 在执行前校验授权和 Schema，未知工具默认拒绝。MCP 工具不会整库暴露，只把显式白名单方法适配为 Skill；`search_hits` 必须进入 Safe Fetch，普通 `observation` 只能用于 Worker 思考。

安全措施包括：外部正文标记为不可信、每次重定向执行 DNS/IP 检查、阻止 localhost/私网/metadata/内嵌凭据、blocked domain 优先、并发 Worker 状态隔离、Artifact 原子写入和 SHA-256 Manifest。

<!-- PAGEBREAK -->

## 3. 实验与测试结果

以下均为项目内部工程评测，不等同于官方外部排行榜成绩。

| 评测 | 数据与设置 | 主要结果 |
| --- | --- | --- |
| 全项目回归 | 单元、集成、MCP、并发、故障、安全 | 339 passed，1 skipped，1 个依赖弃用 Warning |
| Memory Write v4 Live | 158 个完整基础 Turn | Write F1 87.36%；Relation 98.41%；Scope 97.62% |
| 状态机与幂等 | 12,000 步随机演化；20 Worker 同 Operation | 0 状态机失败；20 调用只产生 1 次业务变更 |
| CompactionBench v3 Sealed | 24 条状态转移、8 组 3 次采样、4 条持久化 E2E、17 项故障 | Canonical/Verbatim/must_keep 100%；Ghost 0%；Secret Safe 100%；Field Accuracy 80.95% |
| Temporal RetrievalBench | 10 类生命周期、30 个 Sealed 查询 | Active Coverage 100%；Stale Suppression 100%；Contradictory Packet 0%；Scope Leakage 0%；MRR 0.6074；P95 17.48 ms |
| RetrievalBench v1 | 300 条记忆、100 条查询 | 中文 BM25 Candidate R@50：4.4% -> 74.4%；BGE-M3 Validation nDCG@5 61.0% |

### 3.1 如何解读

Memory Write 的 87.36% 来自真实 Live 基础 Track，较适合作为保守对外指标。后续冻结模型输出上的确定性后处理重放达到更高分数，但属于开发集调优结果，不能替代独立泛化成绩。

Compaction 的关键结论不是“摘要更流畅”，而是新旧状态关系得到正确演化：Rolling Summary 对照的 Ghost Fact 为 26.32%，Full v3 降至 0%。Field Accuracy 仍为 80.95%，说明信息虽然保留，但 `constraint / decision` 分类仍会漂移。

RetrievalBench 证明中文 BM25 表示修复有效，但三个 Embedding 配置仍各有 7 个同 Scope、Active 的困难负例进入 `Forbidden@5`，所以最终发布门禁为 NO-GO。生产仍使用线性混合排序，不宣称 RRF、Null Gate 或 BGE-M3 已上线。

<!-- PAGEBREAK -->

## 4. 完整运行记录与失败案例

### 4.1 两轮 Standard 运行

示例问题为“截至 2026 年 8 月 4 日，日本现任首相是谁？请用官方来源核实”。第一轮 Planner 创建 1 个核心事实任务；Worker 完成 6 次抽取并形成 16 条 Evidence。Synthesizer 判断 `need_more`，指出“目标日期的现任状态”和“改组后的法定任命”两个高优先级缺口。第二轮 Planner 生成 2 个并行补证任务，分别形成 9 和 8 条 Evidence；Synthesizer 转为 `ready`，Writer 生成报告并通过 Citation Gate。

| 项目 | 结果 |
| --- | ---: |
| 研究轮次 / Worker 任务 | 2 / 3 |
| Search / 候选结果 | 15 / 142 |
| Fetch 尝试 / 成功抽取 | 24 / 13 |
| Sources / Evidence / Claims | 12 / 31 / 33 |
| LLM 调用 | Planner 2、Worker 6、Extraction 18、Synthesizer 2、Writer 1 |
| Token | Input 608,587；Output 21,654 |
| 终态 | completed_verified |

完整目录保存 Request、Plan、DAG/Task Runtime、Planner/Worker/ReAct/Search/Fetch/Extraction/Synthesis/Writer Trace、Evidence/Claim/Citation、事件流、审计报告和 SHA-256 Manifest，能够逐步还原 Agent 决策和工具结果。

### 4.2 代表性失败与改进方向

- **检索困难负例**：语义或数字相近的错误记忆仍可能进入 Top-5。计划在限定候选上加入 Cross-Encoder，以及否定、数值、时间和关系一致性特征。
- **来源 Tier 误分**：示例运行中的日本政府页面被透明启发式归入 Tier 3。应维护机构域名注册表，并把“来源性质”和“主题相关性”拆成独立特征。
- **Compaction 字段漂移**：Canonical 信息完整，但约 19% 的目标字段分类不一致。计划增加 typed-slot validator 和字段级修复，不扩大摘要 Prompt。
- **恢复仍需触发器**：硬退出后的 Archive 补偿已证明幂等可恢复，但生产启动阶段尚未自动扫描未完成压缩事务。
- **代码复杂度**：`agents.py` 与 `graph.py` 职责偏重；后续按 Planner、Worker Runtime、Scheduler、Reducer 拆分，并以类型化 Delta 保持行为兼容。

<!-- PAGEBREAK -->

## 5. 局限、改进路线与复现

### 5.1 已知局限

- 内部 Sealed 数据规模有限；24 条压缩案例或 30 个时序查询上的 100% 不代表开放域普遍达到 100%。
- 证据链证明来源和支持语段可追溯，不自动完成事实真伪判定；多个网页也可能共享同一错误来源。
- Worker 是有界 ReAct Agent，受角色授权、迭代和全局预算限制；不是无限自治系统。
- SQLite + 精确向量检索适合当前每 Scope 约 500 条 Fact 的规模，若单 Scope 达到万级，需要 ANN/独立索引服务。
- 生产默认检索策略与实验 RRF/Reranker 保持隔离，只有通过独立 Validation、Sealed 和回滚门禁后才能切换。

### 5.2 优先改进路线

1. 拆分 `ResearchWorker.run()` 和 Graph Scheduler，建立 `WorkerResultDelta / EvidenceDelta` 类型边界。
2. 让生产与 Benchmark 共享 `RetrievalStrategy` 接口，以 Feature Flag 和 Shadow Mode 接入 Reranker。
3. 增加启动时 Compaction Reconciler，自动扫描“已归档但未完成 Segment/State 提交”的事务。
4. 扩充跨语言、否定、数值、时序困难负例和外部 Memory Benchmark，报告置信区间。

### 5.3 最小复现

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,memory,search,browser]"
Copy-Item .env.example .env   # 只在本地填写，不提交
python -m deepresearch doctor
python -m deepresearch run "研究问题" --mode quick --json
python -m pytest -q --basetemp=.test-runtime/submission
```

项目提交时应包含 `deepresearch/`、`evaluation/`、`skills/`、`config/`、`examples/`、`tests/`、`docs/`、`pyproject.toml`、`requirements.txt` 和 `.env.example`；必须排除 `.env`、`.runtime/`、`artifacts/`、SQLite 运行库、模型缓存和真实凭据。

## 结论

该项目的主要价值不在于简单串联搜索与大模型，而在于把 DAG 调度、Worker ReAct、证据账本、分层 Memory、Skill/MCP 权限和可审计评测组合成一个可恢复、可度量、可解释的 Agent Runtime。现有结果证明核心工程链路可用，同时明确披露了检索困难负例、字段分类、来源识别和模块复杂度等未解决问题。
