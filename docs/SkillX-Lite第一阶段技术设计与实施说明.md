# SkillX-Lite 第一阶段技术设计与实施说明

## 文档信息

| 项目 | 内容 |
|---|---|
| 版本 | v0.1 |
| 日期 | 2026-09-02 |
| 状态 | 第一阶段实现基线，待产品验收 |
| 对应 PRD | 《Skill 自进化三阶段融合产品需求书》KT-SX-01 至 KT-SX-05 |
| 实施范围 | 观察、标准化、Episode 沉淀和隔离；不生成 Candidate，不修改 Stable Skill |

## 1. 设计结论

第一阶段采用一条不影响研究决策的旁路：

```text
现有 Research Run
    ├─ Search Sensor
    ├─ Failure Memory
    ├─ Fetch / Extraction Trace
    ├─ AgentLoop Trace
    └─ Claim / Evidence Ledger
              ↓
      SkillEvolutionObserver
              ↓
      EvolutionSignal（统一语义）
              ↓
      EvolutionEpisode（可审查、可定位、可回放输入）
              ↓
  .runtime/skill-evolution/episodes/<run_id>.json
              +
  artifacts/<run_id>/details/skill-evolution/
```

该旁路只在研究流程结束后运行，不参与 Planner、Worker、Synthesizer、Citation Gate 或报告状态判定。观察失败采用 fail-soft，不得使已完成的研究任务失败。

## 2. 当前架构审计

可直接复用的真实能力：

- `SkillRegistry` 已有 `orchestrator / strategy / access` 运行时加载层级。
- `Search Sensor` 已输出零增益、重复查询、重复来源域和来源家族失败指标。
- `Failure Memory` 已记录查询、抓取、抽取、Skill 和 Claim 类失败。
- Fetch、Extraction、AgentLoop 均已有结构化 Trace。
- Claim、Link 和 Evidence Binding 已能表达弱证据、拒绝和待验证状态。
- `ArtifactWorkspace` 已提供运行级原子写入与安全路径校验。

实施前的缺口：

- 运行时加载层级被误当成能力粒度，无法表达“access 层的 Atomic Skill”。
- 各类失败信号没有统一 ID、目标 Skill、版本、范围和证据引用。
- 没有 Evolution Episode 数据契约。
- 没有独立于 Stable Skill 的 Episode 存储。
- 没有默认关闭、观察模式和 fail-soft 集成。

## 3. 双维度 Skill 分类

每个 `SkillDefinition` 同时保存：

| 维度 | 值 | 用途 |
|---|---|---|
| `skill_layer` | orchestrator / strategy / access | 决定角色、加载和提示注入 |
| `capability_granularity` | strategic / functional / atomic | 决定允许形成何种进化产物 |

同时增加显式 `skill_version`。项目内 Skill manifest 使用 `version: 1.0.0`，避免 Episode 只指向一个会变化的名称。

首版边界固定为：

| 能力粒度 | 第一阶段策略 |
|---|---|
| strategic | `observe_only` |
| functional | `episode_requires_review` |
| atomic | `automatic_candidate_allowed` |

“允许自动形成 Candidate”只是写入 Episode 的边界标签。第一阶段没有 Candidate Generator，因此不会实际创建 Candidate。

## 4. 标准信号契约

`EvolutionSignal` 必须包含：

- 确定性 `signal_id`、schema version、时间；
- run、task、window 与适用 scope；
- 目标 Skill、当时版本、运行时层级和能力粒度；
- Candidate 边界、严重度和置信度；
- 有界局部上下文；
- 指向运行产物的 evidence references。

首版映射：

| 标准信号 | 主要来源 | 默认目标 Skill |
|---|---|---|
| `consecutive_no_evidence_gain` | 相邻 Sensor Snapshot / Failure Memory | web-search |
| `duplicate_query` | Sensor Snapshot / Failure Memory | web-search |
| `repeated_source_domain` | Sensor Snapshot | web-search |
| `source_family_repeated_failure` | Sensor Snapshot / Failure Memory | safe-fetch |
| `fetch_route_failure` | Fetch Trace / Failure Memory | safe-fetch |
| `extraction_without_binding` | Fetch Trace / Failure Memory | evidence-extraction |
| `skill_execution_failure` | AgentLoop Observation / Failure Memory | 实际调用 Skill |
| `weak_or_rejected_claim` | Claim、Link、Evidence Binding / Failure Memory | evidence-extraction |

信号由原始 record ID 和目标 Skill 生成确定性 ID，同一观察重复处理不会重复产生语义相同的记录。

## 5. Evolution Episode 契约

第一阶段采用“一条标准信号形成一个 Episode”的保守策略，避免少量轨迹被过早聚合和泛化。Episode 包含：

- run、task、目标 Skill 与父版本；
- 运行时层级、能力粒度和 Candidate 边界；
- 触发信号 ID 与类型；
- scope、局部上下文、已执行动作和局部结果；
- 证据引用；
- `replay_spec`：所需 artifact path、record ID 和 `external_writes_allowed: false`；
- 置信度和创建时间。

Episode 不保存整份聊天或完整网页正文。回放时按引用读取已有运行产物，控制存储量并保持来源可核查。

## 6. 存储与加载隔离

```text
skills/                                  # Stable Skill，现有 Loader 唯一扫描根
.runtime/skill-evolution/episodes/       # 第一阶段 Episode
.runtime/skill-evolution/candidates/     # 第二阶段预留，第一阶段不创建
.runtime/skill-evolution/releases/       # 第三阶段预留，第一阶段不创建
artifacts/<run_id>/details/skill-evolution/  # 每次运行的审计副本
```

中央 Episode Store 使用每个 run 一个 JSON 快照并执行原子写入。相同内容可幂等重写；同一 run 的不同内容会触发冲突，避免静默覆盖历史 Episode。Store 支持按目标 Skill、任务和信号类型检索，审查者无需读取原始聊天即可定位回放输入。

现有 Skill Loader 仍只扫描项目根下的 `skills/`，不会扫描 `.runtime` 或 Candidate 路径。

## 7. 运行时配置

```yaml
skill_evolution:
  mode: "off"                 # off / observe
  store_root: .runtime/skill-evolution
  duplicate_query_threshold: 0.7
  repeated_domain_threshold: 0.65
  max_signals_per_run: 100
```

默认值为 `off`。开启 `observe` 后仅增加观察产物，不改变模型 Prompt、工具列表、研究预算、控制流决策或 Stable Skill。

## 8. 端到端时序

```text
Citation Gate
→ Finalize Run
→ Collect Skill Evolution
    → 检查 mode
    → 读取完成态中的已有 Trace/Ledger
    → 生成 Signals
    → 生成 Episodes
    → 写入隔离 Episode Store
    → 将审计副本放入 State
→ Persist Artifacts
→ End
```

若观察或隔离存储失败，只记录 `skill_evolution_observation_failed`，研究报告仍正常持久化。

## 9. 代码改动

| 模块 | 职责 |
|---|---|
| `deepresearch/skill_evolution/models.py` | Signal、Episode 和边界契约 |
| `deepresearch/skill_evolution/observer.py` | 现有信号归一化与 Episode 构建 |
| `deepresearch/skill_evolution/store.py` | 隔离、幂等、不可变 Episode 存储 |
| `skill_registry.py` / `skill_loader.py` | 双维度分类和 Skill 版本 |
| `runtime_config.py` / `config/runtime.yaml` | 默认关闭的观察配置 |
| `state.py` / `graph.py` / `service.py` | 完成态观察旁路与状态合并 |
| `artifacts.py` | 运行级 Signal、Episode 和观察结果 |
| `skills/*/skill.json` | 当前 Skill 的显式能力粒度与版本 |

## 10. 测试与验收映射

| PRD 验收项 | 验证方式 |
|---|---|
| 关闭功能时行为不变 | Runtime 默认和项目配置均为 `off`；关闭时不增加 Skill Evolution 审计段 |
| 开启后形成可追溯 Episode | 端到端运行验证 artifact 副本和中央 Store |
| 定位 Skill、版本、任务、证据 | Signal/Episode schema 单测 |
| Strategic 不进入自动 Candidate | `observe_only` 边界单测 |
| Candidate 不被普通 Loader 加载 | 存储根与 `skills/` 分离；第一阶段不创建 Candidate 目录 |
| 信号覆盖满足首版要求 | 八类来源组合单测 |
| 幂等且不覆盖历史 | Episode Store 幂等和冲突测试 |

## 11. 第一阶段明确不做

- 不调用 LLM 生成或概括 Skill。
- 不生成 Candidate Patch。
- 不修改任何 `SKILL.md` 正文。
- 不把 Episode 注入普通研究 Prompt。
- 不执行在线回放、Canary、Stable 晋级或回滚。
- 不改变 Planner、Writer、权限或工具集合。

## 12. 下一审查点

第一阶段产品验收只审查信号质量、可追溯性、噪声水平和隔离性。通过后再设计第二阶段的局部输入胶囊、Candidate Patch 和独立 Verifier；不得直接从当前 Episode Store 自动写入 Stable Skill。
