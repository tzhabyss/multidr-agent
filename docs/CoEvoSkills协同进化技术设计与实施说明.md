# CoEvoSkills 协同进化技术设计与实施说明

## 文档信息

| 项目 | 内容 |
|---|---|
| 版本 | v0.1 |
| 日期 | 2026-09-03 |
| 状态 | 实现基线，默认关闭，待产品实测 |
| 前置能力 | SkillX-Lite Episode、CoEvoSkills-Lite Candidate |
| 实施范围 | Candidate 局部回放结果、Project Oracle、VerifierGap、版本化 Verifier Suite、人工激活、Skill 子 Candidate 修订 |

## 1. 目标

原第二阶段只让固定 Semantic Verifier 驱动 Skill Generator 修订。本文增加论文中的第二条进化路径：当 Candidate 已通过代理验证、但在独立的项目局部 Oracle 中失败时，不向 Generator 泄露隐藏答案，而是记录 VerifierGap，生成更强的代理断言，并在人工激活新版 Verifier 后让 Skill Generator 重新修订。

```text
Episode → Candidate → Semantic Verifier → accept_for_replay
                                         ↓
                               Paired Local Replay
                                         ↓
                                  Project Oracle
                         ┌───────────────┴───────────────┐
                       pass                            fail
                         ↓                               ↓
                 ready_for_shadow                  VerifierGap
                                                         ↓
                                               Verifier Suite Candidate
                                                         ↓ 人工激活
                                               Skill 子 Candidate 修订
                                                         ↓
                                                    再次 Replay
```

## 2. 与论文的对应和改造

| CoEvoSkills 原论文 | 项目实现 |
|---|---|
| Surrogate Verifier | 固定 Semantic Verifier + 版本化声明式 Verifier Suite |
| Ground-Truth Oracle | Schema、权限、Evidence、Citation、Search、成本等局部 Project Oracle |
| Surrogate Pass + Oracle Fail | `VerifierGap` |
| Verifier 自主升级测试 | LLM 只生成声明式、只增不减的 `VerifierAssertion` |
| Skill 按新测试继续修订 | 激活 Suite 后生成不可变子 Candidate |

项目 Oracle 只判断可观测的局部性质，不证明整份研究报告的真值或洞察质量。

## 3. 核心契约

### 3.1 CandidateReplayObservation

由隔离回放器或人工评测工具提供：

- Candidate、Episode 和 Replay ID；
- Stable 与 Candidate 的局部指标；
- Schema、权限、Evidence Gate、外部写入等硬检查；
- 可审计 Evidence 引用；
- 明确 `external_writes_allowed=false`。

本模块不伪造运行结果，也不自动重启完整 Deep Research 任务。

### 3.2 ProjectOracleResult

结果为：

```text
pass → ready_for_shadow
hold → 指标不足，继续补充回放证据
fail → Candidate 暂缓，并形成 VerifierGap
```

首版按 Episode 的触发类型选择目标指标。例如：

- `fetch_route_failure`：`fetch_route_failures` 降低或 `fetch_success_rate` 提高；
- `duplicate_query`：重复查询数或重复率降低；
- `extraction_without_binding`：Evidence Binding 增加或未绑定抽取减少；
- `weak_or_rejected_claim`：受支持 Claim 增加或弱 Claim 减少。

### 3.3 VerifierGap

Gap 只保存：

- 失败检查 ID；
- Stable/Candidate 局部指标；
- 可观察失败摘要；
- Evidence 引用；
- 不泄露答案的反馈。

不得保存隐藏答案或要求 Generator 模仿某个 Benchmark 输出。

### 3.4 VerifierAssertion

断言采用无代码、无表达式执行的声明式白名单：

```text
required_hard_check
metric_not_increase
metric_not_decrease
metric_strict_increase
metric_strict_decrease
```

每条断言必须绑定目标 Skill、触发类型和来源 Gap。断言不得申请工具或权限。

### 3.5 VerifierSuite 与 Release

- Suite 文件不可变并保留父版本；
- 新版本继承父版本全部断言，只允许追加；
- Candidate Suite 不参与回放；
- 人工批准产生独立 Release Record 后才成为 Active；
- Verifier 激活不会修改 Stable Skill。

## 4. 信息隔离

| 角色 | 可见输入 | 禁止输入/权限 |
|---|---|---|
| Skill Generator | Episode Capsule、父 Patch、失败摘要、已激活断言 | 隐藏答案、发布权限、外部工具 |
| Semantic Verifier | Capsule、Candidate Patch、固定检查规范 | Generator 思维过程、Stable 发布权限 |
| Verifier Generator | VerifierGap、已有断言、允许的断言类型 | Skill 发布、代码执行、工具与权限扩张 |
| Project Oracle | Replay 指标和硬检查 | Skill/Verifier 生成和修订 |

## 5. 存储隔离

```text
.runtime/skill-evolution/
├─ episodes/
├─ candidates/
├─ replays/
├─ verifier-gaps/
├─ verifier-suites/<suite_id>/<version>.json
├─ verifier-releases/
└─ releases/                    # Stable Skill 发布仍未实现
```

普通 Skill Loader 仍只扫描 `skills/`。

## 6. 公共入口

Python：

```python
service.evaluate_skill_candidate_replay(observation)
await service.generate_verifier_candidate(gap_id)
service.activate_verifier_candidate(
    suite_id, version, human_review_approved=True
)
await service.revise_skill_candidate_from_verifier_gap(gap_id)
```

CLI：

```text
tools/run_verifier_coevolution.py evaluate <replay.json>
tools/run_verifier_coevolution.py propose-verifier <gap_id>
tools/run_verifier_coevolution.py activate-verifier <suite_id> <version> --approve
tools/run_verifier_coevolution.py revise-candidate <gap_id>
```

## 7. 配置与安全默认值

```yaml
skill_evolution:
  verifier_coevolution:
    enabled: false
    generator_tier: craft
    max_assertions_per_gap: 3
    generator_max_tokens: 2200
    max_total_tokens_per_gap: 30000
    max_cost_regression_ratio: 0.2
    require_human_activation: true
    stable_skill_auto_promotion: false
```

正常研究任务不会调用该链路。Candidate、Verifier Suite 和 Skill 子 Candidate 均不能自动进入生产。

## 8. 当前边界

本次已经实现：

- 结构化 Replay Observation；
- Project Oracle 局部判断；
- Surrogate Pass/Oracle Fail 的 Gap 记录；
- Verifier Assertion/Suite 的 LLM 生成和确定性白名单解析；
- Suite 不可变存储、父版本和人工激活；
- 激活后按 Gap 生成 Skill 子 Candidate；
- 新 Candidate 重新经过 Patch Guard 和 Semantic Verifier；
- 整条链路默认关闭且不修改 Stable Skill。

尚未实现：

- 自动执行 Stable/Candidate 局部回放；
- Output Verifier 的独立模型 Session；
- Shadow/Canary 流量分配；
- Stable Skill 版本发布与回滚；
- 对 Verifier 新断言的历史正例回归执行器；
- 整个研究运行时的全局 1,000,000 Token 硬上限。

因此本阶段属于“可记录、可生成、可激活、可回到 Skill 修订”的协同进化控制面，不代表自动发布已经完成。

