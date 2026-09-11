# CoEvoSkills-Lite 第二阶段技术设计与实施说明

## 文档信息

| 项目 | 内容 |
|---|---|
| 版本 | v0.1 |
| 日期 | 2026-09-02 |
| 状态 | 第二阶段实现基线，待产品验收 |
| 对应 PRD | KT-CE-01 至 KT-CE-05 |
| 实施范围 | Episode → 局部 Capsule → Candidate Patch → 硬门禁 → 独立 Verifier → 最多两轮修订 |

## 1. 设计结论

第二阶段是显式触发的离线旁路，不自动挂入普通研究运行：

```text
Evolution Episode
    ↓ explicit episode_id
CandidateInputCapsuleBuilder
    ↓ 局部 Trace + 父 Skill 片段 + 少量同类 Episode
Candidate Generator
    ↓ add / delete / replace
Deterministic Guard
    ├─ 越权、跨 Skill、整段改写、无锚点 → reject
    ↓
Independent Semantic Verifier
    ├─ accept_for_replay
    ├─ revise → Generator（总计最多两轮）
    ├─ hold
    └─ reject
    ↓
.runtime/skill-evolution/candidates/<candidate_id>.json
```

该阶段不执行 Candidate、不写 `skills/`、不进入 Shadow/Canary，也不能发布 Stable。`accept_for_replay` 只表示可以交给第三阶段做成对回放。

## 2. 输入胶囊

`CandidateInputCapsule` 只包含一个主 Episode、Episode 引用的局部 Artifact record、目标 Stable Skill 的有界正文片段、局部动作与结果、最多三个同类 Episode，以及允许的修改边界。

路径必须解析在 `artifacts/<run_id>/` 内，缺失记录或越界路径会直接 `hold`。默认 Capsule 上限 24,000 字符，不读取完整聊天或完整网页。

## 3. Candidate Patch 契约

Candidate 保存目标 Skill、父版本、能力粒度、触发条件、前置条件、预期局部改善、风险和来源 Episode。实际修改只能使用最多三个结构化操作：

```text
add：必须提供父 Skill 中存在的 anchor 和新增文本
delete：old_text 必须在父 Skill 中存在，new_text 必须为空
replace：old_text 必须在父 Skill 中存在，new_text 不得为空
```

修改文本默认不超过 1,800 字符。Patch fingerprint 由目标 Skill、父版本、前置条件和标准化操作生成，用于重复候选检测。

## 4. 两级验证

### 4.1 确定性硬门禁

在调用语义 Verifier 前拒绝跨 Skill/父版本、Strategic 修改、缺少触发或前置条件、操作超限、锚点不存在、工具或权限增加、Gate bypass、无变化修订和重复 fingerprint。

### 4.2 独立语义 Verifier

Verifier 不读取 Generator 的隐藏过程，只看到固定规则、原始 Capsule 和当前 Patch。它检查过度泛化、前置条件、父 Skill 冲突、权限与工具、Evidence/Citation 门禁及可回放性。Verifier 不能证明整份报告正确，也无权发布 Candidate。

## 5. 状态与修订

第二阶段终态只有：

```text
accept_for_replay / hold / reject
```

`revise` 是轮内状态。每个 Revision 保存当轮 Patch、确定性问题、Verifier 决定、Token 和是否与上一轮有差异。首版总共最多两轮 Generator-Verifier；第二轮仍要求修订时自动 `hold`。

Atomic Episode 可以显式触发生成；Functional Episode 必须传入人工审查批准；Strategic/observe-only Episode 不调用模型并直接拒绝。

## 6. 存储与隔离

```text
skills/                                      # Stable，Loader 唯一扫描根
.runtime/skill-evolution/episodes/           # Phase 1 输入
.runtime/skill-evolution/candidates/         # Phase 2 不可变审计记录
.runtime/skill-evolution/releases/           # Phase 3 预留
```

同一 Episode 再次调用时直接返回既有 Candidate，不重复调用模型。Candidate 文件使用原子写入；相同 ID 的不同内容会触发冲突。普通 Skill Loader 不扫描 Candidate Store。

## 7. 配置与 Token

默认关闭。关键配置包括 Generator/Verifier tier、两轮上限、Capsule 与 Patch 大小、每次输出上限，以及每个 Candidate 默认 60,000 Token 的总上限。

Candidate 调用绕过模型路由内部重试，由第二阶段直接掌握每个 Provider response 的 usage。每次调用前以 UTF-8 输入字节、完整输出额度和 framing allowance 做保守预检。

## 8. 公共入口

- Python：`ResearchService.generate_skill_candidate(episode_id, human_review_approved=False)`
- CLI：`tools/generate_skill_candidate.py <episode_id>`

两者都要求显式 Episode ID。普通 `ResearchService.run()` 永远不会自动生成 Candidate。

## 9. 实测结果

使用第一阶段受控 arXiv 抓取故障 Episode 运行：

- Generator 首轮提出对 `safe-fetch` 的单条 `add` Patch；
- Verifier 指出过度泛化、Failure Memory 前置条件不足和与父 Skill 回退顺序冲突；
- Generator 第二轮收窄条件并补充检查；
- Verifier 仍认为存在父规则冲突与触发范围问题；
- 达到两轮上限后状态为 `hold`；
- 实际 Token：输入 8,031、输出 2,069，共 10,100；
- Stable Skill 修改与生产加载均为 `false`。

另一次实测发现并修复了模型把字符串列表逐字符拆分的问题。错误 Candidate 已移入隔离 smoke store 的 `quarantine/` 作为审计样本，不再出现在可查询候选目录，也没有进入正式 Store 或 Stable Skill。

## 10. 第二阶段不做

- 不应用 Patch 到运行时 Skill。
- 不评价 Candidate 是否真的改善任务结果。
- 不执行 Stable/Candidate 成对 Replay。
- 不进入 Shadow、Canary 或 Stable。
- 不允许自动晋级或扩大权限。

以上内容由第三阶段 CoEvoSkills 协同进化、局部回放和发布控制面负责；项目不再依赖 SkillOpt 作为独立框架。

## 11. 后续扩展：Verifier 协同进化

项目现已增加独立的 CoEvoSkills 协同进化控制面：Candidate 通过当前代理验证后，可提交局部回放结果；Project Oracle 发现漏判时形成 `VerifierGap`，生成只增不减的版本化 Verifier Suite，并在人工激活后驱动 Skill 子 Candidate 修订。详细契约、配置、公共入口和当前边界见《CoEvoSkills 协同进化技术设计与实施说明》。
