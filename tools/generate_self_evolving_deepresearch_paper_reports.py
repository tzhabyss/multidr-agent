from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "tmp" / "pdfs" / "self-evolving-deepresearch"
ASSET_ROOT = WORK_ROOT / "all-assets"
MANIFEST = WORK_ROOT / "text" / "manifest.json"
NOTE_ROOT = ROOT / "docs" / "paper-reading" / "self-evolving-deepresearch"
PDF_ROOT = ROOT / "output" / "pdf" / "self-evolving-deepresearch-paper-analysis"
TODAY = "2026-08-27"


PAPERS = [
    {
        "id": "2605.23904",
        "order": 1,
        "slug": "skillopt",
        "short": "SkillOpt",
        "title": "SkillOpt: Executive Strategy for Self-Evolving Agent Skills",
        "direction": "受控文本优化与防漂移",
        "fit": 10.0,
        "read": "第一优先级：直接决定候选 Skill 应该怎样修改、拒绝和保留。",
        "thesis": "把冻结 Agent 外部的 Skill 文档视为可训练状态，但每次只能执行有边界的 add/delete/replace 编辑，并由独立验证集决定是否接受，从而把随意反思改写变成可审计的文本优化。",
        "plain": "普通做法像是让模型每次复盘后重写操作手册；SkillOpt 更像 Git 小步提交：每次只改一点，跑回归，变好才合并，失败的提交也记住，防止下次重复犯同一个优化错误。",
        "findings": [
            "跨 6 个 benchmark、7 个目标模型和 3 种 Harness 的 52 个单元中，SkillOpt 全部最优或并列最优。",
            "GPT-5.5 相对无 Skill 的平均绝对提升为：direct chat +23.5、Codex +24.8、Claude Code +19.1 个百分点。",
            "去掉 rejected-edit buffer 会造成多个任务退化；同时去掉 meta skill 和 slow update 时，SpreadsheetBench 从 77.5 降至 55.0。",
            "最终 Skill 仍然紧凑，6 个任务中约 379 到 1,995 tokens；优化器只在离线阶段调用，不增加部署时模型调用数。",
        ],
        "method": [
            "Forward pass：冻结目标 Agent 使用当前 Skill 执行任务，收集带分数的 rollout 证据。",
            "Backward pass：优化器按小批次归纳可复用的成功规律和失败原因。",
            "Bounded update：只允许 add/delete/replace，并使用 textual learning-rate 控制本轮改动量。",
            "Validation gate：候选版本只有在独立 selection split 严格提升时才成为 best skill。",
            "Rejected-edit buffer：保存被拒绝的编辑及原因，避免振荡和重复尝试。",
            "Slow/meta update：跨 epoch 总结稳定方向，避免只追逐最近一批 rollout。",
        ],
        "experiments": [
            {"name": "实验1：跨模型、任务与Harness主实验", "what": "将 SkillOpt 与无 Skill、人工 Skill、一次生成、Trace2Skill、EvoSkill、TextGrad、GEPA 等逐单元比较。", "why": "验证受控文本优化是否只对某一模型或某一执行框架有效。", "setup": "SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMath、ALFWorld；7 个 GPT/Qwen 模型；direct chat、Codex、Claude Code；selection 与 test 严格分离。", "result": "52/52 单元最优或并列最优；GPT-5.5 direct chat 从 58.8 提升到 82.3，程序性任务增益最大。", "conclusion": "外部 Skill 可以像参数一样被系统优化，但收益必须由独立验证门控。"},
            {"name": "实验2：训练证据量与批大小", "what": "改变训练样本比例、reflection mini-batch 和 rollout batch。", "why": "判断结果是否依赖脆弱的批大小或少量偶然轨迹。", "setup": "GPT-5.5；SearchQA、SpreadsheetBench、LiveMath；训练比例从 1% 到 100%，多档 batch。", "result": "SpreadsheetBench 随证据量从 47.5 升至 78.0，LiveMath 从 59.1 升至 70.5；batch 变化总体稳定。", "conclusion": "足够的评分证据重要，但具体批大小不是主要增益来源。"},
            {"name": "实验3：有界编辑与稳定器消融", "what": "改变文本学习率、调度器，并移除 rejected buffer、meta skill、slow update。", "why": "确认系统是在稳定进化，而不是反复漂移。", "setup": "多档编辑预算及 constant/cosine/linear 调度；对照无边界更新和组件移除。", "result": "中等编辑预算持续优于无预算改写；去掉 rejected buffer 分别下降 1.6/4.6/2.4；去掉 meta+slow update 时 SpreadsheetBench 下降 22.5。", "conclusion": "小步修改、拒绝记忆和慢更新是稳定性的核心。"},
            {"name": "实验4：跨模型、Harness与相邻任务迁移", "what": "把一个环境优化出的 Skill 迁移到不同模型、不同 Harness 和相邻数学 benchmark。", "why": "判断 Skill 是可复用程序知识还是某个环境的 Prompt 补丁。", "setup": "GPT-5.4 到 mini/nano；Codex 与 Claude Code 双向迁移；OlympiadBench 到 Omni-MATH。", "result": "跨模型四个迁移均为正；Codex 到 Claude Code 的 Spreadsheet Skill 带来 +59.7，反向 +43.6；跨数学 benchmark 增益较小但均为正。", "conclusion": "部分学到的是结构化程序知识，但跨任务收益明显小于同域迁移。"},
            {"name": "实验5：Skill长度、训练成本与定性案例", "what": "统计最终文档长度、接受编辑数、训练 token 与代表性规则。", "why": "确认效果不是靠无限拉长上下文获得。", "setup": "6 个 benchmark 的 best_skill.md 和编辑历史。", "result": "最终长度 379 到 1,995 tokens，中位约 920；学到的内容多为工具使用、格式约束和验证步骤。", "conclusion": "高价值 Skill 应紧凑且可检查，优化器无需在运行时常驻。"},
        ],
        "limits": [
            "依赖可重复执行、可评分且有独立 selection split 的任务，开放式研究很难直接满足。",
            "52 个单元来自同一套论文实现和评分协议，不能等同所有真实业务环境。",
            "严格提升门禁可能偏向容易量化的能力，忽视洞察质量等软目标。",
        ],
        "integration": [
            "新增 CandidatePatch，只允许 add/delete/replace，不允许自动全文覆盖 SKILL.md。",
            "用现有 FailureRecord、CoverageChange、ClaimEvidenceLink 生成 patch 理由和证据引用。",
            "把整体准确率替换为局部指标向量：引用覆盖、无支持 Claim、来源权威性、重复搜索、成本和回归数。",
            "建立 rejected-edit buffer；被拒绝的修改仍保留 diff、触发信号和拒绝原因。",
            "正式 skills/ 保持只读；候选先进入 .runtime/skill-candidates，再由回放和 Canary 晋升。",
        ],
        "dont": ["没有微测试或回放集时，不要照搬 strict-improvement 自动发布。", "不要让同一个生成器同时修改 Skill、选择指标并判定自己成功。"],
        "mvp": ["定义 SkillSignal 与 CandidatePatch。", "限制每轮修改字符数和操作种类。", "在 3-10 个历史任务上比较局部指标。", "保留拒绝编辑和回滚版本。"],
        "scores": [9.3, 9.2, 9.4, 10.0, 9.8],
    },
    {
        "id": "2512.23760",
        "order": 2,
        "slug": "audited-skill-graph",
        "short": "Audited Skill-Graph",
        "title": "Audited Skill-Graph Self-Improvement for Agentic LLMs via Verifiable Rewards, Experience Synthesis, and Continual Memory",
        "direction": "可审计晋升、证据包与回滚",
        "fit": 9.6,
        "read": "第一优先级：当作治理架构蓝图阅读，不要把它当成已有充分实验验证的方法。",
        "thesis": "自改进不应隐藏在模型权重或不可复现的反思中，而应编译成带接口、验证器、证据包和版本关系的 Skill Graph；只有可回放验证通过的候选能力才能晋升。",
        "plain": "它像给自进化系统加一套软件发布流程：新能力先打包、写接口、跑测试、保存证据，再进入仓库。出问题时能追到是哪条轨迹、哪个验证器和哪个版本导致的。",
        "findings": [
            "论文提出技能编译、验证器支持奖励、审计证据包、经验合成和持续记忆控制的完整架构。",
            "参考实现能够记录结构有效性、结果正确性、复用奖励和步骤惩罚，并将验证事件写入追加式审计目录。",
            "论文核心贡献是系统设计与安全论证；没有报告与主流基线的大规模完整对比结果。",
            "作者明确指出：验证器或证据存储一旦被策略运行时污染，审计可能只剩表面形式。",
        ],
        "method": [
            "从成功轨迹提取候选 Skill，并规范化为显式接口、前置条件和后置条件。",
            "用可回放 verifier 和 contract check 决定是否晋升，而不是相信 Agent 自评。",
            "把奖励拆成可从日志重建的组件，减少不可解释总分。",
            "用有向 Skill Graph 表示依赖、组合和版本关系。",
            "用 append-only evidence bundle 保存程序、轨迹、验证结果、时间和稳定哈希。",
            "通过经验合成扩展压力测试，并周期性回放已晋升 Skill 检测遗忘和回归。",
        ],
        "experiments": [
            {"name": "原型记录1：可运行参考实现", "what": "实现 Skill 编译、程序验证、奖励分解和 JSON 审计轨迹。", "why": "证明架构不是只有概念图，至少能形成可重放的证据包。", "setup": "确定性工具环境与可选本地模型策略；验证器独立于模型内部推理。", "result": "论文展示了验证报告、程序稳定哈希、追加式审计记录和可重建奖励组件；未给出大规模对比数字。", "conclusion": "实现层可行，但不能据此推断在开放式 Deep Research 上已经有效。"},
            {"name": "评测计划2：能力增长与归因", "what": "提出把成功拆成直接推理、已验证 Skill 复用和已验证组合。", "why": "避免最终分数上升却无法证明 Skill 的因果贡献。", "setup": "追踪新晋升 Skill 数、held-out 复用率、工具参数正确率和输出利用一致性。", "result": "这是论文提出的指标体系，不是已完成实验结果。", "conclusion": "你的项目应新增 Skill 使用到证据增量的归因账本。"},
            {"name": "评测计划3：持续流中的保留与回归", "what": "周期性重放已晋升 Skill 的固定验证套件。", "why": "检查新版本是否破坏旧能力或因环境变化失效。", "setup": "类似持续任务流，记录策略漂移和工件稳定性。", "result": "论文提出实施方案，尚未给出完整长期曲线。", "conclusion": "版本晋升必须同时满足新能力增益和旧能力不回归。"},
            {"name": "评测计划4：约束违反、奖励投机与审计完整性", "what": "在诱发违规的环境中检查 reward hacking、伪造工具轨迹和契约绕过。", "why": "自进化可能学会优化可见分数而不是用户真实目标。", "setup": "证据重建率、验证器跨版本复现性、Harness 敏感性和证据完整性。", "result": "属于威胁模型和未来评测建议。", "conclusion": "在你的系统中，生产运行时不得拥有改写验证证据和发布日志的权限。"},
        ],
        "limits": [
            "主要是架构论文与参考原型，缺少充分的端到端基准对比。",
            "回放要求环境相对确定；网页、搜索结果和模型输出都具有时间与随机性。",
            "前后置条件过松会漏掉边缘失败，过严则会阻止有价值 Skill 晋升。",
            "证据包可能包含敏感信息，需要访问控制、最小化和保留期限。",
        ],
        "integration": [
            "将 SkillVersion、SkillRelease、SkillEvidenceBundle 作为独立控制面模型。",
            "每个候选必须绑定 parent_version、signal_ids、run_ids、verifier_version 和 rollback_version。",
            "复用现有 append-only ObservationDelta 和审计日志，但生产 Agent 只读发布决策。",
            "把非确定性网页任务降级为统计回放或局部契约测试，不要求逐 token 重现。",
            "建立 canary、stable、deprecated 状态和签名 denylist。",
        ],
        "dont": ["不要把论文的 Evaluation Plan 写成已验证实验结论。", "不要允许生成器控制 verifier、证据库和发布状态。"],
        "mvp": ["SkillVersion 状态机。", "追加式证据包。", "固定微测试回放。", "发布审批和一键回滚。"],
        "scores": [8.2, 7.8, 4.8, 9.8, 8.5],
    },
    {
        "id": "2508.06433",
        "order": 3,
        "slug": "memp",
        "short": "MemP",
        "title": "Memp: Exploring Agent Procedural Memory",
        "direction": "程序性记忆的构建、检索、更新与废弃",
        "fit": 9.7,
        "read": "第一优先级：解决少样本经验先存在哪里，以及何时不该立即变成 Skill。",
        "thesis": "Agent 经验不应只保存原始轨迹，也不应全部立即固化为 Skill；更有效的做法是同时保留具体轨迹与抽象脚本，并研究构建、检索和更新策略。",
        "plain": "轨迹像完整录像，脚本像操作手册。录像保留细节但容易过拟合，手册更能迁移但可能漏细节；MemP 发现两者结合通常最好。",
        "findings": [
            "在 TravelPlanner 和 ALFWorld 上，Script、Trajectory、Proceduralization 均优于无记忆；组合具体轨迹和抽象脚本通常最强。",
            "GPT-4o 在 ALFWorld test 上从 42.14 提升到 77.86，平均步骤从 23.76 降至 15.01。",
            "用 query 或关键事实进行语义检索优于随机取样；检索过多记忆后性能会回落。",
            "强模型生成的程序性记忆可迁移给弱模型；Qwen2.5-14B 在 TravelPlanner 完成率提升约 5%，步骤减少 1.6。",
        ],
        "method": [
            "Build：原始成功轨迹、抽象 Script，或两者组合的 Proceduralization。",
            "Retrieve：随机、Query embedding、AveFact 关键事实向量匹配。",
            "Update：直接追加、只保留成功轨迹的 Validation、失败后结合原记忆进行 Adjustment。",
            "Lifecycle：新增、删除和修改程序性记忆，而不是只增不减。",
        ],
        "experiments": [
            {"name": "实验1：Build Policy", "what": "比较无记忆、Script、Trajectory 和 Proceduralization。", "why": "判断原始经验与抽象规则各自适合什么任务。", "setup": "TravelPlanner、ALFWorld；GPT-4o、Claude-3.5-Sonnet、Qwen2.5-72B。", "result": "所有记忆形式整体优于无记忆；GPT-4o 的 Proceduralization 在 ALFWorld dev/test 达 87.14/77.86，步骤 15.01。", "conclusion": "具体案例和高层规则互补，少样本阶段不必急着二选一。"},
            {"name": "实验2：Retrieve Policy", "what": "比较随机、Query key 和 AveFact key。", "why": "验证记忆收益是否依赖正确召回。", "setup": "TravelPlanner，三种模型，向量相似度检索。", "result": "AveFact 在三种模型上总体最好；例如 Qwen2.5-72B 的 Hard Constraint 从无记忆 7.34 提升至 12.66。", "conclusion": "检索键应表达任务核心约束，而不只是完整问题文本。"},
            {"name": "实验3：Memory Update", "what": "比较 Vanilla append、成功过滤 Validation 和失败驱动 Adjustment。", "why": "测试持续更新会改善还是污染记忆库。", "setup": "任务分组按序执行，每组后更新记忆，跟踪奖励增益与步骤减少。", "result": "三种策略均随任务组改善；Adjustment 最强，最终领先次优约 0.7 奖励点并减少 14 步。", "conclusion": "失败可以用于修补，但前提是失败与原记忆之间存在明确关联。"},
            {"name": "实验4：与记忆基线比较", "what": "在 ALFWorld 对比 ReAct、ExpeL、AWM 和 MemP。", "why": "确认完整 Build-Retrieve-Update 体系是否优于已有记忆机制。", "setup": "GPT-4o，dev/test 成功率和步骤。", "result": "MemP 获得最高 dev/test 成功率和最低平均步骤。", "conclusion": "程序性记忆同时改善效果和试错效率。"},
            {"name": "实验5：跨模型迁移", "what": "把 GPT-4o 形成的记忆交给 Qwen2.5-14B。", "why": "验证经验是否独立于产生它的模型。", "setup": "TravelPlanner 与 ALFWorld。", "result": "TravelPlanner 完成率提升约 5%，步骤减少 1.6；ALFWorld 也呈现相似趋势。", "conclusion": "外部程序知识可成为强模型到弱模型的轻量迁移载体。"},
            {"name": "实验6：检索规模", "what": "增加单次取回的程序性记忆数量。", "why": "判断更多经验是否总是更好。", "setup": "GPT-4o 在 ALFWorld 上改变检索条数。", "result": "性能先上升后平台，检索过多开始下降。", "conclusion": "必须控制上下文和噪声，记忆库规模不能直接等于注入规模。"},
        ],
        "limits": ["基准具有明确环境奖励，与开放式 Deep Research 的弱监督不同。", "许多构建策略依赖 gold 或成功轨迹。", "论文中的 memory 与工程化 Skill 边界仍较宽，没有生产权限和版本治理。"],
        "integration": [
            "保留三层：Episode 原始轨迹、Procedural Memory 经验、Stable Skill 正式能力。",
            "用户修改和单次异常先进入 Procedural Memory，不直接更新 skills/。",
            "检索键使用 task_type、requirement、entity、source_family 和 failure_signature 的组合。",
            "当记忆重复出现、可操作且具备微测试时才提出 SkillHypothesis。",
            "为程序性记忆增加 revise、deprecate 和 expires_at，而不是无限追加。",
        ],
        "dont": ["不要把成功轨迹直接当通用 Skill。", "不要一次注入大量相似记忆。", "不要把事实记忆、用户偏好和可执行步骤混成一个仓库。"],
        "mvp": ["Episode 与 ProceduralMemory 分表。", "从现有 FailureRecord 形成轻量经验。", "重复两次后才进入 postmortem。", "满足契约后再创建 SkillHypothesis。"],
        "scores": [8.2, 8.1, 7.8, 9.8, 9.0],
    },
    {
        "id": "2608.01913",
        "order": 4,
        "slug": "search-failure-diagnosis",
        "short": "长程搜索失败诊断",
        "title": "Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents",
        "direction": "Deep Research 局部失败信号",
        "fit": 10.0,
        "read": "第一优先级：它直接回答你的问题——没有整体失败标签时，可以观察什么。",
        "thesis": "搜索投入与答案质量只弱相关；决定结果的是是否取回了关键证据，以及取回后是否正确利用。错误应拆成 Retrieval Gap 和 Utilization Gap，而不是统一归为研究失败。",
        "plain": "有的报告错，是因为根本没找到材料；有的报告错，是材料已经在上下文里却没用对。这两种病不能都用“继续搜索”治疗。",
        "findings": [
            "在同一问题上，找到 gold evidence 的 Agent 正确率为 73.8%，未找到者只有 2.2%。",
            "77.5% 到 93.6% 的搜索 episode 没有增加新证据；真正 productive 的只有约 6% 到 23%。",
            "错误轨迹比正确轨迹长 1.9 到 2.9 倍，但最后一次新增证据通常只晚 0.6 到 6.7 个 episode。",
            "上下文中搜索摘要占 66% 到 85%，访问页面正文只占 2% 到 16%；无限搜索主要在堆积摘要。",
            "开放网页复现实验中，retrieval gap 占错误的 68% 到 88%，accuracy 与 coverage 的相关系数为 0.99。",
        ],
        "method": [
            "固定 retriever 和评测 Harness，比较 6 个长程搜索 Agent，避免检索器差异混入 Agent 行为。",
            "利用人工文档相关性判断，逐步计算 gold recall、evidence coverage 和 episode 增量。",
            "将错误拆成 RG：从未达到充分证据；UG：曾达到充分证据但答案仍错。",
            "分析 productive、redundant、unproductive episode，定位有效阶段和 wasted tail。",
            "在 live open web 上用独立 sufficiency judge 做方向性复现。",
        ],
        "experiments": [
            {"name": "RQ1：什么信号最能跟踪正确率", "what": "比较搜索次数、context、检索 recall 和访问精度与正确率。", "why": "找到不依赖最终答案标签的过程代理。", "setup": "BrowseComp-Plus 830 题、6 个 Agent、同一 Qwen3-Embedding-8B retriever。", "result": "累计 gold recall 明显优于搜索量；最弱 Agent 反而搜索最多。", "conclusion": "搜索预算消耗不是质量指标，证据覆盖才是核心。"},
            {"name": "RQ2：Retrieval Gap 与 Utilization Gap", "what": "把每个错误按是否曾取回 gold evidence 分类。", "why": "区分应该修查询还是修证据利用。", "setup": "用 qrels 重建每条轨迹的证据可得性。", "result": "同题 gold-hit disagreement 中，命中者正确 73.8%，未命中者 2.2%；多数错误属于 RG。", "conclusion": "先诊断证据是否出现，再决定补搜或重做 Claim-Evidence 对齐。"},
            {"name": "RQ3：episode级机制与浪费尾巴", "what": "标记 productive/redundant/unproductive，并比较正确与错误轨迹。", "why": "判断 Agent 在什么时候停止获得信息。", "setup": "逐 episode 计算新增相关文档和最后增益点 k*。", "result": "77.5%-93.6% episode 无新增证据；错误轨迹长 1.9-2.9 倍，浪费尾巴长 3.0-6.1 倍。", "conclusion": "应分别设计饱和停止和无证据时的查询重构。"},
            {"name": "RQ4：查询策略", "what": "分析 repeat、pivot、add/drop constraint 和并行 batching。", "why": "定位哪些搜索动作体现策略质量。", "setup": "用相邻查询 token overlap 分类，跨 Agent 做描述性相关。", "result": "repeat 与准确率相关为 -0.83；batching 为 +0.75；pivot 本身并不坏。", "conclusion": "问题不是换角度，而是重复和新角度无法落到证据。"},
            {"name": "RQ5：同题控制下的机制检验", "what": "用 question fixed effects 分离模型能力和行为差异。", "why": "避免只在 6 个 Agent 均值上做脆弱相关。", "setup": "4,919 条可用 agent-question 轨迹；变量标准化，标准误按问题聚类。", "result": "gold recall 每 1 个标准差对应约 +30pp 正确率，t=34；控制 recall 后吸收强 Agent 优势的 56%-60%。", "conclusion": "检索覆盖是主要中介，很多搜索行为通过影响 recall 起作用。"},
            {"name": "RQ6：开放网页外部有效性", "what": "在 live commercial web search 上复现过程。", "why": "确认结论不是固定语料和 qrels 的产物。", "setup": "BrowseComp-en 200 题，4 个完成预算内运行的 Agent，LLM sufficiency judge。", "result": "accuracy-coverage r=0.99，accuracy-calls r=0.12；RG 占错误 68%-88%。", "conclusion": "方向性结论复现，但只有 4 个 Agent，统计力度有限。"},
        ],
        "limits": ["主实验是短答案检索，不是长报告写作。", "开放网页部分用 LLM judge 判断充分性，且只有 4 个 Agent。", "qrels 中的 gold 文档不一定覆盖真实世界所有有效证据。", "相关与固定效应分析仍不能完全证明某种查询动作具有因果作用。"],
        "integration": [
            "在 FailureRecord 增加 retrieval_gap、utilization_gap、coverage_gap、saturation_without_evidence。",
            "RG 触发 query reformulation、entity disambiguation、source-family switch；UG 触发 evidence extraction、claim linking 和 contradiction review。",
            "Search Sensor 同时跟踪新增 EvidenceRecord、新增 Binding、CoverageChange，而不是只看 search/fetch 数量。",
            "记录 last_evidence_gain_step，超过阈值后区分：已有充分证据则停止，没有则换策略。",
            "对搜索摘要做去重、老化和淘汰，避免 snippet 填满上下文。",
        ],
        "dont": ["不要把长轨迹直接判为努力充分。", "不要把所有错误都交给 Planner 增加更多搜索。", "不要用一个 LLM Judge 的总分替代过程信号。"],
        "mvp": ["增加 RG/UG 分类器。", "记录 productive episode 和 last_gain_step。", "为两类 Gap 配置不同 corrective_route。", "在审计报告显示浪费尾巴和证据覆盖曲线。"],
        "scores": [9.0, 9.1, 9.0, 10.0, 9.8],
    },
    {
        "id": "2604.01687",
        "order": 5,
        "slug": "coevoskills",
        "short": "CoEvoSkills",
        "title": "CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification",
        "direction": "Generator-Verifier隔离共进化",
        "fit": 9.2,
        "read": "第二优先级：借鉴角色隔离和代理验证器，不要照搬隐藏真值假设。",
        "thesis": "复杂 Skill 是多文件工件，单次生成不可靠；应让 Skill Generator 与独立 Surrogate Verifier 在信息隔离下循环打磨，并只让隐藏 oracle 返回不泄题的通过/失败信号。",
        "plain": "一个 Agent 写操作包，另一个 Agent 专门设计测试和挑错；测试通过后再交给隐藏验收。写的人看不到标准答案，减少为了过测试而投机。",
        "findings": [
            "在 SkillsBench 上达到 71.1% pass rate，比无 Skill 高 40.5pp，比人工 Skill 高 17.6pp。",
            "移除 Surrogate Verifier 后从 71.1% 降至 41.1%；只有背景上下文、不做演化时为 42.4%。",
            "Opus 演化出的 Skill 迁移给另外 6 个模型时，相对无 Skill 获得约 35-44pp 提升。",
            "平均每个任务 4.1 次验证循环、2.4 次 Ground Truth Oracle 调用；超过 60% 在 2 次 oracle 内收敛。",
        ],
        "method": [
            "Generator 创建完整多文件 Skill 包并在沙箱中执行。",
            "Surrogate Verifier 根据任务说明、可见输入和输出工件生成测试与结构化故障诊断。",
            "代理测试失败时固定测试并修补 Skill；代理测试通过才调用隐藏 oracle。",
            "Oracle 只返回 opaque pass/fail；失败时由 Surrogate 自主升级测试，而不是泄露真值。",
            "达到满分或预算后保存最佳 Skill，再部署给目标 Agent。",
        ],
        "experiments": [
            {"name": "RQ1：Skill质量主比较", "what": "比较无 Skill、一次生成、CoT 生成、官方 Skill-Creator、人工 Skill 和 CoEvoSkills。", "why": "验证增益来自验证闭环还是更强生成 Prompt。", "setup": "SkillsBench，Claude Opus 4.6 + Claude Code；多次运行。", "result": "CoEvoSkills 71.1%；人工 53.5%；Skill-Creator 34.1%；一次生成 32.0%；无 Skill 30.6%。", "conclusion": "多文件 Skill 的关键不是生成一次，而是获得独立、可操作的验证反馈。"},
            {"name": "RQ2：跨模型迁移", "what": "把 Opus 演化 Skill 交给来自 5 家厂商的 6 个额外模型。", "why": "检查是否过拟合生成模型。", "setup": "不同模型使用各自 Agent Harness 进行隐藏 oracle 评估。", "result": "所有模型相对无 Skill 提升约 35-44pp；GPT-5.2 自演化 69.8%，用 Opus Skill 为 65.0%。", "conclusion": "大部分流程可迁移，但模型匹配演化仍有小幅优势。"},
            {"name": "RQ3：领域分解", "what": "按 11 个专业领域比较无 Skill、人工 Skill 和自演化 Skill。", "why": "检查平均数是否掩盖领域差异。", "setup": "SkillsBench 领域划分。", "result": "自演化 Skill 在 9/11 领域超过人工；Finance 高 56.9pp，Cybersecurity 高 23.2pp。", "conclusion": "人工直觉与模型执行习惯存在错位，但结果仍受各领域任务数影响。"},
            {"name": "演化动态", "what": "跟踪 0-5 轮后的 pass rate 和验证次数。", "why": "确认收益是否来自迭代而非初始上下文。", "setup": "最多 5 次 oracle、15 次 surrogate repair。", "result": "round 0 约等于无 Skill，round 2 为 44%，round 3 为 63%，round 5 约 75%；平均 4.1 次验证循环。", "conclusion": "主要增益在前几轮出现，但失败任务集中在高迭代区。"},
            {"name": "组件消融", "what": "移除 Surrogate Verifier 或整个演化过程。", "why": "测量结构化诊断和迭代各自贡献。", "setup": "Claude Opus 4.6 + Claude Code；因成本为单次消融。", "result": "完整 71.1；无 Surrogate 41.1；只有 Context 42.4；无 Skill 30.6。", "conclusion": "验证器贡献很大，但单次消融和隐藏 oracle 条件限制外推。"},
        ],
        "limits": ["最终仍依赖可执行的隐藏 Ground Truth Oracle，开放研究通常没有。", "代理验证器由 LLM 生成测试，可能共享盲点或制造伪约束。", "主要任务是可自动验收的制品生成，不等于长报告洞察。", "高成本消融只有单次运行。"],
        "integration": [
            "保留 Generator 与 Verifier 独立上下文，Verifier 不读取生成器思维过程。",
            "把隐藏 oracle 替换为多个局部验证器：引用、Evidence-Link、Coverage、Schema、安全和成本。",
            "Verifier 只输出失败断言、证据和修复建议，不拥有候选发布权限。",
            "代理测试通过后仍进入历史回放和 Canary，而不是直接 stable。",
            "只让 CoEvoSkills 生成候选 Skill 包，不让其改变正式 Registry。",
        ],
        "dont": ["不要声称 Surrogate Verifier 解决了开放研究真值问题。", "不要让两个 LLM 的一致意见等同事实正确。", "不要为了通过局部测试不断扩大 Skill。"],
        "mvp": ["独立 Candidate Generator。", "确定性证据验证器。", "独立 LLM 结构审查器。", "失败断言驱动一次 bounded patch。"],
        "scores": [9.0, 8.7, 8.6, 9.3, 9.1],
    },
    {
        "id": "2607.15257",
        "order": 6,
        "slug": "searchos-v1",
        "short": "SearchOS-V1",
        "title": "SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration",
        "direction": "Deep Research共享状态与分层Skills底座",
        "fit": 10.0,
        "read": "第二优先级：你的项目已经实现其大量思想，应以实施审计和缺口复核为主。",
        "thesis": "长程搜索失败往往来自进展状态隐含、团队不同步和低收益循环；SearchOS 将任务前沿、证据图、覆盖图和失败记忆外部化，并以流水并行调度、工具中间件和分层 Skill 维持协作。",
        "plain": "它把“Agent 脑子里大概知道搜到哪了”改成所有 Worker 都能看到的研究控制台：还缺什么、证据在哪里、哪条路失败过、谁下一步补哪个格子。",
        "findings": [
            "WideSearch Item F1 为 80.3，比最强基线高 4.3；Row F1 为 56.5，高 2.0。",
            "GISA 的 Set F1 为 76.5，比最强基线高 13.4；对完整枚举任务优势最大。",
            "持续补位调度平均时间减少 24.3%，Slot 利用率 34.6% 提升到 41.7%，Item F1 79.66 提升到 86.75。",
            "分层 Skill 使 Item F1 +2.0、Row F1 +3.4，同时 session time -36.6%、search calls -39.1%、page calls -42.7%。",
        ],
        "method": [
            "把开放信息搜集建模为带引用的 relational schema completion。",
            "SOCM 外部化 Frontier Task、Evidence Graph、Coverage Map 和 Failure Memory。",
            "流水并行调度在 Worker 释放槽位后立即补位，不等待整批完成。",
            "Search Tool Middleware Harness 记录工具事件，并在停滞或预算耗尽时干预。",
            "Strategy Skills 编码搜索策略，Access Skills 编码来源和工具访问方式。",
        ],
        "experiments": [
            {"name": "实验1：WideSearch与GISA主结果", "what": "与 ReAct、Plan-and-Solve 及多 Agent 表格检索系统比较。", "why": "验证共享显式状态能否提高完整性和一致性。", "setup": "WideSearch 200 问、GISA 373 问；GLM-5 Agent、Qwen3.5-35B-A3B 抽取；Max@3。", "result": "WideSearch Item/Row F1 80.3/56.5；GISA Set F1 76.5，领先 13.4。", "conclusion": "优势主要体现在 recall、集合完整性和整行一致性。"},
            {"name": "实验2：动态Schema规划", "what": "比较固定单表、固定多表、二者 oracle 和搜索时动态规划。", "why": "测试信息结构是否应在搜索前固定。", "setup": "40 个任务，Item/Row Precision、Recall、F1。", "result": "即使 oracle 选择更好的固定结构，仍比 SearchOS 低 8.2 Item F1 和 7.7 Row F1。", "conclusion": "Schema 应随发现的实体和关系更新，而不是预先锁死。"},
            {"name": "实验3：流水并行调度", "what": "同查询、模型和并发下比较 batch barrier 与持续补位。", "why": "判断动态补位是否真正改善吞吐而不伤质量。", "setup": "10 个 WideSearch case，3 轮配对；K=8。", "result": "平均时间 629.13s 降至 476.34s，LLM calls 341.4 降至 296.6，Item F1 79.66 升至 86.75。", "conclusion": "持续补位减少空闲并可能让剩余预算更集中于缺口。"},
            {"name": "实验4：Middleware干预案例", "what": "展示早、中、晚停滞时 Loop Sensor 触发策略切换。", "why": "说明控制面如何根据覆盖和实体增量介入执行。", "setup": "WideSearch 代表轨迹，观察 Coverage 和累计实体曲线。", "result": "案例显示传感器能在低进展环中切换策略；属于案例分析而非总体因果估计。", "conclusion": "传感器适合作为局部控制信号，但阈值仍需任务化校准。"},
            {"name": "实验5：分层Skill消融", "what": "同时关闭层级 Strategy/Access Skills。", "why": "测量 Skill 对质量与效率的联合影响。", "setup": "100 个 WideSearch 问题，Max@3。", "result": "Item F1 +2.0、Row F1 +3.4；时间 -36.6%、搜索 -39.1%、页面调用 -42.7%。", "conclusion": "主要价值可能是减少无效探索并提升记录一致性，但联合消融无法分离各层贡献。"},
        ],
        "limits": ["Max@3 报最好结果，会高于单次生产运行期望。", "主要输出是可确定评分的集合/表格，不覆盖所有长文研究。", "Skill 消融同时关闭多层，无法识别具体哪类 Skill 贡献最大。", "你的项目已经实现很多 M1-M12，继续照搬可能重复建设。"],
        "integration": [
            "保留现有 SearchOS M1-M12 主流程，不重构 Graph。",
            "将 CoverageChange、EvidenceRecord、FailureRecord 和 SensorDecision 同步输出为 SkillSignal。",
            "区分运行控制面的短期 Failure Memory 与演化控制面的长期 Evidence。",
            "对现有 Strategy/Access Skill 做独立消融，避免只测所有 Skill 开关。",
            "为每个 Sensor 干预记录触发前后 evidence gain，作为候选 Skill 的局部效用。",
        ],
        "dont": ["不要再实现第二套 Evidence Graph 或 Coverage Map。", "不要把 Max@3 指标当单次线上表现。", "不要从一个低增益波次直接生成全局 Skill。"],
        "mvp": ["ObservationDelta 转 SkillSignal。", "记录传感器干预前后增量。", "按 Strategy/Access 分层消融。", "把失败模式送入候选控制面。"],
        "scores": [9.2, 9.0, 8.5, 10.0, 9.6],
    },
    {
        "id": "2607.07676",
        "order": 7,
        "slug": "skillcenter",
        "short": "SkillCenter",
        "title": "SkillCenter: A Large-Scale Source-Grounded Skill Library for Autonomous AI Agents",
        "direction": "来源绑定、质量门禁与知识缺口",
        "fit": 9.5,
        "read": "第二优先级：重点读来源追踪和真实知识缺口实验，同时注意其质量分和安全局限。",
        "thesis": "Skill 不只是流程，也可能携带模型参数中缺失的、来源可追踪的操作知识；但 Skill 只有在模型存在知识缺口、库中确有相应知识、且检索准确命中时才真正有用。",
        "plain": "仓库大不等于有效。任务需要的知识必须真的缺、库里必须有、路由还必须找对；三件事少一件，塞 Skill 只会增加上下文噪声。",
        "findings": [
            "发布 216,938 个 Skill，其中 114,565 个由 SkillGate 管线生成并进行来源绑定，102,373 个来自社区集合。",
            "真实知识缺口探针中，baseline、placebo 和 title-only keyword retrieval 都为 0%，oracle 注入真实相关 Skill 后为 61%-78%，合并 69%。",
            "在模型本来就能完成的任务上，keyword Skill 与等长无关文本基本相同，部分模型反而显著下降。",
            "全库有 4.53% 文本完全重复、11.52% 处于词法近重复簇；重复主要集中在社区集合。",
            "来源绑定只证明能映射到原文引句，不证明来源正确、最新或权威。",
        ],
        "method": [
            "多来源获取：论文、GitHub、技术网页和论坛。",
            "SkillGate 先判断来源是否可操作，再按模板生成不同 Skill kind。",
            "迭代改进后用精确引句匹配执行 source-grounding check。",
            "Publish gate 检查质量分、占位符、抄袭率和部分许可证规则。",
            "按领域打包为 SQLite FTS5，本地离线检索。",
        ],
        "experiments": [
            {"name": "分析1：Corpus构建与组成", "what": "统计来源、领域、Skill kind 和质量分。", "why": "说明大规模来源驱动管线实际产生了什么。", "setup": "24 个 bundle；管线子集 114,565，社区子集 102,373。", "result": "总计 216,938；约 82% 的受评分 Skill 得分为 4，平均分区分度有限。", "conclusion": "规模很大，但内部单模型质量分不能当作外部真值。"},
            {"name": "分析2：重复度审计", "what": "做主键、规范化文本哈希和 MinHash 近重复检测。", "why": "检查规模是否被复制和模板膨胀。", "setup": "128-permutation MinHash，5-word shingles，Jaccard >=0.8。", "result": "完全重复 9,831 个，占 4.53%；近重复 24,988 个，占 11.52%；主要集中在社区 bundle。", "conclusion": "来源管线较干净，但公开 Skill 生态需要语义去重和版本归并。"},
            {"name": "实验3：模型已具备能力的任务", "what": "比较 baseline、keyword top-3 和等长无关 placebo。", "why": "分离 Skill 内容效果与增加上下文的效果。", "setup": "约 2,000 个算法/数据任务，4 个 solver，配对 McNemar。", "result": "keyword 从未优于 placebo；Claude Haiku 从 baseline 87% 降到 83%，净 -77，p<.001。", "conclusion": "模型已有能力时，泛化 Skill 往往冗余甚至干扰。"},
            {"name": "实验4：合成知识缺口", "what": "用任意、不可猜测的自定义规则构造 260 个任务。", "why": "确认准确相关的 Skill 注入机制本身能工作。", "setup": "规则只存在于 oracle Skill；baseline/placebo/keyword 对照。", "result": "前三者均 0%，oracle Skill 在不同 solver 上达到 72%-100%。", "conclusion": "当模型确实缺知识且 Skill 精确命中时，外部 Skill 很有效。"},
            {"name": "实验5：真实Corpus知识缺口", "what": "选 18 个论文来源 Skill，询问研究中特定参数。", "why": "验证真实 Skill 内容而非合成规则能否补知识。", "setup": "4 solver、72 个配对结果；exact value recall。", "result": "baseline/placebo/keyword 均 0%，oracle 为 61%-78%，合并 69%，p约2e-15。", "conclusion": "库中有真实增量知识，但 title-only FTS 检索是主要瓶颈。"},
            {"name": "分析6：许可与安全基线扫描", "what": "统计许可证并用高精度正则扫描 secrets、注入短语和破坏命令。", "why": "评估发布库的再分发与自动执行风险。", "setup": "全库 216,938；管线与社区子集分开报告。", "result": "风险命中率整体低但集中于社区集合；作者明确说明目前只有质量过滤，没有安全专用门禁。", "conclusion": "来源可追踪不等于安全，可执行前仍需权限和语义审查。"},
        ],
        "limits": ["质量评分来自单一专有 LLM，约 82% 都为 4，区分度不足。", "source grounding 是可追踪性，不是事实核验。", "真实知识缺口只有 18 个窄参数任务，且是单轮 solver。", "社区集合绕过 SkillGate；当前管线没有完整安全筛查。", "网页来源的许可证过滤不是法律清权。"],
        "integration": [
            "在 Skill manifest 中增加 source_url、source_hash、quote、license、retrieved_at、validity_scope。",
            "把现有 EvidenceRecord ID 直接绑定到候选 Skill 中的每条知识性主张。",
            "新增 gap-aware injection：只有检测到模型/任务知识缺口时才注入领域 Skill。",
            "Router 必须检索正文、适用条件和证据字段，不能只搜标题。",
            "来源更新或失效时，把相关 Skill 标为 stale 并触发重新验证。",
        ],
        "dont": ["不要把引句匹配称为事实正确。", "不要把 216K 数量当质量。", "不要直接接入未经安全审核的社区 bundle。", "不要对模型已擅长的任务无差别注入 Skill。"],
        "mvp": ["SkillProvenance Schema。", "候选内容逐条绑定 Evidence ID。", "正文感知检索。", "来源哈希变化触发 stale。"],
        "scores": [8.8, 8.0, 7.4, 9.5, 8.9],
    },
    {
        "id": "2606.03024",
        "order": 8,
        "slug": "skillguard",
        "short": "SkillGuard",
        "title": "SkillGuard: A Permission-Centric Framework for Agent Skill Security",
        "direction": "Skill权限Manifest与运行时完整仲裁",
        "fit": 9.0,
        "read": "第二优先级：自生成 Skill 一旦能调用工具，这篇就从可选变为必需。",
        "thesis": "Skill 是带副作用的可执行制品，安全不能只靠静态扫描；应为 Skill 声明受保护对象、操作和约束，并在每次工具调用前由运行时策略做 allow/confirm/deny。",
        "plain": "SKILL.md 说自己只读文件，不代表运行时真的只读。SkillGuard 像手机权限：清单声明只是第一步，每次访问文件、网络、密钥或进程都要在实际调用点拦截。",
        "findings": [
            "在 1,260 个真实 Skill 上，权限 taxonomy 对预定义 protected object 的覆盖率为 99.93%，组级覆盖 100%。",
            "在三种 Agent 上，contextual injection ASR 从 35.3% 降到 20.7%，obvious injection 从 36.7% 降到 18.0%。",
            "合法任务总体 TSR 从 87.1% 降到 80.6%，存在 6.5pp 效用代价，主要来自 Codex 的 manifest/重规划交互。",
            "平均额外 token 开销为 24.7%；不同 Harness 的时间开销差异很大，Codex +92.1%，Claude Code 反而 -43.0%。",
        ],
        "method": [
            "Permission taxonomy 将文件、代码、网络、进程、秘密、工具、子 Agent 等抽象为 Protected Objects。",
            "SkillManifest 声明权限、effect（allow/confirm/deny）和 workspace_only 等约束。",
            "运行时通过 Agent lifecycle hook 拦截每个 tool call，映射到抽象 capability。",
            "Policy engine 默认拒绝未声明访问，并记录审计日志。",
            "适配层处理不同 Harness 的工具命名、阻断返回和失败语义。",
        ],
        "experiments": [
            {"name": "RQ1：权限分类表达力", "what": "人工标注 SkillsMP 63 类、每类 20 个 Skill。", "why": "检查权限 DSL 能否覆盖现实 Skill 行为。", "setup": "1,260 个 Skill，统计 protected object 和具体 permission。", "result": "784 个含资源操作；1,408 次对象出现中 1,407 次落在预定义对象，覆盖 99.93%，8 个组覆盖 100%。", "conclusion": "对象体系覆盖广，但人工语义标注和真实运行行为仍可能不同。"},
            {"name": "RQ2：注入防御效果", "what": "比较开启和关闭 SkillGuard 时恶意 Skill 的攻击成功率。", "why": "验证 Manifest 加运行时仲裁是否真正阻断越权。", "setup": "SkillInject 50 个安全 Skill 及 contextual/obvious 变体；Codex、Claude Code、Gemini CLI；隔离容器。", "result": "contextual ASR 35.3%->20.7%，obvious 36.7%->18.0%；绝对下降 14.7/18.7pp。", "conclusion": "权限边界有效但没有消除攻击，且某些攻击失败可能来自 Agent 本身未完成任务。"},
            {"name": "RQ3：正常效用保留", "what": "比较合法任务在有无 SkillGuard 下的 TSR。", "why": "检查 deny-by-default 是否让正常工作不可用。", "setup": "124 个 eligible runs，三种 Harness。", "result": "总体 87.1%->80.6%，下降 6.5pp；Gemini 与 Claude Code 不变，Codex 97.6%->78.0%。", "conclusion": "安全和可用性存在 Harness 特定权衡，Manifest 完整性非常关键。"},
            {"name": "RQ4：运行开销", "what": "统计 token 和 wall-clock。", "why": "评估逐次权限判断的生产成本。", "setup": "三种 Agent，对照有无 SkillGuard。", "result": "平均 token +24.7%；Codex 681K->844K，Claude 266K->285K；时间 Codex +92.1%、Gemini +42.3%、Claude -43.0%。", "conclusion": "权限拒绝后的 Agent 重试策略决定大量开销，不能只优化权限引擎本身。"},
        ],
        "limits": ["攻击评估依赖 LLM-as-Judge。", "权限 Manifest 的参考集需要人工编写，自动生成质量会更低。", "三个 Harness 的失败和阻断语义差异很大。", "ASR 显著下降但没有归零；权限系统不是内容安全的全部。"],
        "integration": [
            "在 skill.json 增加 read/write/network/process/secret/subagent 等权限和 scope。",
            "由 agent_loop 的工具调用入口统一做 capability check，而不是依赖 Skill 文本自律。",
            "Access Skill 与 Strategy Skill 使用不同权限：Strategy 默认无外部副作用。",
            "候选 Skill 默认 deny，只有 Manifest、静态扫描和微测试通过后才能 Canary。",
            "记录 permission_denied 后的重试次数，防止 Worker 反复规划造成成本放大。",
        ],
        "dont": ["不要只做正则安全扫描。", "不要自动批准候选 Skill 新增的网络、密钥或写权限。", "不要假设不同 MCP/Harness 的阻断语义相同。"],
        "mvp": ["PermissionManifest。", "tool-to-capability 映射。", "执行前 deny-by-default。", "权限审计与重试上限。"],
        "scores": [8.8, 8.6, 8.4, 9.3, 9.0],
    },
]


def esc(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br/>")


def make_styles() -> dict[str, ParagraphStyle]:
    registerFont(TTFont("MSYH", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    registerFont(TTFont("MSYH-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCN2", parent=base["Title"], fontName="MSYH-Bold", fontSize=23, leading=31, textColor=colors.HexColor("#142D4C"), alignment=TA_LEFT, spaceAfter=10),
        "subtitle": ParagraphStyle("SubtitleCN2", parent=base["Normal"], fontName="MSYH", fontSize=11, leading=17, textColor=colors.HexColor("#4B6577"), spaceAfter=7),
        "h1": ParagraphStyle("H1CN2", parent=base["Heading1"], fontName="MSYH-Bold", fontSize=15, leading=21, textColor=colors.HexColor("#087F8C"), spaceBefore=12, spaceAfter=7),
        "h2": ParagraphStyle("H2CN2", parent=base["Heading2"], fontName="MSYH-Bold", fontSize=11.5, leading=17, textColor=colors.HexColor("#142D4C"), spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("BodyCN2", parent=base["BodyText"], fontName="MSYH", fontSize=9.2, leading=14.5, textColor=colors.HexColor("#22333F"), spaceAfter=5),
        "small": ParagraphStyle("SmallCN2", parent=base["BodyText"], fontName="MSYH", fontSize=7.4, leading=10.8, textColor=colors.HexColor("#587080"), spaceAfter=3),
        "bullet": ParagraphStyle("BulletCN2", parent=base["BodyText"], fontName="MSYH", fontSize=9.0, leading=14, leftIndent=12, firstLineIndent=-8, bulletIndent=2, spaceAfter=4),
        "label": ParagraphStyle("LabelCN2", parent=base["BodyText"], fontName="MSYH-Bold", fontSize=8.4, leading=12.5, textColor=colors.HexColor("#22333F")),
        "callout": ParagraphStyle("CalloutCN2", parent=base["BodyText"], fontName="MSYH", fontSize=10.3, leading=17, textColor=colors.HexColor("#0B4B55"), borderColor=colors.HexColor("#8EC7C9"), borderWidth=0.8, borderPadding=9, backColor=colors.HexColor("#EAF7F5"), spaceAfter=9),
        "caption": ParagraphStyle("CaptionCN2", parent=base["BodyText"], fontName="MSYH", fontSize=7.2, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#587080"), spaceAfter=6),
        "diagram": ParagraphStyle("DiagramCN2", parent=base["BodyText"], fontName="MSYH-Bold", fontSize=9.4, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#142D4C")),
    }


class PaperTemplate(BaseDocTemplate):
    def __init__(self, filename: str, short_title: str, **kwargs: object) -> None:
        super().__init__(filename, pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm, topMargin=18 * mm, bottomMargin=17 * mm, **kwargs)
        self.short_title = short_title
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=self._decorate))

    def _decorate(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B8CDD4"))
        canvas.setLineWidth(0.4)
        canvas.line(19 * mm, A4[1] - 13 * mm, A4[0] - 19 * mm, A4[1] - 13 * mm)
        canvas.setFont("MSYH", 7.2)
        canvas.setFillColor(colors.HexColor("#587080"))
        canvas.drawString(19 * mm, A4[1] - 10 * mm, self.short_title[:55])
        canvas.drawRightString(A4[0] - 19 * mm, 9.5 * mm, str(doc.page))
        canvas.restoreState()


def bullets(items: list[str], styles: dict[str, ParagraphStyle]):
    return [Paragraph("- " + esc(item), styles["bullet"]) for item in items]


def fit_image(path: Path, max_w: float = 168 * mm, max_h: float = 92 * mm) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_w / width, max_h / height)
    return Image(str(path), width=width * scale, height=height * scale)


def figure_block(path: Path, caption: str, styles: dict[str, ParagraphStyle], max_h: float = 92 * mm):
    return [Spacer(1, 2 * mm), fit_image(path, max_h=max_h), Paragraph(esc(caption), styles["caption"])]


def lifecycle_diagram(styles: dict[str, ParagraphStyle]):
    cells = ["运行信号", "改进假设", "候选Patch", "微测试/回放", "Canary", "Stable/回滚"]
    data = [[Paragraph(cell, styles["diagram"]) for cell in cells]]
    table = Table(data, colWidths=[27 * mm] * len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF7F5")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#79B8BC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8CDD4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def write_note(paper: dict[str, object], authors: str) -> None:
    note_dir = NOTE_ROOT / str(paper["id"])
    image_dir = note_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((ASSET_ROOT / str(paper["id"])).glob("*")):
        if source.is_file():
            shutil.copy2(source, image_dir / source.name)
    lines = [
        "---",
        f'date: "{TODAY}"',
        f'paper_id: "arXiv:{paper["id"]}"',
        f'title: "{paper["title"]}"',
        f'authors: "{authors}"',
        'domain: "Self-Evolving-Deep-Research"',
        f'quality_score: "{sum(paper["scores"]) / 5:.1f}/10"',
        'status: "analyzed"',
        "---", "", f'# {paper["short"]} 中文深度解读', "",
        f'- 原文：[arXiv](https://arxiv.org/abs/{paper["id"]}) | [PDF](https://arxiv.org/pdf/{paper["id"]})',
        f'- 阅读建议：{paper["read"]}', "", "## 一页速读", "", str(paper["thesis"]), "", str(paper["plain"]), "",
    ]
    lines.extend(f'- {item}' for item in paper["findings"])
    lines.extend(["", "## 方法概述", ""])
    lines.extend(f'- {item}' for item in paper["method"])
    lines.extend(["", "## 按论文顺序实验记录", ""])
    for exp in paper["experiments"]:
        lines.extend([f'### {exp["name"]}', "", f'- 做了什么：{exp["what"]}', f'- 为什么：{exp["why"]}', f'- 设置：{exp["setup"]}', f'- 关键结果：{exp["result"]}', f'- 结论：{exp["conclusion"]}', ""])
    lines.extend(["## 局限性", ""] + [f'- {item}' for item in paper["limits"]])
    lines.extend(["", "## 对项目的融入", ""] + [f'- {item}' for item in paper["integration"]])
    lines.extend(["", "### 不宜照搬", ""] + [f'- {item}' for item in paper["dont"]])
    lines.extend(["", "### 最小实现", ""] + [f'{index}. {item}' for index, item in enumerate(paper["mvp"], 1)])
    lines.extend(["", "## 原论文图片索引", ""])
    for source in sorted(image_dir.glob("*")):
        lines.append(f'- ![{source.stem}](./images/{source.name})')
    (note_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pdf(paper: dict[str, object], meta: dict[str, object], styles: dict[str, ParagraphStyle]) -> Path:
    authors = str(meta.get("metadata", {}).get("Author", "--") or "--")
    author_short = authors if authors.count(";") <= 7 else ";".join(authors.split(";")[:7]) + "; et al."
    output = PDF_ROOT / f'{paper["order"]:02d}_{paper["id"]}_{paper["slug"]}_中文深度解读.pdf'
    doc = PaperTemplate(str(output), str(paper["short"]), title=str(paper["title"]), author=authors, subject="自进化 Deep Research 论文中文深度解读")
    story = [
        Spacer(1, 9 * mm), Paragraph(esc(paper["short"]), styles["title"]), Paragraph(esc(paper["title"]), styles["subtitle"]),
        Paragraph(f'arXiv:{paper["id"]} | {esc(paper["direction"])} | {meta.get("pages", "--")} 页原文', styles["subtitle"]),
        Paragraph("作者：" + esc(author_short), styles["small"]), Spacer(1, 4 * mm),
        Paragraph("核心判断", styles["h1"]), Paragraph(esc(paper["thesis"]), styles["callout"]),
        Paragraph("先用人话理解", styles["h2"]), Paragraph(esc(paper["plain"]), styles["body"]),
    ]
    quick = Table([[Paragraph("项目适配度", styles["label"]), Paragraph(f'{paper["fit"]:.1f}/10', styles["body"]), Paragraph("阅读建议", styles["label"]), Paragraph(esc(paper["read"]), styles["body"])]], colWidths=[25 * mm, 23 * mm, 22 * mm, 94 * mm])
    quick.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7F8")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8CDD4")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([quick, Paragraph("你应该记住的结果", styles["h1"]), *bullets(paper["findings"], styles)])

    assets = sorted((ASSET_ROOT / str(paper["id"])).glob("*"))
    for index, path in enumerate(assets[:2], 1):
        story.extend(figure_block(path, f'原论文关键图 {index}：{path.stem}', styles))

    story.extend([Paragraph("1. 方法拆解", styles["h1"]), *bullets(paper["method"], styles), Paragraph("统一映射到你的项目", styles["h2"]), lifecycle_diagram(styles)])
    story.append(Paragraph("2. 按论文顺序实验记录", styles["h1"]))
    for exp in paper["experiments"]:
        rows = [
            [Paragraph("做了什么", styles["label"]), Paragraph(esc(exp["what"]), styles["body"])],
            [Paragraph("为什么", styles["label"]), Paragraph(esc(exp["why"]), styles["body"])],
            [Paragraph("设置", styles["label"]), Paragraph(esc(exp["setup"]), styles["body"])],
            [Paragraph("关键结果", styles["label"]), Paragraph(esc(exp["result"]), styles["body"])],
            [Paragraph("支持结论", styles["label"]), Paragraph(esc(exp["conclusion"]), styles["body"])],
        ]
        table = Table(rows, colWidths=[25 * mm, 139 * mm])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF7F5")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8CDD4")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.extend([Paragraph(esc(exp["name"]), styles["h2"]), table, Spacer(1, 1.5 * mm)])

    story.extend([Paragraph("3. 可信度与局限", styles["h1"]), *bullets(paper["limits"], styles), Paragraph("4. 怎样融入你的 Deep Research 项目", styles["h1"]), *bullets(paper["integration"], styles), Paragraph("不宜照搬", styles["h2"]), *bullets(paper["dont"], styles), Paragraph("最小落地闭环", styles["h2"])])
    story.extend(Paragraph(f'{index}. {esc(item)}', styles["body"]) for index, item in enumerate(paper["mvp"], 1))

    score_names = ["创新性", "技术质量", "实验充分性", "对项目实用性", "综合推荐"]
    score_rows = [[Paragraph("维度", styles["label"]), Paragraph("分数", styles["label"])]] + [[Paragraph(name, styles["body"]), Paragraph(f"{score:.1f}/10", styles["body"])] for name, score in zip(score_names, paper["scores"])]
    score_table = Table(score_rows, colWidths=[72 * mm, 35 * mm])
    score_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#142D4C")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8CDD4")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8F9")]), ("ALIGN", (1, 1), (1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([Paragraph("5. 综合评价", styles["h1"]), score_table, Paragraph("建议带着这些问题精读", styles["h2"]), *bullets(["论文依赖什么样的成功信号或验证器？你的项目是否真的具备？", "数字来自完整对比实验、案例分析，还是作者提出的评测计划？", "方法改善的是检索、程序执行、知识补充，还是最终研究洞察？", "如果候选 Skill 失败，系统能否解释、隔离并回滚？"], styles)])

    story.extend([PageBreak(), Paragraph("原论文图表附录", styles["h1"]), Paragraph("以下图表从 arXiv 源码包的 figures/figs/img 目录提取。保留全部科学图和案例截图，排除 Logo 等装饰资源。", styles["body"])])
    if not assets:
        story.extend([Paragraph("该论文源码包没有独立论文图片。下面使用结构化文本表示其核心架构。", styles["callout"]), lifecycle_diagram(styles)])
    for index, path in enumerate(assets, 1):
        story.extend(figure_block(path, f'图表 {index}/{len(assets)}：{path.stem}', styles, max_h=105 * mm))
    story.extend([Paragraph("来源与版本", styles["h1"]), Paragraph(f'论文：{esc(paper["title"])}', styles["body"]), Paragraph(f'arXiv：https://arxiv.org/abs/{paper["id"]}', styles["body"]), Paragraph(f'原始 PDF：https://arxiv.org/pdf/{paper["id"]}', styles["body"]), Paragraph(f'解读日期：{TODAY}。所有实验数字来自该 arXiv 版本；预印本更新后数字可能变化。', styles["small"])])
    doc.build(story)
    write_note(paper, authors)
    return output


def main() -> None:
    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    NOTE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {item["paper_id"]: item for item in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    styles = make_styles()
    outputs = [build_pdf(paper, manifest[paper["id"]], styles) for paper in PAPERS]
    print("\n".join(str(output) for output in outputs))


if __name__ == "__main__":
    main()
