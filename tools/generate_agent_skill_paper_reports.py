from __future__ import annotations

import html
import json
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
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tmp" / "pdfs" / "agent-skills-papers" / "text" / "manifest.json"
NOTE_ROOT = ROOT / "docs" / "paper-reading" / "agent-skills" / "individual"
PDF_ROOT = ROOT / "output" / "pdf" / "agent-skills-paper-analysis"
TODAY = "2026-08-26"


PAPERS = [
    {
        "id": "2605.07358",
        "order": 1,
        "slug": "agent-skills-survey",
        "title": "A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications",
        "short": "Agent Skills 综合综述",
        "direction": "领域综述与生命周期分类",
        "thesis": "Agent Skill 是连接高层推理与可靠执行的外部程序性知识，其工程问题应按表示、获取、检索、演化四个生命周期阶段处理。",
        "findings": [
            "将 Skill 形式化为 S=(M,R,C)：主指令文档、辅助资源与适用条件。",
            "区分工具与技能：工具暴露原子操作，技能编码何时调用、如何组合、怎样恢复和如何验收。",
            "把文献统一到表示、获取、检索、演化四阶段，并进一步拆出选择、验证、仓库演化和运行时治理。",
            "指出真正的开放问题不是继续堆积技能，而是质量控制、跨平台互操作、安全更新、退役和长期因果评测。",
        ],
        "method": [
            "概念层：从被动参数知识、运行时主动知识和程序性缺口推导 Skill 的位置。",
            "表示层：文本型、代码型、混合资源型 Skill。",
            "获取层：人工经验、执行轨迹、任务求解和外部语料。",
            "检索层：稠密、稀疏、生成式、层级与结构感知检索，再叠加上下文、成本和反馈选择。",
            "演化层：修订、验证、策略耦合、仓库演化与运行时治理。",
        ],
        "experiments": [
            {
                "name": "证据记录 1：概念与形式化归纳",
                "what": "综合工具增强、记忆、程序性经验和 Agent Skill 文献，定义技能边界与形式化表示。",
                "why": "避免把工具、Prompt、原始轨迹、记忆条目和可复用 Skill 混为一谈。",
                "setup": "以可复用性、有限作用域、触发条件、辅助资源和可治理性作为纳入标准。",
                "result": "得到 S=(M,R,C) 的统一表示，并强调 Skill 必须表达触发、步骤、约束、失败模式和完成标准。",
                "conclusion": "项目中的 skill.json、SKILL.md、脚本资源和 Profile 约束应被视为同一技能契约。",
            },
            {
                "name": "证据记录 2：四阶段生命周期分类",
                "what": "按表示、获取、检索和演化重新组织分散研究路线。",
                "why": "把单篇方法放回全生命周期，识别系统缺口而非只比较模型分数。",
                "setup": "逐类比较代表性系统的技能来源、检索方式、更新触发器、验证条件与复用范围。",
                "result": "演化部分进一步区分修订、验证、策略耦合、仓库演化与治理；只有改变持久技能并在后续复用，才构成真正演化。",
                "conclusion": "你的项目当前运行面较完整，主要缺口正位于获取后的验证、版本发布和治理。",
            },
            {
                "name": "证据记录 3：开放挑战综合",
                "what": "总结规模化技能库的失效机制。",
                "why": "为后续研究路线和工程优先级提供问题清单。",
                "setup": "从抽象质量、触发、资源漂移、检索规模、组合约束、多目标选择、退役和治理等维度归纳。",
                "result": "高频风险包括过度抽象、触发弱、只增不删、成本与正确性目标冲突，以及最终成功无法证明技能有因果贡献。",
                "conclusion": "应先建立版本化经验账本和配对评测，再考虑在线自演化。",
            },
        ],
        "limits": [
            "综述覆盖快速变化的 2026 生态，分类和平台规模会继续变化。",
            "大量代表性结果来自预印本或不同运行环境，横向数字不可直接等价比较。",
            "它提供研究地图，不直接给出一套可部署的统一 Skill Schema。",
        ],
        "fit": 9.6,
        "integration": [
            "把现有 layered skills 明确成运行面，把新增生命周期模块作为控制面。",
            "以 S=(M,R,C) 检查现有 SkillDefinition：M 对应指令正文，R 对应 handler/脚本/参考资料，C 对应角色、Profile、触发和兼容条件。",
            "为每个技能建立 acquisition_source、selection_trace、validation、release、retirement 五类记录。",
            "评测不能只看最终报告，要关联技能是否被发现、读取、执行、遵循和产生证据增量。",
        ],
        "dont": ["不要把综述中的平台规模当作稳定事实写入运行时逻辑。", "不要仅因为一个方法被归入 Skill Evolution 就开放生产自修改。"],
        "mvp": ["新增统一 SkillContract。", "补充 SkillVersion 与 SkillRelease。", "将 Trace、Evidence、Evaluation 串成生命周期审计链。"],
        "scores": [9.5, 8.5, 7.5, 9.5, 9.8],
        "key_figures": ["003_figures_skill-evole.png", "006_figures_retrieval_workflow.png"],
    },
    {
        "id": "2602.12670",
        "order": 2,
        "slug": "skillsbench",
        "title": "SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks",
        "short": "SkillsBench",
        "direction": "通用 Skill 边际效用评测",
        "thesis": "高质量人工技能通常有效，但收益强烈依赖模型、Harness、任务和技能设计；自动生成技能甚至可能稳定低于无技能基线。",
        "findings": [
            "87 个任务、18 个模型-Harness 配置的配对评测中，策划 Skill 平均把通过率从 33.9% 提升到 50.5%，绝对增益 16.6 个百分点。",
            "13/87 个任务出现负增益；技能被发现或调用不等于任务成功。",
            "1 个或 2-3 个聚焦技能的收益高于 4 个以上技能；详细而全面的长文档收益接近于零。",
            "三种自生成配置分别低于无技能基线 8.1、11.3、11.5 个百分点。",
        ],
        "method": [
            "为每个任务提供容器、输入、验证器和策划 Skill。",
            "同一配置运行 no-Skills 与 curated-Skills，采用三次试验的任务宏平均。",
            "使用绝对通过率差和归一化增益，另审计调用率、Token、成本、时延和失败轨迹。",
            "单独评估 self-generated Skills，不与主配对结果混在一起。",
        ],
        "experiments": [
            {"name": "实验 1：跨模型与 Harness 主实验", "what": "比较 18 个配置在有无策划技能下的任务通过率。", "why": "检验 Skill 是否具有跨系统的稳定边际收益。", "setup": "87 个任务，OpenHands、Gemini CLI、Claude Code、Codex CLI；每个条件固定三次试验。", "result": "均值 33.9% -> 50.5%，+16.6pp；单配置增益从 +4.1pp 到 +25.7pp。最高有技能通过率为 OpenHands+GPT-5.5 的 67.3%。", "conclusion": "Skill 效果属于 model-harness-task 联合属性，不能只给技能打一个全局分数。"},
            {"name": "实验 2：领域和任务异质性", "what": "按领域及单任务拆分 Skill 增益。", "why": "定位什么任务最需要外部程序性知识。", "setup": "8 个领域并审计正负增益极端任务。", "result": "自然科学 +28.8pp、媒体内容 +24.1pp、安全 +18.9pp；软件工程 +11.6pp、数学与运筹 +9.7pp。13 个任务为负增益。", "conclusion": "专业、格式脆弱、流程明确的任务最受益；已有强先验或流程过重时可能受害。"},
            {"name": "实验 3：技能数量与长度消融", "what": "比较不同数量和文档长度的技能。", "why": "验证信息越多是否越好。", "setup": "按任务绑定技能数与 SKILL.md 文档长度分桶。", "result": "1 个技能 +18.0pp，2-3 个 +19.0pp，4 个以上仅 +10.1pp；紧凑/标准长度 +19.0/+21.5pp，全面文档仅 +0.7pp。", "conclusion": "应优化不可推断的步骤、约束和验收细节，而不是追求篇幅。"},
            {"name": "实验 4：自生成技能", "what": "Agent 先用 skill-creator 生成技能包，再独立求解。", "why": "测试自动生成能否替代人工策划。", "setup": "Claude Code+Opus 4.7、Codex+GPT-5.5、Gemini CLI+Gemini 3.1 Pro。", "result": "相对无技能分别 -8.1、-11.3、-11.5pp，而策划技能在同配置上 +18.2 至 +24.8pp。审计发现生成技能未被读取、作者阶段挤占求解工作以及内容自信但错误。", "conclusion": "自动生成技能必须有独立验证和调用可观测性。"},
            {"name": "实验 5：调用与成本审计", "what": "分析技能调用率、总 Token、成本和运行时间。", "why": "区分被发现、被使用和实际产生收益。", "setup": "基于 BenchFlow 完整轨迹与固定 87x3 试验框架。", "result": "高调用率并不保证高通过率；不同 Harness 对同一模型的技能收益明显不同。", "conclusion": "Router 指标必须与执行结果和证据质量联结，不能把调用率当效用。"},
        ],
        "limits": ["以终端、容器化任务为主，不能直接代表 GUI、多智能体或超长研究任务。", "技能注入增加上下文，尚未完全排除长度本身的影响。", "系统与模型版本变化可能改变结果。"],
        "fit": 10.0,
        "integration": ["在 evaluation/ 增加同任务、同模型、同 Profile 的 skill/no-skill 配对运行。", "为每个 skill_id+version 按 task_family 建立 Utility Ledger。", "把 citation grounding、unsupported claim、requirement coverage、Token、搜索次数和 wall time 纳入联合指标。", "发布门禁要求无安全和引用回归，并在多个不同任务上有一致收益。"],
        "dont": ["不要用平均收益掩盖任务级负增益。", "不要把 Skill Invocation Rate 当成功代理。", "不要允许自生成技能直接覆盖生产技能。"],
        "mvp": ["固定 10-20 个代表性研究任务。", "运行 no-skill/skill 配对实验。", "生成 positive/neutral/negative/unsafe 四类判定。"],
        "scores": [9.4, 9.3, 9.4, 10.0, 10.0],
        "key_figures": ["001_figs_method.png", "011_figs_selfgen_comparison.png", "006_figs_heatmap_skills_uplift.png"],
    },
    {
        "id": "2604.20087",
        "order": 3,
        "slug": "skilllearnbench",
        "title": "SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks",
        "short": "SkillLearnBench",
        "direction": "自动技能生成与持续学习评测",
        "thesis": "生成式技能学习可以超过无技能基线，但仍远逊人工技能；没有新外部信号的反复自反馈会漂移，技能内容和实际采用同样重要。",
        "findings": ["20 个任务、15 个子领域、6 个生成模型和固定求解 Agent。", "四种生成方法平均准确率 27.33%-31.08%，人工技能 74.50%，无技能 10.17%。", "Skill Creator 使用率最高 84.47%，但准确率仅 27.33%，证明高采用不等于高质量。", "多轮外部 Teacher Feedback 持续改善；Self Feedback 后续轮次出现对齐和准确率下降。"],
        "method": ["三级评测：Skill 文档质量、执行轨迹、最终任务结果。", "比较 One-Shot、Self Feedback、Teacher Feedback、Skill Creator。", "技能生成模型变化，求解模型固定为 Claude Sonnet 4.6，Judge 固定为 GPT-5-mini。", "种子实例生成技能，再在保持任务逻辑但改变参数的实例上测试复用。"],
        "experiments": [
            {"name": "实验 1：四类持续学习方法主比较", "what": "跨 6 个生成 LLM 比较四种技能生成方法。", "why": "分离生成方式、生成模型和固定求解器的影响。", "setup": "20 个真实任务；求解器 Claude Sonnet 4.6、温度 0、最多 100 轮。", "result": "平均准确率：One-Shot 30.44%、Self Feedback 31.08%、Teacher Feedback 27.47%、Skill Creator 27.33%；人工 74.50%，无技能 10.17%。", "conclusion": "当前自动方法有增益但离人工质量很远，方法排名还依赖生成 LLM。"},
            {"name": "实验 2：类别与复用分析", "what": "按任务类别和种子/保留实例分析。", "why": "判断失败是否来自过拟合。", "setup": "软件工程、信息检索、生产工具、数据分析、内容和 Utilities。", "result": "流程清晰的软件工程与生产工具受益较大；保留实例准确率与种子实例相近或略高。", "conclusion": "主要瓶颈不是种子过拟合，而是未捕获稳定的核心任务逻辑。"},
            {"name": "实验 3：Agent 行为改变", "what": "比较 Coverage、Alignment、Usage、步骤和工具调用。", "why": "解释文档质量如何转化为执行结果。", "setup": "所有生成方法使用同一求解器和三级指标。", "result": "Self Feedback 轨迹更短且准确率略高；Teacher Feedback 覆盖高但采用低、执行重；Skill Creator 采用最高但准确率低。", "conclusion": "技能是否包含正确知识与 Agent 是否采用它是两个独立门槛。"},
            {"name": "实验 4：多轮反馈演化", "what": "在 Productivity Tools 上把 Self/Teacher Feedback 延长到四轮。", "why": "检验持续迭代是真改善还是自我漂移。", "setup": "Claude Sonnet 4.6 作为生成模型，跟踪覆盖、对齐、准确率和技能数。", "result": "Self Feedback 覆盖不变、对齐持续下降，准确率短暂上升后明显下跌；Teacher Feedback 首轮提升覆盖，后续对齐稳定、准确率持续增长。", "conclusion": "只有带新信息的外部反馈才支持可靠演化。"},
            {"name": "实验 5：实例数、模型规模与结构", "what": "增加生成实例数、比较模型规模并分析技能包结构。", "why": "寻找提高自动生成质量的直接杠杆。", "setup": "One-Shot 使用 1/3/5 个实例；Claude/Gemini 三档模型；技能按纯指令、脚本、MCP/子 Agent 分类。", "result": "增加实例反而使准确率从 56.67% 降到 23.33%/27.78%；生成技能几乎全是纯指令，人工技能中 33.3% 带脚本、2.4% 带 MCP/子 Agent；更大模型不单调更好。", "conclusion": "简单增加上下文或模型规模不会自动得到可执行、可迁移的技能。"},
            {"name": "实验 6：成本与鲁棒性", "what": "分解输入/输出 Token 并进行三次重复运行。", "why": "确认单次评测排序以及成本来源。", "setup": "每任务代表实例三次运行，统计均值和标准差。", "result": "生成方法总 Token 390K-528K，低于无技能 727K；输入 Token 占总成本 97% 以上；方法相对排序基本保持，但准确率标准差约 13%-17%。", "conclusion": "技能可能缩短轨迹，但必须多次评测最终正确性。"},
        ],
        "limits": ["固定一个求解模型，跨执行 Agent 的外推仍有限。", "许多 Level 1/2 指标使用 LLM Judge。", "任务数量 20，部分消融仅在单类别和单模型上运行。"],
        "fit": 9.9,
        "integration": ["Candidate Compiler 只提出新版本，不允许自评后直接发布。", "将 Evidence Gate、Citation Gate 和独立 Evaluation Runner 作为外部反馈。", "分别记录 coverage、selection、usage、alignment、outcome，避免只看最终分。", "对多轮自修订设置最大轮次和漂移检测，失败后回滚而非继续自改。"],
        "dont": ["不要把更强生成模型等同于更好技能。", "不要用一个成功轨迹不断自我改写。", "不要把高调用率当内容正确。"],
        "mvp": ["候选生成器输出 candidate-v1。", "独立评测器给出缺陷与外部证据。", "生成 candidate-v2 后重新配对评测，禁止原地覆盖 active。"],
        "scores": [9.1, 9.0, 9.1, 10.0, 9.9],
        "key_figures": ["001_figures_baselines.png", "006_figures_skill_evolution.png"],
    },
    {
        "id": "2602.08234",
        "order": 4,
        "slug": "skillrl",
        "title": "SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning",
        "short": "SkillRL",
        "direction": "层级技能库与强化学习共同演化",
        "thesis": "将冗长轨迹蒸馏成层级技能，并让技能库随 RL 失败信号持续扩展，可同时提升策略学习、跨任务泛化和上下文效率。",
        "findings": ["ALFWorld 成功率 89.9%，WebShop 72.7%，显著超过 GRPO 的 77.6% 和 66.1%。", "搜索增强 QA 平均 47.1%，训练仅使用 NQ 与 HotpotQA，仍在多个 OOD 数据集保持收益。", "去掉层级结构分别下降 13.1 和 11.3 个百分点；换成原始轨迹最大下降约 25 个百分点。", "技能库从 55 增至 100；动态演化让 ALFWorld 在约 60 步越过 80%，静态版本约需 90 步且峰值更低。"],
        "method": ["从成功与失败轨迹蒸馏一般技能和任务特定技能。", "检索 K=6 个相关技能辅助 Rollout。", "先用冷启动 SFT 教会策略使用技能，再用 GRPO 优化。", "在验证检查点根据失败模式生成、细化 SkillBank。"],
        "experiments": [
            {"name": "实验 1：ALFWorld 与 WebShop 主结果", "what": "比较 Prompt、记忆、RL 和记忆增强 RL 基线。", "why": "验证层级 SkillBank 相比原始经验和普通 RL 的增益。", "setup": "Qwen2.5-7B-Instruct；GRPO 学习率 1e-6、batch 16、group 8。", "result": "SkillRL 为 89.9%/72.7%；GRPO 为 77.6%/66.1%；Raw/Memory 增强基线明显更低。", "conclusion": "抽象技能与策略共同训练能补偿小模型规模。"},
            {"name": "实验 2：搜索增强 QA 泛化", "what": "在 7 个单跳/多跳 QA 数据集评估。", "why": "测试技能是否迁移到未训练的数据分布。", "setup": "仅 NQ、HotpotQA 训练；TriviaQA、PopQA、2Wiki、MuSiQue、Bamboogle 为 OOD。", "result": "平均 47.1%，高于 EvolveR 43.1% 和 Search-R1 38.5%；Bamboogle 达 73.8%。", "conclusion": "部分检索与多跳整合策略具有跨数据集迁移性。"},
            {"name": "实验 3：组件消融", "what": "移除层级、技能库、冷启动 SFT 或动态演化。", "why": "识别性能来源。", "setup": "ALFWorld/WebShop 平均成功率。", "result": "无层级 76.8/61.4；原始轨迹 61.7/50.2；无冷启动 65.2/46.5；无动态演化 84.4/70.3。", "conclusion": "抽象、层级和冷启动是主要贡献，动态演化提供较小但稳定增益。"},
            {"name": "实验 4：SkillBank 增长", "what": "跟踪 150 个训练步中的技能数。", "why": "观察递归演化是否覆盖新失败。", "setup": "初始 12 个通用、43 个任务技能；验证点更新。", "result": "最终 20 个通用、80 个任务技能，总计 100。", "conclusion": "增长主要来自任务级失败模式，但论文尚未充分研究去重与退役。"},
            {"name": "实验 5：上下文与收敛效率", "what": "比较原始记忆与蒸馏技能的 Prompt 长度及训练曲线。", "why": "检验技能是否减少上下文膨胀并加速学习。", "setup": "Raw Memory 对比 SkillRL，ALFWorld 验证曲线。", "result": "原始记忆平均约 1450 tokens，SkillRL 小于 1300，约减 10.3%；演化版约 60 步超过 80% 成功率。", "conclusion": "技能抽象兼具压缩和学习引导作用。"},
        ],
        "limits": ["依赖高成本教师、SFT 和在线 RL，工程门槛高。", "SkillBank 从 55 增到 100，但去重、冲突、版本和安全治理不足。", "交互环境奖励较明确，与开放式 Deep Research 的稀疏、多目标质量不同。"],
        "fit": 8.3,
        "integration": ["短期只借鉴层级 SkillBank 和失败驱动候选生成，不直接上线 RL。", "把 ALFWorld 奖励替换为 citation grounding、coverage、evidence gain、成本和安全联合指标。", "先记录候选集合和选择概率，建立离线可学习的路由日志。", "动态演化只生成 candidate；active 版本仍通过发布门禁。"],
        "dont": ["不要把线上用户研究任务当 RL 探索环境。", "不要让一次失败立即改写生产技能。", "不要允许 SkillBank 只增不删。"],
        "mvp": ["实现一般技能/任务技能两级关系。", "Failure Memory 触发候选建议。", "用离线 replay 验证候选，再考虑 contextual bandit。"],
        "scores": [9.0, 8.7, 8.4, 8.2, 8.3],
        "key_figures": ["002_asset_method.png", "004_asset_success_rate_curve.png"],
    },
    {
        "id": "2604.04804",
        "order": 5,
        "slug": "skillx",
        "title": "SkillX: Automatically Constructing Skill Knowledge Bases for Agents",
        "short": "SkillX",
        "direction": "三级技能知识库自动构建",
        "thesis": "将经验拆成规划、功能、原子三级技能，并结合合并、过滤、迭代细化和经验引导探索，可提高复杂工具 Agent 的性能与迁移。",
        "findings": ["三级结构分别表达任务组织、子任务工具组合和单工具调用模式。", "SkillX 在 BFCL-v3、AppWorld、tau2-bench 上普遍优于 No Memory、A-Mem、AWM、ExpeL。", "不同模型需要不同技能组合：弱模型可能因过度模仿功能/原子技能而退化。", "迭代优化存在过拟合，最佳轮次依赖模型和任务。"],
        "method": ["成功轨迹提取 Planning Skill，过滤探索和回溯。", "依据规划步骤抽取 Functional Skill，并从工具反馈提取 Atomic Skill。", "使用一般过滤器和工具 Schema 过滤器阻止不可复用或幻觉调用。", "经验引导探索优先覆盖未使用或高失败工具，再合成任务扩充技能库。", "推理时先用规划技能重写伪计划，再逐步骤检索功能与原子技能。"],
        "experiments": [
            {"name": "实验 1：三基准主结果", "what": "跨 Qwen3-32B、Kimi-K2、GLM-4.6 比较多种经验表示。", "why": "验证三级技能知识库是否优于轨迹和工作流记忆。", "setup": "BFCL-v3、AppWorld、tau2-bench；Avg@4、Pass@4 或四次运行 Pass 指标。", "result": "Qwen3 在 BFCL Avg@4 从 53.67 提到 63.67，tau2 Retail 从 53.75 提到 66.87；Kimi AppWorld Avg@4 从 46.88 到 56.40；GLM AppWorld Pass@4 从 83.33 到 88.69。", "conclusion": "技能表示形式比单纯使用更强提取模型更关键。"},
            {"name": "实验 2：跨模型经验迁移", "what": "让 GLM-4.6 提取技能，再由较弱模型执行。", "why": "评估强模型知识是否可作为共享程序性资产。", "setup": "AWM、ExpeL 与 SkillX 采用相同强提取器。", "result": "即使提取模型相同，SkillX 仍领先，说明三级表示和检索流程贡献明显。", "conclusion": "可以用强模型集中构建技能，再分发给不同 Worker。"},
            {"name": "实验 3：多级技能组合分析", "what": "分别启用规划、功能、原子以及组合。", "why": "判断每层对不同能力模型的作用。", "setup": "AppWorld 上跨三个模型比较性能、步骤和 Token。", "result": "规划技能稳定减少步骤；功能技能通常贡献最大；原子技能补足 API 约束。Qwen 只用规划技能最好，Kimi 用功能+原子最好，GLM 用全部最好。", "conclusion": "Router 需要模型条件化，不能对所有 Worker 注入相同层级。"},
            {"name": "实验 4：迭代细化与扩展", "what": "比较 1-3 轮 refinement 及随机/经验引导扩展。", "why": "测试如何在训练任务有限时扩大覆盖。", "setup": "BFCL-v3 与 AppWorld；GLM-4.6。", "result": "AppWorld 的 Expand-Iter3 Pass@4 达 88.69，高于 No Memory 83.33；BFCL 并非轮数越多越好，第三轮下降。", "conclusion": "扩展有效，但文本自优化会过拟合，必须用验证集早停。"},
            {"name": "实验 5：更强模型与案例", "what": "在 DeepSeek-V3.2、GPT-4.1 上测试，并分析失败避免。", "why": "验证方法是否只适用于原三模型。", "setup": "自提取和 GLM 提取两种模式。", "result": "多数配置获得增益，但 GPT-4.1 的 GLM 提取在 BFCL 提升更明显，自提取并不总是更好。", "conclusion": "共享技能库可跨模型，但提取器与执行器匹配仍需评测。"},
        ],
        "limits": ["跨工具生态迁移有限，技能与具体 Schema 绑定。", "主要评测工具调用环境，开放式对话和研究写作覆盖不足。", "伪计划不注入最终 Prompt，实际收益归因仍有多个耦合因素。"],
        "fit": 9.2,
        "integration": ["把现有 orchestrator/strategy/access 层扩成 Planning/Functional/Atomic 语义关系。", "为技能增加 parent、requires、composes、conflicts、supersedes。", "利用 AgentLoop 轨迹抽取功能技能，利用工具 Schema 与错误反馈抽取原子约束。", "按 Worker 模型维护最佳层级组合，不默认全量注入。"],
        "dont": ["不要把三级技能全部塞入同一上下文。", "不要用语义相似度合并不同权限或不同输出契约的技能。", "不要无限迭代文本细化。"],
        "mvp": ["先构建静态 Skill Graph。", "实现无环验证和按层投影。", "在 academic-research 一条链上做组合消融。"],
        "scores": [8.8, 8.8, 8.5, 9.3, 9.2],
        "key_figures": ["008_image_overview_model.png", "002_image_analysis.png"],
    },
    {
        "id": "2606.09316",
        "order": 6,
        "slug": "anything2skill",
        "title": "Anything2Skill: Compiling External Knowledge into Reusable Skills for Agents",
        "short": "Anything2Skill",
        "direction": "外部知识到证据化技能契约",
        "thesis": "文档、手册、对话、日志和轨迹中的程序性知识，可以经证据窗口、分类先验、受约束编译和版本化协调转化为可复用 SkillBank。",
        "findings": ["技能契约同时记录触发、禁用条件、流程、约束、输出、证据和置信度。", "SkillBank 支持 CREATE、STRENGTHEN、REVISE、MERGE、SPLIT、UNCHANGED、DISCARD。", "qsv 上 Base 81.60%、Skill 91.95%、RAG 95.41%、Skill+RAG 98.85%。", "GitHub CLI 上 Base 64.70%、Skill 82.30%、RAG 76.50%、Skill+RAG 94.10%。"],
        "method": ["把输入切为带标题、跨度、锚点和来源的 Evidence Window。", "先规划候选，再独立展开；证据不足返回 null，避免自由补全。", "以 taxonomy 限制资产类型和层级，避免安全规则与执行微技能错误合并。", "文档内编译后，再与持久 Registry 进行混合检索和生命周期协调。", "Registry 是事实源，技能树只是可重建投影。"],
        "experiments": [
            {"name": "实验 1：技能单独增强 CLI Agent", "what": "只检索已编译技能，不提供任务时原文段落。", "why": "验证 SkillBank 是否只是另一种文档索引。", "setup": "qsv 和 GitHub CLI；统一 gpt5.4-mini；成功率指标。", "result": "qsv 81.60% -> 91.95%，+10.35pp；GitHub CLI 64.70% -> 82.30%，+17.60pp。", "conclusion": "结构化程序性知识可独立于任务时 RAG 提供收益。"},
            {"name": "实验 2：RAG 单独增强", "what": "每任务检索 top-3 原始知识条目。", "why": "建立声明性知识基线。", "setup": "与技能条件相同模型和协议。", "result": "qsv 95.41%，GitHub CLI 76.50%。qsv 更依赖精确选项文档，复杂 GitHub 工作流更依赖程序。", "conclusion": "RAG 与 Skill 的相对价值取决于事实查询还是过程组合。"},
            {"name": "实验 3：Skill 与 RAG 组合", "what": "同时提供原始证据片段和程序性技能。", "why": "检验二者互补性。", "setup": "qsv 与由 110 份文档编译出 179 个技能的 GitHub CLI。", "result": "qsv 98.85%，相对 Base +17.25pp；GitHub CLI 94.10%，相对 Base +29.40pp，均为最佳。", "conclusion": "Deep Research 应同时保留 Evidence Retrieval 与 Procedural Skill Retrieval。"},
        ],
        "limits": ["仅两个 CLI 基准，任务和工具接口较稳定。", "没有系统报告编译错误、冲突、污染和安全攻击下的表现。", "管理动作依赖模型判断，尚需确定性 Lint、配对评测和人工审批。"],
        "fit": 10.0,
        "integration": ["用 EvidenceRecord/ClaimEvidenceLink 作为编译器唯一允许引用的事实边界。", "从成功轨迹提取流程不变量，从失败轨迹提取 failure_signals，但不复制具体答案。", "候选技能写入 candidates/ 或独立数据库，不能直接进入 skills/。", "增加 triggers、contraindications、output_contract、source_refs、confidence 和 parent_version。", "保留现有 RAG；Router 同时检索声明性证据与程序性 Skill。"],
        "dont": ["不要从未经 Citation Gate 的报告编译技能。", "不要在技能中固化论文标题、人名、日期和测试答案。", "不要把 Registry 投影视为不可恢复的唯一数据。"],
        "mvp": ["选择 academic-paper-verification 任务族。", "从 5-10 个已验证运行生成 candidate contract。", "Lint、配对评测、人工批准后发布 0.1.0。"],
        "scores": [9.4, 8.6, 7.4, 10.0, 10.0],
        "key_figures": ["001_frame.png"],
    },
    {
        "id": "2605.25430",
        "order": 7,
        "slug": "codeskill",
        "title": "CODESKILL: Learning Self-Evolving Skills for Coding Agents",
        "short": "CODESKILL",
        "direction": "可学习的技能抽取、演化与库维护",
        "thesis": "技能管理本身可被训练为策略：从轨迹抽取任务级和事件级技能，再用下游可验证结果与质量 Judge 联合优化添加、演化、合并和丢弃。",
        "findings": ["Qwen3.5-35B 冻结执行器下，平均通过率从 29.57% 提到 39.26%，推理步骤从 44.12 降至 35.15。", "GPT-5.4-mini 执行器下从 21.80% 提到 30.73%。", "Terminal-Bench 2 未参与训练，仍从 25.88% 提到 34.12%。", "完整生命周期把技能库从 1252 压到 676，平均性能仅下降约 2 个百分点。"],
        "method": ["Skill Manager 由 Qwen3.5-4B 初始化，执行器保持冻结。", "抽取任务级工作流和事件级触发-响应技能。", "Skill evolution 修改匹配技能，maintenance 执行 add/merge/drop。", "奖励结合技能质量、执行结果和技能与轨迹的对齐因子。", "三阶段课程逐步训练抽取、演化和维护。"],
        "experiments": [
            {"name": "实验 1：多 SWE 基准主结果", "what": "比较无技能、Subtask Memory、Prompt 管理、SFT 和 RL 管理。", "why": "验证可学习管理策略是否优于固定 Prompt。", "setup": "EnvBench Python/Java、SWE-Bench Verified、Terminal-Bench 2；两种冻结执行器。", "result": "Qwen 执行器平均 29.57% -> 39.26%；GPT 执行器 21.80% -> 30.73%，均领先强 Prompt/Memory 基线。", "conclusion": "下游反馈可学到比固定技能管理模板更有效的操作策略。"},
            {"name": "实验 2：推理效率", "what": "统计成功实例的平均 reasoning steps。", "why": "确认技能是否只增加上下文还是缩短求解路径。", "setup": "跨四类基准宏平均。", "result": "Qwen 执行器 44.12 -> 35.15；多个技能方法普遍减少步骤。", "conclusion": "合适的程序性指导可减少探索，但应同时核算技能注入成本。"},
            {"name": "实验 3：任务与策略跨域泛化", "what": "在未训练 Terminal-Bench 2 和另一执行模型上测试。", "why": "识别管理策略是否过拟合训练任务或执行器。", "setup": "Terminal-Bench 2 完全 OOD；管理策略不变，执行器切换 GPT-5.4-mini。", "result": "Terminal-Bench 2 25.88% -> 34.12%；切换执行器后仍相对无技能 +8.93pp。", "conclusion": "学习到的部分管理行为可以跨任务和执行策略迁移。"},
            {"name": "实验 4：粒度与演化消融", "what": "比较事件级、任务级、抽取组合和演化。", "why": "判断技能粒度与更新动作的贡献。", "setup": "固定 Qwen3.5-35B-A3B。", "result": "Extraction-only 平均约 38.63%，加 Evolution 约 40.75%；两种粒度在不同任务上互补。", "conclusion": "研究任务也应区分全局研究策略与局部失败恢复技能。"},
            {"name": "实验 5：库维护与训练动态", "what": "跟踪 add/merge/drop、技能库大小与奖励趋势。", "why": "验证维护能否限制无限增长。", "setup": "测试时顺序处理候选，完整生命周期维护。", "result": "技能数 1252 -> 676，接近减半；环境奖励 20 步趋势从 0.004 上升到 0.158。", "conclusion": "有界技能库需要合并和丢弃，即使会牺牲少量短期峰值性能。"},
        ],
        "limits": ["目前只有自然语言技能，不包含脚本、工具定义等复杂资产。", "每次只对一个候选执行一种操作，无法联合拆分或多技能协调。", "依赖 LLM Judge 和大量训练 Rollout，开放式研究奖励设计更困难。"],
        "fit": 8.6,
        "integration": ["直接复用 AgentLoop Trace、Failure Memory 和 Evaluation 作为经验与反馈源。", "把代码通过率替换成证据充分性、引用正确性、交付覆盖、成本和时效。", "先用规则实现 add/merge/drop/retire，再收集数据训练管理器。", "维护决策必须保留候选、依据、旧版本和回滚点。"],
        "dont": ["不要用 Writer 的主观分数作为 RL 奖励。", "不要在在线用户任务中探索高风险维护动作。", "不要只优化平均成功率而忽略引用与安全回归。"],
        "mvp": ["建立 SkillExperience 与 SkillEvaluation。", "实现重复候选合并和低效技能退役规则。", "积累离线日志后再训练轻量管理策略。"],
        "scores": [9.0, 8.8, 8.5, 8.8, 8.6],
        "key_figures": ["002_latex_pics_pipeline_crop.png", "003_latex_pics_ablation_croped.png"],
    },
    {
        "id": "2603.00718",
        "order": 8,
        "slug": "skillcraft",
        "title": "SkillCraft: Can LLM Agents Learn to Use Tools Skillfully?",
        "short": "SkillCraft",
        "direction": "可执行工具组合、缓存与复用",
        "thesis": "把验证过的工具链缓存为可执行技能能大幅降本，但深层递归组合会放大底层错误；技能创建质量往往比执行模型能力更重要。",
        "findings": ["126 个重复子结构任务中，多数模型用 Skill Mode 提高成功率并减少 Token。", "GPT-5.2 Token 从 1.23M 降至 0.26M，约 -79%，成本约 -75%。", "Skill 执行率与任务成功相关系数 0.65。", "深层技能层级不稳定：GPT-5.2 从平面技能 90% 降到层级模式 79%。", "Claude 创建的技能跨四个执行模型均 100% 成功，并带来 54%-81% Token 节省。"],
        "method": ["Agent 在受控代码环境中组合原子工具。", "只有通过执行和 Coding Verifier 的组合才进入 Skill Library。", "通过 save/get/list/execute 四个轻量接口控制技能库。", "评估平面复用、层级递归、跨任务和跨模型静态复用。"],
        "experiments": [
            {"name": "实验 1：Base 与 Skill Mode 主比较", "what": "8 个模型在 126 个任务上比较无缓存与技能缓存。", "why": "同时测正确性和长工具链成本。", "setup": "统一任务、工具端点和验证规则；成功阈值为专家评分至少 90%。", "result": "GPT-5.2 成功 87% -> 90%，Token -79%；Claude 94% -> 96%，Token -71%；弱模型收益较不稳定。", "conclusion": "可执行缓存能显著减少重复规划，但效果取决于模型会不会正确创建和使用技能。"},
            {"name": "实验 2：指标相关性", "what": "分析成功、执行率、复用率和成本指标相关性。", "why": "识别技能行为与效果之间的关系。", "setup": "跨模型聚合相关系数。", "result": "Skill 执行率与任务成功 r=0.65；基线能力与节省收益如 Turns Saved 的相关为 0.53。", "conclusion": "执行正确性是组合能力核心，强模型更能兑现缓存收益。"},
            {"name": "实验 3：层级组合", "what": "允许技能递归调用技能，最大深度 10。", "why": "检验组合深度是否带来乘法效率。", "setup": "DeepSeek、Claude、GPT-5.2 比较 Base/Skill/Hierarchy。", "result": "GPT-5.2 平面 90%，层级 79%；Claude 成功同为 96%，但 Token 0.40M -> 0.63M；底层 null 等错误会级联。", "conclusion": "浅层、充分测试的组合目前优于深层自动递归。"},
            {"name": "实验 4：跨任务难度迁移", "what": "Easy->Hard、Hard->Easy、Hard->Hard 静态复用。", "why": "验证技能是否抽象了共享过程而非实例答案。", "setup": "Claude 和 Gemini，同任务族不同难度。", "result": "Gemini Easy->Hard 成功 76% -> 90%、Token -41%；Hard->Hard 76% -> 100%、Token -42%。Claude Hard->Hard 成功保持 95%，Token -76%。", "conclusion": "高质量技能可跨难度迁移并显著降本。"},
            {"name": "实验 5：跨模型迁移", "what": "4 个模型互相执行对方创建的技能。", "why": "比较创建者质量和执行者能力。", "setup": "8 个困难任务，16 个创建-执行组合，执行期不允许新建技能。", "result": "Claude 技能对所有执行器均 100% 成功，Token 节省 54%-81%；Minimax 技能出现 -48% 到 +18% 的节省，即部分增加成本。", "conclusion": "应集中由高质量创建器产出，再通过兼容评测分发。"},
        ],
        "limits": ["任务具有重复子结构，真实开放式任务的缓存命中率可能更低。", "代码技能运行在受控接口，复杂权限和外部副作用未充分覆盖。", "深层组合失败说明现有验证主要保证单技能，不保证组合闭包。"],
        "fit": 8.9,
        "integration": ["将高频 Search/Fetch/Extract/Verify 序列封装为浅层 Composite Skill。", "组合技能必须记录父子 Trace、预算、依赖和降级路径。", "只有原子组件和组合整体都通过验证，才允许进入 active。", "测量 Token、工具调用、成功、证据增量及跨模型复用。"],
        "dont": ["不要允许无限递归技能调用。", "不要只验证子技能而忽略组合端到端输出。", "不要把高质量模型生成的脚本默认信任。"],
        "mvp": ["实现 academic-paper-verification 组合技能。", "最大组合深度设为 2。", "组件失败时退化为原子调用，并记录差异。"],
        "scores": [9.2, 9.0, 8.8, 9.0, 8.9],
        "key_figures": ["001_Figures_main.png", "008_Figures_cross_model_4x4_token_saving.png"],
    },
    {
        "id": "2603.15401",
        "order": 9,
        "slug": "swe-skills-bench",
        "title": "SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering?",
        "short": "SWE-Skills-Bench",
        "direction": "真实软件工程中的技能收益与上下文干扰",
        "thesis": "在高基线能力的软件工程任务中，大多数技能不提高通过率，却可能显著增加成本；版本错配和具体模板还会通过锚定、幻觉和概念串扰降低结果。",
        "findings": ["49 个技能、565 个固定仓库实例，平均通过率仅从 89.8% 到 91.0%，+1.2pp。", "39/49 个技能零增益，24 个在两种条件下都 100%。", "平均 Token 增加 10.5%，单技能可从 -77.6% 到 +450.8%。", "3 个技能产生负增益；Linkerd 案例显示版本和协议模板导致连续错误。"],
        "method": ["从 84,192 个公开 Skill 中筛选 49 个可确定性验证的 SWE Skill。", "每个技能匹配固定 commit 的真实 GitHub 项目和结构化需求文档。", "把每条验收条件转为 pytest 确定性测试。", "SKILL.md 仅在实验条件下物理注入，任务文档不提及技能。"],
        "experiments": [
            {"name": "实验 1：49 个技能配对评测", "what": "对每个仓库任务比较有无 SKILL.md。", "why": "测量真实 SWE 中的边际收益。", "setup": "Claude Code + Claude Haiku 4.5；Ubuntu 24.04 容器；565 实例。", "result": "平均 89.8% -> 91.0%，+1.2pp；39 个技能零收益。", "conclusion": "当基础模型已能完成任务时，通用技能经常冗余。"},
            {"name": "实验 2：Token 成本分析", "what": "比较 C+、C-、相对开销与成本效率。", "why": "识别不改变成功率却改变推理路径的技能。", "setup": "对 49 个技能按任务平均 Token。", "result": "总体 +10.5%；service-mesh-observability +450.8%，python-resilience -77.6%，但二者通过率均不变。", "conclusion": "正确性收益和推理成本基本解耦，必须分别评测。"},
            {"name": "实验 3：正收益技能", "what": "分析 7 个正增益技能。", "why": "总结真正有效的技能形态。", "setup": "按 Delta Pass 与成本效率比较。", "result": "risk-metrics-calculation +30.0pp 且 Token -34.8%；tdd-workflow +7.1pp 但 Token +78.6%。", "conclusion": "清单式技能可补易遗漏要求，但有时以更长验证路径换正确率。"},
            {"name": "实验 4：负收益和上下文干扰", "what": "深挖 springboot、linkerd、django 三个退化技能。", "why": "解释相关内容为何仍会伤害执行。", "setup": "对 Linkerd 有无技能轨迹和产物做逐步比较。", "result": "Linkerd -9.1pp：复制旧 v1beta1 与 HTTP/1、虚构 rules/metricsServers、额外生成 NetworkPolicy。", "conclusion": "硬编码模板会造成表面锚定、字段幻觉和相邻概念渗漏。"},
        ],
        "limits": ["只使用一个模型和一个 Agent Scaffold。", "一任务一技能，未测试大型技能库检索和组合。", "不少任务基线已近饱和，限制可提升空间。"],
        "fit": 9.8,
        "integration": ["给 Skill Manifest 增加 runtime/model/tool/schema 兼容范围和 review_after。", "保存 content_hash、dependency versions 和 rollback_version。", "评测报告单独显示质量增益与 Token/搜索成本。", "Lint 检查硬编码年份、API 版本、模型名、路径和示例常量。", "Canary 发现负增益后自动回滚旧版本。"],
        "dont": ["不要在 Skill 里固化搜索 Provider 的旧参数或网页 Schema。", "不要把相关示例原样当默认模板。", "不要仅因最终通过率不变就认为技能无影响。"],
        "mvp": ["扩展 skill.json 的 semver 与兼容字段。", "Trace 写入实际版本和依赖快照。", "构造旧 API/旧来源/旧模型的版本错配回归集。"],
        "scores": [8.7, 9.0, 8.6, 9.9, 9.8],
        "key_figures": ["002_Figures_bench-pipeline.png", "003_Figures_finding4_failure_case.png"],
    },
    {
        "id": "2601.10338",
        "order": 10,
        "slug": "agent-skills-in-the-wild",
        "title": "Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale",
        "short": "Agent Skills in the Wild",
        "direction": "Skill 生态安全与漏洞检测",
        "thesis": "Agent Skill 同时承载自然语言控制和可执行资产，形成提示注入、数据外传、提权与供应链四类复合风险；脚本、体量和依赖数是显著风险信号。",
        "findings": ["31,132 个技能中 26.1% 至少命中一种潜在危险模式，估计区间 23%-30%。", "高危 5.2%、中危 8.1%、仅低危 12.8%；不能把 26.1% 等同恶意软件比例。", "数据外传 13.3%、提权 11.8%、供应链 7.4%、提示注入 0.7%。", "带脚本技能漏洞率 40.6% 对 24.2%，OR=2.12；超过 500 行 OR=2.14。"],
        "method": ["从两个市场收集并过滤、去重得到 31,132 个技能和 3,574 个脚本。", "建立 4 类 14 模式漏洞分类和高/中/低严重度。", "SkillScan 结合宽召回静态规则、LLM-Guard 和语义 LLM 分类。", "使用时间隔离的 taxonomy、calibration、validation 样本防止循环验证。"],
        "experiments": [
            {"name": "实验 1：检测框架验证", "what": "在 200 个留出技能上验证静态与混合检测。", "why": "评估生态比例前先校准检测误差。", "setup": "63 个漏洞样本，专家标注；不同开发、校准、验证样本。", "result": "静态规则 precision 71.4%、recall 91.2%；混合后 precision 86.7%、recall 82.5%。", "conclusion": "静态规则适合高召回和复现，语义模型适合减少误报但会牺牲召回。"},
            {"name": "实验 2：漏洞分类与总体流行率", "what": "扫描全部 31,132 个技能。", "why": "量化早期 Skill 生态风险。", "setup": "最终 positive 需完整流程确认且置信度至少 0.6。", "result": "8,126 个被标记，26.1%；高危 1,619、5.2%；中危 2,522、8.1%；仅低危 3,985、12.8%。", "conclusion": "生产系统应把外部技能默认视为待审查供应链输入。"},
            {"name": "实验 3：类别与共现", "what": "比较不同功能类别及漏洞共现。", "why": "识别需要更强审查的 Skill 类型。", "setup": "1,218 个分层类别样本，并用 IPW 校正。", "result": "44.2% 的漏洞技能跨两个以上类别；Supply Chain -> Data Exfiltration 条件共现 81%。安全类原始标记高，但大量属于合法工具，需要上下文审查。", "conclusion": "权限和行为上下文比关键词标签更可靠。"},
            {"name": "实验 4：结构性风险因素", "what": "计算脚本、长度、依赖和维护频率的 OR。", "why": "为发布门禁提供优先检查信号。", "setup": "Fisher exact test 与 Bonferroni 校正。", "result": "脚本 OR=2.12，超过 500 行 OR=2.14，至少 5 个外部依赖 OR=1.58；90 天内活跃维护不显著，p=.47。", "conclusion": "最近更新不是安全证明，脚本和依赖必须单独审查。"},
            {"name": "实验 5：案例与动态确认", "what": "分析外传、Prompt Injection、供应链案例，并对高置信样本做小规模动态检查。", "why": "验证静态模式是否对应可利用行为。", "setup": "87 个最高风险技能中选案例；25 个高置信技能试点。", "result": "试点 72% 表现出可利用行为；论文同时强调未在全量数据上执行潜在恶意技能。", "conclusion": "隔离沙箱和假凭据动态测试应作为高风险技能的补充。"},
        ],
        "limits": ["26.1% 混合了恶意、疏忽和危险但可能合理的模式。", "数据是 2025 年 12 月早期生态快照，17.3% 已删除仓库可能造成存活偏差。", "混合 LLM 检测有 5.5% 运行波动和 9.0% Prompt 变体差异。"],
        "fit": 9.9,
        "integration": ["新增 skill_security.py，与网页 Safe Fetch 分离。", "外部技能先进入 quarantine；扫描 SKILL.md、脚本、资源和依赖。", "Manifest 声明 network/filesystem/subprocess/credentials，沙箱观测实际行为并做 capability diff。", "使用假凭据、Canary 文件和封闭网络测试，不接触真实密钥。", "按高危阻断、中危人工审查、低危告警建立发布策略。"],
        "dont": ["不要把正则命中直接等同恶意。", "不要因为仓库 star 或最近更新而跳过审查。", "不要在正常开发机直接运行未知 Skill 脚本。"],
        "mvp": ["静态扫描 subprocess/env/credential/network/eval。", "生成权限差异报告。", "隔离执行高风险候选并保存完整日志。"],
        "scores": [8.9, 8.8, 8.1, 10.0, 9.9],
        "key_figures": ["001_Fig_methodology.png", "002_Fig_threat_model.png"],
    },
    {
        "id": "2605.11418",
        "order": 11,
        "slug": "skillmd-semantic-supply-chain",
        "title": "Under the Hood of SKILL.md: Semantic Supply-chain Attacks on AI Agent Skill Registry",
        "short": "SKILL.md 语义供应链攻击",
        "direction": "发现、选择与治理阶段的语义攻击",
        "thesis": "SKILL.md 不是被动说明书，而是影响检索、选择和审核的控制面；仅修改自然语言元数据即可完成跨阶段供应链操纵。",
        "findings": ["20-token 发现触发器可让 OpenAI 检索 win rate 达 86.14%，平均分数提升 16.40%。", "直接攻击可进入 OpenAI Top-3/5/10 的比例为 56%/65%/80%。", "仅改 description 后，4 个模型平均 77.6% 选择对抗版本。", "治理上下文溢出使 87.1% 变体被判 clean，且没有一个判 malicious。"],
        "method": ["共享数据为 ClawHub 五类别、每类 top 20，共 100 个真实技能。", "发现攻击用黑盒 beam search 和白盒 gradient 优化短触发器。", "选择攻击只向 description 增加虚假宣传、命令式提示、活跃维护或可信安全声明。", "治理攻击测试 Judge jailbreak、改写、Definition-of-Done 和审核上下文溢出。"],
        "experiments": [
            {"name": "实验 1：发现操纵", "what": "优化 20-token 触发器提高技能与目标查询相似度。", "why": "测试注册表检索是否可被纯文本操纵。", "setup": "BGE base/small 与 OpenAI text-embedding-3-small；100 技能。", "result": "OpenAI 黑盒 win rate 86.14%，平均 +16.40%；BAAI 触发器迁移到 OpenAI 仍约 79%-80% win rate。", "conclusion": "不同 Embedding 之间存在非对称但显著的攻击迁移。"},
            {"name": "实验 2：Top-k 与真实排名", "what": "把对抗技能与同类 top 20 竞争，并模拟下载量。", "why": "验证分数变化是否转化为可见性。", "setup": "Top-3/5/10；ClawHub 风格 lexical+vector+popularity 评分。", "result": "OpenAI 直接攻击 Top-3/5/10 为 56%/65%/80%；平均日下载情景 win 74.14%；0-day 发布时 94%，1 小时后仍 40%。", "conclusion": "Popularity 不能消除冷启动语义操纵。"},
            {"name": "实验 3：选择操纵", "what": "功能相同的原版和描述增强版二选一。", "why": "隔离语言框架对 Agent 选择的影响。", "setup": "100 技能 x 4 变体 x 5 Prompt，每模型 2,000 次；4 个模型。", "result": "对抗版本平均选择率 77.6%；各模型均高于 50%，Gemma 和 Qwen 平均超过 80%。", "conclusion": "Router 不能把自我宣称的最好、最新、安全当可信特征。"},
            {"name": "实验 4：治理基线", "what": "给 47 个 clean 技能加入 94 条直接恶意指令。", "why": "确认审核系统对简单攻击有效。", "setup": "静态扫描、LLM ClawScan、VirusTotal 三阶段。", "result": "没有变体被判 clean，68.2% 判 malicious；若只阻断 malicious，仍约 32% suspicious 可访问。", "conclusion": "治理效果也取决于 enforcement policy，不只是分类器。"},
            {"name": "实验 5：治理规避", "what": "比较四种 SKILL.md-only 改写。", "why": "测试恶意意图能否在语义不变时绕过门禁。", "setup": "Judge jailbreak、paraphrase、DoD、超出 10K 字符审核窗口。", "result": "改写后 73.9% 避免 malicious；DoD 66.3% 避免阻断且 23.9% clean；overflow 87.1% clean、0% malicious。", "conclusion": "必须全文件分块审核，且 suspicious 默认不应自动发布。"},
        ],
        "limits": ["数据仅 100 个热门技能，且五个类别。", "治理管线是论文模拟，不等同任何平台完整生产实现。", "发现实验依赖具体 Embedding 与排名公式。"],
        "fit": 10.0,
        "integration": ["把 description 当不可信输入，独立于 Skill Body 做语义审查。", "Router 排序只使用可验证元数据；忽略 best、trusted、latest 等自我声明。", "比较 description、正文、handler 和实际行为的一致性。", "审核必须覆盖完整文件和引用资源，不能只看前 N 字符。", "提供排名原因、选择原因、审核原因和 revocation 日志。"],
        "dont": ["不要把描述直接拼进高权限系统 Prompt。", "不要只扫描代码而忽略 SKILL.md。", "不要让 suspicious 技能在默认配置中可用。"],
        "mvp": ["为 metadata 建立受信字段白名单。", "加入 description-body-handler 一致性 Judge 与规则。", "构造发现、选择、治理三段攻击回归集。"],
        "scores": [9.4, 9.2, 8.9, 10.0, 10.0],
        "key_figures": ["003_images_overview.png", "005_images_plots_Overall_selection_heatmap.png", "011_images_plots_variant_stacked_columns_black.png"],
    },
    {
        "id": "2608.08453",
        "order": 12,
        "slug": "skill-reusability-138k",
        "title": "What Keeps Agent Skills from Being Reusable? Evidence from 138K SKILL.md Files",
        "short": "138K SKILL.md 可复用性研究",
        "direction": "技能工程质量、Lint 与可复用性",
        "thesis": "公开技能的大多数问题发生在执行之前：路由描述、正文可操作性和资源组织缺陷会阻断发现、浪费上下文或制造安全与可移植性风险。",
        "findings": ["138,133 个唯一 SKILL.md、20,556 个仓库；91.8% 至少一个检测缺陷，89.3% 至少一个规范缺陷。", "67.0% 有路由缺陷；缺少触发指导 52.3%，名称重复标题 44.3%，内联示例过多 32.1%。", "R1-clean 在元数据查询中 Hit@1 88.5%，R1-defective 82.6%。", "三条轻量规则覆盖 71.9% 的缺陷实例；自动修复平均修复 58.4%，但安全绕过和注入修复率为 0%。"],
        "method": ["从 GitHub、仓库克隆和 agentskills.in 收集并以 SHA-256 去重。", "建立两层 7 类 31 条检查：规范符合性与最佳实践。", "每个 Detector 有正负 mutation fixture，并对阈值做敏感性检查。", "以 20,000 个技能执行 description-only BM25 路由压力测试。", "结合平台、AI 标记、GitHub issues、优质样本和自动修复实验。"],
        "experiments": [
            {"name": "实验 1：生态缺陷普查", "what": "对 138,133 个技能运行 31 条 Detector。", "why": "判断可复用性问题是边缘现象还是系统性问题。", "setup": "14 条 Tier 1 规范检查，17 条 Tier 2 最佳实践检查。", "result": "89.3% 命中 Tier 1，91.8% 任一层，平均 2.5 个缺陷、中位数 2。", "conclusion": "Skill 生成和发布流程必须内置质量门禁。"},
            {"name": "实验 2：路由压力测试", "what": "只索引 description，检测能否找回来源技能。", "why": "给最直接的路由缺陷提供功能性证据。", "setup": "20,000 技能；元数据查询和更严格的名称/路径查询。", "result": "元数据 Hit@1：clean 88.5% vs defective 82.6%，MRR 0.906 vs 0.855；名称/路径 26.7% vs 24.3%。", "conclusion": "描述质量会影响发现，但 BM25 只是语义 Router 的下界代理。"},
            {"name": "实验 3：平台与来源信号", "what": "比较平台、规范意识和 AI 标记。", "why": "寻找质量关联因素。", "setup": "路径可归因平台，Mann-Whitney/Kruskal-Wallis 与效应量。", "result": "规范感知技能平均 1.83 个缺陷 vs 3.00；AI 标记 3.23 vs 未标记 2.34，安全缺陷 18.9% vs 8.2%，但论文不作因果结论。", "conclusion": "生成阶段需要更强 Lint，但 AI 标记只是审查信号。"},
            {"name": "实验 4：优质样本特征", "what": "人工阅读高星仓库中 419 个零检测缺陷技能。", "why": "从负面缺陷转向可操作的正面规范。", "setup": "2,406 个高星仓库技能中筛选 419 个 exemplars。", "result": "共同特征包括完整触发、命令式正文、少冗余、无禁止内容；非同义反复的发现是它们编码本地程序知识而非通用教程。", "conclusion": "好的 Skill 应提供模型无法轻易重建的局部流程。"},
            {"name": "实验 5：渐进 Lint 覆盖", "what": "按边际覆盖累加 12 条指南。", "why": "寻找低成本质量门禁。", "setup": "模拟累积规则覆盖缺陷实例和技能阻断率。", "result": "G1 覆盖 33.4%；G1+G4+G7 覆盖 71.9% 且标记 85.9% 技能；全套覆盖 99.4%。", "conclusion": "三条轻量规则适合开发期，但不能替代语义和安全审查。"},
            {"name": "实验 6：自动修复", "what": "对 200 个缺陷技能进行基于指南的 LLM 修复。", "why": "判断缺陷能否自动消除。", "setup": "提供检测列表和对应指南，再重新运行 Detector。", "result": "平均修复 58.4% 已检测缺陷，每技能新增 0.06；安全绕过与 Prompt Injection 为 0% 修复。", "conclusion": "结构问题可自动修，安全问题必须非绕过门禁，重度缺陷宜重生成。"},
        ],
        "limits": ["静态正则可能误报或漏报，91.8% 不能解释成 92% 的技能完全不可用。", "BM25 不等同生产 LLM Selector。", "平台比较只覆盖可由路径归因的 41.7%，AI 标记分析存在混杂。"],
        "fit": 10.0,
        "integration": ["实现 Skill Lint：routing、body、resources、prohibited、safety、portability、scope。", "错误阻断加载，警告只允许留在 candidate，不直接 production。", "Description 采用 Verb+what+Use when trigger，并加入 contraindications。", "正文保持命令式；代码、长示例和参考资料渐进披露。", "严重缺陷优先重生成，安全缺陷不允许自动修复后无复核发布。"],
        "dont": ["不要把 cosmetic defect 与凭据泄漏赋予相同严重度。", "不要只做格式 Lint 就宣称安全。", "不要让自动修复覆盖原版本和审计记录。"],
        "mvp": ["先实现 G1、G4、G7 三规则及现有路径/凭据检查。", "把 Lint 接入 loader doctor、tests 和 release gate。", "对现有 skills/ 输出一次基线缺陷报告。"],
        "scores": [8.9, 8.8, 8.0, 10.0, 10.0],
        "key_figures": ["003_figures_defect_prevalence.png", "004_figures_progressive_enforcement.png"],
    },
]


def esc(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br/>")


def styles():
    registerFont(TTFont("MSYH", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
    registerFont(TTFont("MSYH-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
    base = getSampleStyleSheet()
    font = "MSYH"
    return {
        "title": ParagraphStyle("TitleCN", parent=base["Title"], fontName="MSYH-Bold", fontSize=24, leading=31, textColor=colors.HexColor("#12324A"), alignment=TA_LEFT, spaceAfter=12),
        "subtitle": ParagraphStyle("SubtitleCN", parent=base["Normal"], fontName=font, fontSize=12, leading=18, textColor=colors.HexColor("#496679"), spaceAfter=8),
        "h1": ParagraphStyle("H1CN", parent=base["Heading1"], fontName="MSYH-Bold", fontSize=16, leading=22, textColor=colors.HexColor("#0B6E75"), spaceBefore=14, spaceAfter=8),
        "h2": ParagraphStyle("H2CN", parent=base["Heading2"], fontName="MSYH-Bold", fontSize=12.5, leading=18, textColor=colors.HexColor("#12324A"), spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("BodyCN", parent=base["BodyText"], fontName=font, fontSize=9.5, leading=15, textColor=colors.HexColor("#20313D"), spaceAfter=5),
        "small": ParagraphStyle("SmallCN", parent=base["BodyText"], fontName=font, fontSize=7.5, leading=11, textColor=colors.HexColor("#587080"), spaceAfter=3),
        "bullet": ParagraphStyle("BulletCN", parent=base["BodyText"], fontName=font, fontSize=9.2, leading=14.5, leftIndent=12, firstLineIndent=-8, bulletIndent=2, spaceAfter=4),
        "label": ParagraphStyle("LabelCN", parent=base["BodyText"], fontName=font, fontSize=8.8, leading=13.5, textColor=colors.HexColor("#20313D"), spaceAfter=2),
        "callout": ParagraphStyle("CalloutCN", parent=base["BodyText"], fontName=font, fontSize=11, leading=18, textColor=colors.HexColor("#0B4B55"), borderColor=colors.HexColor("#8EC7C9"), borderWidth=0.8, borderPadding=10, backColor=colors.HexColor("#EAF6F5"), spaceAfter=10),
        "caption": ParagraphStyle("CaptionCN", parent=base["BodyText"], fontName=font, fontSize=7.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#587080"), spaceAfter=7),
    }


def bullets(items, st):
    return [Paragraph("- " + esc(item), st["bullet"]) for item in items]


def fit_image(path: Path, max_w=166 * mm, max_h=88 * mm):
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_w / width, max_h / height)
    return Image(str(path), width=width * scale, height=height * scale)


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename, title_text, **kwargs):
        super().__init__(filename, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm, **kwargs)
        self.title_text = title_text
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=self._decorate))

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B8CDD4"))
        canvas.setLineWidth(0.4)
        canvas.line(20 * mm, A4[1] - 13 * mm, A4[0] - 20 * mm, A4[1] - 13 * mm)
        canvas.setFont("MSYH", 7.5)
        canvas.setFillColor(colors.HexColor("#587080"))
        canvas.drawString(20 * mm, A4[1] - 10 * mm, self.title_text[:55])
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{doc.page}")
        canvas.restoreState()


def figure_block(path: Path, caption: str, st, max_h=88 * mm):
    if not path.exists():
        return []
    return [Spacer(1, 3 * mm), fit_image(path, max_h=max_h), Paragraph(esc(caption), st["caption"])]


def markdown_for(paper, authors):
    lines = [
        "---",
        f'date: "{TODAY}"',
        f'paper_id: "arXiv:{paper["id"]}"',
        f'title: "{paper["title"]}"',
        f'authors: "{authors}"',
        'domain: "Agent-Skills"',
        f'quality_score: "{sum(paper["scores"]) / 5:.1f}/10"',
        'status: "analyzed"',
        "---",
        "",
        f'# {paper["short"]} 中文深度解读',
        "",
        f'- 论文：{paper["title"]}',
        f'- arXiv：[{paper["id"]}](https://arxiv.org/abs/{paper["id"]})',
        f'- 方向：{paper["direction"]}',
        f'- 对项目适配度：{paper["fit"]}/10',
        "",
        "## 一页速读",
        "",
        paper["thesis"],
        "",
    ]
    lines.extend([f'- {item}' for item in paper["findings"]])
    lines.extend(["", "## 方法概述", ""])
    lines.extend([f'- {item}' for item in paper["method"]])
    lines.extend(["", "## 按论文顺序实验记录", ""])
    for exp in paper["experiments"]:
        lines.extend([
            f'### {exp["name"]}', "",
            f'- 做了什么：{exp["what"]}',
            f'- 为什么：{exp["why"]}',
            f'- 设置：{exp["setup"]}',
            f'- 关键结果：{exp["result"]}',
            f'- 结论：{exp["conclusion"]}', "",
        ])
    lines.extend(["## 局限性", ""] + [f'- {x}' for x in paper["limits"]])
    lines.extend(["", "## 对 Deep Research 项目的融入", ""] + [f'- {x}' for x in paper["integration"]])
    lines.extend(["", "### 不宜照搬", ""] + [f'- {x}' for x in paper["dont"]])
    lines.extend(["", "### 最小落地闭环", ""] + [f'{i}. {x}' for i, x in enumerate(paper["mvp"], 1)])
    lines.extend(["", "## 来源", "", f'- [arXiv 摘要页](https://arxiv.org/abs/{paper["id"]})', f'- [原始 PDF](https://arxiv.org/pdf/{paper["id"]})', ""])
    return "\n".join(lines)


def build_pdf(paper, meta, st):
    authors = meta.get("metadata", {}).get("Author", "--") or "--"
    author_short = authors
    if author_short.count(";") > 7:
        author_short = ";".join(author_short.split(";")[:7]) + "; et al."
    output = PDF_ROOT / f'{paper["order"]:02d}_{paper["id"]}_{paper["slug"]}_中文解读.pdf'
    doc = PaperDocTemplate(str(output), paper["short"], title=paper["title"], author=authors, subject="Agent Skills 论文中文深度解读")
    story = []
    story.extend([
        Spacer(1, 10 * mm),
        Paragraph(esc(paper["short"]), st["title"]),
        Paragraph(esc(paper["title"]), st["subtitle"]),
        Spacer(1, 4 * mm),
        Paragraph(f'arXiv:{paper["id"]} | {esc(paper["direction"])} | {meta.get("pages", "--")} 页原文', st["subtitle"]),
        Paragraph("作者：" + esc(author_short), st["small"]),
        Spacer(1, 5 * mm),
        Paragraph("核心判断", st["h1"]),
        Paragraph(esc(paper["thesis"]), st["callout"]),
    ])
    quick_data = [[Paragraph("适配度", st["label"]), Paragraph(f'{paper["fit"]:.1f}/10', st["label"]), Paragraph("阅读优先级", st["label"]), Paragraph("核心", st["label"])] ]
    quick = Table(quick_data, colWidths=[25 * mm, 30 * mm, 30 * mm, 30 * mm])
    quick.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F1F6F7")), ("BOX", (0,0), (-1,-1), 0.4, colors.HexColor("#B8CDD4")), ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#D6E3E7")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
    story.extend([quick, Paragraph("你应该记住的结果", st["h1"]), *bullets(paper["findings"], st)])

    images_dir = NOTE_ROOT / paper["id"] / "images"
    for index, filename in enumerate(paper.get("key_figures", []), 1):
        story.extend(figure_block(images_dir / filename, f'原论文关键图 {index}：{filename}。正文后附完整图表索引。', st))

    story.extend([Paragraph("1. 研究问题与方法架构", st["h1"]), Paragraph("论文试图回答：程序性知识怎样被表示、获得、选择、执行、验证或治理，以及这种机制是否在真实任务中带来可测量的边际收益。", st["body"]), *bullets(paper["method"], st)])
    story.append(Paragraph("2. 按论文顺序实验记录", st["h1"]))
    for exp in paper["experiments"]:
        rows = [
            [Paragraph("做了什么", st["label"]), Paragraph(esc(exp["what"]), st["body"])],
            [Paragraph("为什么", st["label"]), Paragraph(esc(exp["why"]), st["body"])],
            [Paragraph("设置", st["label"]), Paragraph(esc(exp["setup"]), st["body"])],
            [Paragraph("关键结果", st["label"]), Paragraph(esc(exp["result"]), st["body"])],
            [Paragraph("支持结论", st["label"]), Paragraph(esc(exp["conclusion"]), st["body"])],
        ]
        table = Table(rows, colWidths=[26 * mm, 137 * mm], repeatRows=0)
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EAF6F5")), ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8CDD4")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
        story.extend([Paragraph(esc(exp["name"]), st["h2"]), table, Spacer(1, 2 * mm)])

    story.extend([Paragraph("3. 可信度与局限", st["h1"]), Paragraph("以下局限决定了论文结论能否直接外推到开放式 Deep Research：", st["body"]), *bullets(paper["limits"], st)])
    story.extend([Paragraph("4. 对你的 Deep Research 项目的融入", st["h1"]), Paragraph(f'适配度：{paper["fit"]:.1f}/10。建议把论文方法转译为以下工程动作：', st["callout"]), *bullets(paper["integration"], st), Paragraph("不宜照搬", st["h2"]), *bullets(paper["dont"], st), Paragraph("最小落地闭环", st["h2"]), *[Paragraph(f'{i}. {esc(item)}', st["body"]) for i, item in enumerate(paper["mvp"], 1)]])

    score_names = ["创新性", "技术质量", "实验充分性", "对项目实用性", "综合推荐"]
    score_rows = [[Paragraph("维度", st["label"]), Paragraph("分数", st["label"])] ] + [[Paragraph(name, st["body"]), Paragraph(f"{score:.1f}/10", st["body"])] for name, score in zip(score_names, paper["scores"])]
    score_table = Table(score_rows, colWidths=[70 * mm, 35 * mm])
    score_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#12324A")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8CDD4")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,1), (1,-1), "CENTER"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F9FA")]), ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
    story.extend([Paragraph("5. 综合评价", st["h1"]), score_table, Paragraph("建议带着这些问题精读", st["h2"]), *bullets(["核心增益来自技能内容、路由、执行器还是额外计算？", "负增益案例是否被平均数掩盖？", "论文的验证器能否替换为证据、引用与交付质量指标？", "若技能可以写入和更新，谁批准、如何回滚、怎样追责？"], st)])

    story.extend([PageBreak(), Paragraph("原论文图表附录", st["h1"]), Paragraph("以下图表来自 arXiv 源码包。它们用于辅助复核论文的方法、实验和附录材料；文件名保留作者源码语义。", st["body"])])
    all_images = sorted(images_dir.glob("*.png"))
    for i, path in enumerate(all_images, 1):
        story.extend(figure_block(path, f'图表 {i}/{len(all_images)}：{path.name}', st, max_h=105 * mm))
    story.extend([Paragraph("来源与版本", st["h1"]), Paragraph(f'论文：{esc(paper["title"])}', st["body"]), Paragraph(f'arXiv：https://arxiv.org/abs/{paper["id"]}', st["body"]), Paragraph(f'原始 PDF：https://arxiv.org/pdf/{paper["id"]}', st["body"]), Paragraph(f'本解读生成日期：{TODAY}。论文状态按 arXiv 预印本处理，不据此推断正式录用。', st["small"])])
    doc.build(story)
    return output


def main():
    PDF_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {item["paper_id"]: item for item in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    st = styles()
    outputs = []
    for paper in PAPERS:
        meta = manifest[paper["id"]]
        authors = meta.get("metadata", {}).get("Author", "--") or "--"
        note_dir = NOTE_ROOT / paper["id"]
        note_dir.mkdir(parents=True, exist_ok=True)
        (note_dir / "analysis.md").write_text(markdown_for(paper, authors), encoding="utf-8")
        outputs.append(build_pdf(paper, meta, st))
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
