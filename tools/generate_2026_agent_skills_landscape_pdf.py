from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp" / "pdfs"
OUTPUT = ROOT / "output" / "pdf" / "2026-agent-skills-landscape-cn.pdf"
CUTOFF = "2026-08-27"


def load_arxiv_entries() -> dict[str, dict[str, str]]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    rows: dict[str, dict[str, str]] = {}
    for filename in ("arxiv-agent-skill.xml", "arxiv-agent-skill-page2.xml"):
        root = ET.parse(TMP / filename).getroot()
        for entry in root.findall("a:entry", ns):
            raw_id = entry.findtext("a:id", "", ns).rsplit("/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)
            published = entry.findtext("a:published", "", ns)[:10]
            if not published.startswith("2026"):
                continue
            title = " ".join(entry.findtext("a:title", "", ns).split())
            abstract = " ".join(entry.findtext("a:summary", "", ns).split())
            authors = [
                author.findtext("a:name", "", ns)
                for author in entry.findall("a:author", ns)
            ]
            rows[arxiv_id] = {
                "id": arxiv_id,
                "title": title,
                "date": published,
                "abstract": abstract,
                "authors": ", ".join(authors),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
                "source": "arXiv API",
            }
    extras = {
        "2604.08224": ("2026-04-09", "Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering"),
        "2604.24594": ("2026-04-27", "Skill Retrieval Augmentation for Agentic AI"),
        "2604.27660": ("2026-04-30", "From Context to Skills: Can Language Models Learn from Context Skillfully?"),
        "2605.17734": ("2026-05-18", "Harnessing LLM Agents with Skill Programs"),
        "2607.07676": ("2026-07-08", "SkillCenter: A Large-Scale Source-Grounded Skill Library for Autonomous AI Agents"),
        "2607.15257": ("2026-07-16", "SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration"),
        "2607.29468": ("2026-07-31", "Self-Play Meets Skill Evolution: Self-Evolving Search Agents that Pose, Solve, and Remember"),
        "2608.01913": ("2026-08-03", "Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents"),
        "2608.16114": ("2026-08-17", "HyperSkill: Self-Evolving LLM Agents via Hypergraph-Structured Skill Memory"),
    }
    for arxiv_id, (date, title) in extras.items():
        rows.setdefault(
            arxiv_id,
            {
                "id": arxiv_id,
                "title": title,
                "date": date,
                "abstract": "",
                "authors": "",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
                "source": "交叉补充",
            },
        )
    return rows


CATEGORIES = [
    {
        "title": "1. 综述、定义与领域边界",
        "question": "Agent Skill 到底是什么，生命周期如何划分，与工具、记忆和协议有什么区别？",
        "papers": [
            ("2605.07358", "四阶段生命周期：表示、获取、检索、进化；适合作为总目录。", 5),
            ("2602.12430", "强调 SKILL.md、渐进加载、MCP 与四级信任治理。", 4),
            ("2602.20867", "从 SoK 视角划清 Skill 与单次工具调用、提示词的边界。", 4),
            ("2606.11435", "把评测与演化放进同一框架，适合设计 Skill 生命周期指标。", 5),
            ("2607.10113", "聚焦动态技能库、版本漂移与持续维护。", 4),
            ("2604.08224", "把记忆、Skill、协议和 Harness 统一为外部化能力层。", 5),
        ],
    },
    {
        "title": "2. Skill 表示、规格与工程化",
        "question": "Skill 应保存成文本、代码、图、类型化 Harness，还是多模态资源包？",
        "papers": [
            ("2604.24026", "把自然语言 Skill 降解为调度、结构、逻辑三层表示。", 4),
            ("2607.18970", "提出持久行为制品的软件本体与完整工程生命周期。", 4),
            ("2607.27309", "把 Skill 编译为类型化 Harness，强化接口和运行时约束。", 5),
            ("2605.03353", "面向跨框架移植与安全编译，直接关联 Windows/macOS 可移植性。", 4),
            ("2606.01414", "论证纯文本不足，提出视觉 Skill 作为混合表示。", 2),
            ("2605.17734", "使用 Skill Program 表达可执行流程，减少纯提示词歧义。", 4),
        ],
    },
    {
        "title": "3. Skill 获取、蒸馏与自动生成",
        "question": "如何从轨迹、上下文、资源、文档和在线反馈中形成可复用 Skill？",
        "papers": [
            ("2603.25158", "Trace2Skill：逐轨迹独立提炼补丁，再统一合并。", 5),
            ("2604.27660", "Ctx2Skill：从上下文中抽取触发条件与程序性经验。", 5),
            ("2606.29538", "RESOURCE2SKILL：从人类多模态资源蒸馏可执行技能。", 4),
            ("2605.10999", "SkillGen：推理时生成 Skill，并通过验证器筛选。", 4),
            ("2606.11897", "Notes2Skills：从实验记录生成带置信度的科研 Skill。", 5),
            ("2608.01678", "用强化学习逐步生成 Skill，代表训练式获取路线。", 3),
        ],
    },
    {
        "title": "4. Skill 检索、选择、装载与规模化",
        "question": "技能库变大后，如何找对 Skill，并判断该不该加载？",
        "papers": [
            ("2604.24594", "提出 SRA 与 SRA-Bench，拆分检索、吸收和任务执行。", 5),
            ("2605.05726", "SkillRet：面向大规模 Skill Retrieval 的专用基准。", 4),
            ("2606.03565", "强调 Skill 不是普通文档，路由必须建模查询兼容性。", 5),
            ("2605.06978", "通过技能组结构提升检索，避免扁平向量库的孤立匹配。", 4),
            ("2608.08640", "用显式推理增强隐式需求下的 Skill 检索。", 4),
            ("2608.14943", "比较预加载、按需加载、渐进披露和混合策略。", 5),
            ("2608.25500", "以反事实因果 Skill Graph 支持可扩展检索。", 4),
        ],
    },
    {
        "title": "5. 自进化、优化与自博弈",
        "question": "怎样让 Skill 在反馈下持续变好，同时避免漂移和错误放大？",
        "papers": [
            ("2605.23904", "SkillOpt：有界 add/delete/replace、验证门、拒绝缓冲和慢更新。", 5),
            ("2604.01687", "CoEvoSkills：Skill Generator 与 Surrogate Verifier 共同进化。", 5),
            ("2605.27760", "SkillGrad：把文本 Skill 更新类比为梯度优化。", 4),
            ("2608.07449", "SkillProx：使用近端文本梯度限制更新跨度。", 4),
            ("2607.22529", "Skill Self-Play：提议者、求解器和技能控制器共同训练。", 3),
            ("2607.29468", "SESA：搜索自博弈与外部技能记忆双向共演化。", 5),
            ("2607.05297", "两时间尺度 Meta-Skill Evolution，区分快改动与慢规则。", 5),
            ("2606.14239", "SkillAudit：无 Ground Truth 时用成对轨迹审计演化结果。", 5),
        ],
    },
    {
        "title": "6. 程序记忆、知识图谱与技能库生命周期",
        "question": "少量经验如何保存、组合、检索、去重、淘汰并保持可追溯？",
        "papers": [
            ("2602.02474", "MemSkill：把记忆能力本身做成可学习、可演化 Skill。", 5),
            ("2608.16114", "HyperSkill：用超图连接子任务、轨迹和 Skill，双路径检索。", 5),
            ("2606.04781", "AIP：图表示同时服务学习、依赖分析和治理。", 4),
            ("2607.07676", "SkillCenter：大规模来源绑定技能库与离线 FTS5 分发。", 5),
            ("2606.16523", "SkillWiki：把技能库建设成持续更新的知识基础设施。", 4),
            ("2605.13716", "SkillOps：把技能库当作需要监控、测试和维护的软件生态。", 5),
            ("2608.05204", "SkillTrace：多轨迹来源审计，追踪一条 Skill 的经验依据。", 5),
            ("2608.21964", "Repo2Skill-Evo：研究代码库 Skill 静默过期与维护问题。", 4),
        ],
    },
    {
        "title": "7. Benchmark、可解释评测与负结果",
        "question": "如何分别评估 Skill 的检索、吸收、执行、泛化、成本与副作用？",
        "papers": [
            ("2602.12670", "SkillsBench：跨任务评估 Skill 是否真正带来增益。", 5),
            ("2608.06891", "SkillEval：把质量拆成可解释的多个信号，而不是单一成功率。", 5),
            ("2603.28815", "SkillTester：同时测试效用与安全。", 4),
            ("2607.16345", "AEVAL：从案例式验证转向确定性工作流测试。", 5),
            ("2607.01874", "SkillCoach：让 Rubric 自进化，并反过来指导 Skill 改进。", 4),
            ("2605.20023", "When Skills Don't Help：重要负结果，提醒 Skill 不是万能增益。", 5),
            ("2605.24050", "More Skills, Worse Agents：技能库扩张可能导致 Skill Shadowing。", 5),
            ("2608.14036", "分析 Skill 为什么时灵时不灵，强调环境与 Harness 依赖。", 4),
        ],
    },
    {
        "title": "8. 安全、权限、供应链与治理",
        "question": "第三方 Skill 如何隔离权限、发现恶意行为、审计组合风险？",
        "papers": [
            ("2606.03024", "SkillGuard：Manifest、Protected Object 与运行时权限控制。", 5),
            ("2607.13987", "建立 Agent Skill 威胁模型、攻击、防御与评测总览。", 5),
            ("2606.07131", "MalSkillBench：运行时验证恶意 Skill 的检测能力。", 4),
            ("2605.09163", "FORTIS：评测 Skill 的过度授权问题。", 5),
            ("2605.05868", "SkillScope：细粒度最小权限执行。", 5),
            ("2606.02540", "SkillHarm：覆盖构造、传播、执行的生命周期攻击。", 4),
            ("2608.16246", "CompoSkill：单独安全的 Skill 组合后仍可能形成攻击链。", 5),
            ("2608.03485", "SkillSentry：用 Honey World 动态测试 Skill 行为。", 4),
            ("2608.21929", "SkillBloat：通过 Skill 注入制造 Token 放大与上下文挤占。", 4),
        ],
    },
    {
        "title": "9. Deep Research、搜索与科研工作流",
        "question": "Skill 如何服务长程搜索、证据覆盖、科研数据和领域研究流程？",
        "papers": [
            ("2607.15257", "SearchOS：Coverage Map、Failure Memory、并行调度与层级 Skill。", 5),
            ("2608.01913", "区分 Retrieval Gap 与 Utilization Gap，给出低样本弱信号。", 5),
            ("2607.29468", "搜索任务自博弈与技能记忆共演化。", 5),
            ("2606.05525", "SciVisAgentSkills：科研数据分析和可视化中的 Skill 设计。", 4),
            ("2606.11897", "从实验笔记生成 certainty-aware 科研技能。", 5),
            ("2608.19625", "把科学数据服务封装成 Agent-ready Skills。", 4),
            ("2604.03964", "从异构科研资源构建自进化 Skill Library。", 4),
        ],
    },
    {
        "title": "10. 生态、市场、软件工程、GUI 与具身应用",
        "question": "公开技能生态如何演化，不同应用域的 Skill 有何工程特征？",
        "papers": [
            ("2602.08004", "分析 40,285 个公开 Skill 的类型、采用、冗余和风险。", 4),
            ("2608.10906", "GitSkills：GitHub 上 Agent Skills 的数据集。", 3),
            ("2607.15557", "SkillCorpus：整合并评估真实开放技能生态。", 4),
            ("2607.09065", "分析技能市场中软件工程活动如何转成可复用 Skill。", 3),
            ("2603.15401", "SWE-Skills-Bench：软件工程中 Skill 是否真正有效。", 4),
            ("2601.21123", "CUA-Skill：面向计算机操作 Agent 的 Skill 构建。", 3),
            ("2607.00272", "ASPIRE：面向机器人策略的 Agentic Skill Discovery。", 2),
            ("2608.03387", "RoboReact：从第一视角视频蒸馏可泛化的具身技能。", 2),
        ],
    },
]


PRIORITY = [
    ("2608.01913", "P0", "先建立 Retrieval Gap、Utilization Gap、Evidence Novelty 与停止信号。"),
    ("2605.23904", "P0", "将 Skill 修改限制为可回滚的小 Patch，并引入验证门。"),
    ("2603.25158", "P0", "少量轨迹独立产生 CandidatePatch；不要直接写入 Stable Skill。"),
    ("2606.14239", "P0", "没有绝对 Ground Truth 时，采用新旧轨迹成对审计。"),
    ("2602.02474", "P0", "单次经验先进入 Episode/Procedural Memory，而非全局规则。"),
    ("2607.15257", "P0", "复用 Coverage Map、Failure Memory 和持续任务调度。"),
    ("2606.03024", "P0", "为 Skill 增加 Manifest、工具权限和可访问对象边界。"),
    ("2604.24594", "P1", "把 Skill Retrieval、Skill Incorporation、Execution 分开测。"),
    ("2608.16114", "P1", "用图关系组织子任务、轨迹与 Skill，补足扁平向量检索。"),
    ("2604.01687", "P1", "在确定性检查器可用时引入 Generator-Verifier 迭代。"),
    ("2607.07676", "P1", "引入来源绑定、离线索引和技能库质量门。"),
    ("2605.24050", "P1", "控制 Skill 数量，监控扩库导致的 shadowing 与误路由。"),
]


class LifecycleDiagram(Flowable):
    def __init__(self, width: float, height: float = 62 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        labels = ["Representation", "Acquisition", "Retrieval", "Execution", "Evaluation", "Evolution", "Governance"]
        subtitles = ["文本/代码/图", "轨迹/资源/生成", "召回/选择/装载", "工具与 Harness", "效用/成本/风险", "Patch/验证/回滚", "权限/版本/淘汰"]
        margin = 3 * mm
        gap = 2.3 * mm
        box_w = (self.width - 2 * margin - gap * (len(labels) - 1)) / len(labels)
        y = 20 * mm
        h = 24 * mm
        palette = [colors.HexColor("#DDEFF1"), colors.HexColor("#E9EEF8"), colors.HexColor("#E8F1E4"), colors.HexColor("#F3EBDC"), colors.HexColor("#F6E2DF"), colors.HexColor("#EDE4F4"), colors.HexColor("#E3E6EC")]
        c.setStrokeColor(colors.HexColor("#7A8796"))
        for i, (label, subtitle) in enumerate(zip(labels, subtitles)):
            x = margin + i * (box_w + gap)
            c.setFillColor(palette[i])
            c.roundRect(x, y, box_w, h, 2.2 * mm, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#10253F"))
            c.setFont("YaHeiBold", 8.5)
            c.drawCentredString(x + box_w / 2, y + 14 * mm, label)
            c.setFont("YaHei", 7.4)
            c.drawCentredString(x + box_w / 2, y + 7 * mm, subtitle)
            if i < len(labels) - 1:
                ax = x + box_w + 0.5 * mm
                c.setStrokeColor(colors.HexColor("#617083"))
                c.line(ax, y + h / 2, ax + gap - 1 * mm, y + h / 2)
                c.line(ax + gap - 2.2 * mm, y + h / 2 + 1.2 * mm, ax + gap - 1 * mm, y + h / 2)
                c.line(ax + gap - 2.2 * mm, y + h / 2 - 1.2 * mm, ax + gap - 1 * mm, y + h / 2)
        c.setFont("YaHei", 8)
        c.setFillColor(colors.HexColor("#526174"))
        c.drawString(margin, 10 * mm, "核心闭环：执行反馈进入 Evaluation，经过受限 Evolution，再由 Governance 决定是否发布。")


class MonthBars(Flowable):
    def __init__(self, counts: dict[int, int], width: float, height: float = 57 * mm):
        super().__init__()
        self.counts = counts
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        left, bottom = 16 * mm, 12 * mm
        chart_w = self.width - left - 4 * mm
        chart_h = self.height - bottom - 5 * mm
        values = [self.counts.get(month, 0) for month in range(1, 9)]
        max_v = max(values) or 1
        bar_w = chart_w / 8 * 0.62
        c.setStrokeColor(colors.HexColor("#B1BAC5"))
        c.line(left, bottom, left, bottom + chart_h)
        c.line(left, bottom, left + chart_w, bottom)
        for idx, value in enumerate(values):
            slot = chart_w / 8
            x = left + idx * slot + (slot - bar_w) / 2
            h = chart_h * value / max_v
            c.setFillColor(colors.HexColor("#287E8A"))
            c.rect(x, bottom, bar_w, h, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#172B43"))
            c.setFont("YaHei", 7.8)
            c.drawCentredString(x + bar_w / 2, bottom + h + 2 * mm, str(value))
            c.drawCentredString(x + bar_w / 2, bottom - 4 * mm, f"{idx + 1}月")
        c.saveState()
        c.translate(5 * mm, bottom + chart_h / 2)
        c.rotate(90)
        c.setFont("YaHei", 7.6)
        c.drawCentredString(0, 0, "arXiv 检索池条目数")
        c.restoreState()


def register_fonts() -> None:
    regular_candidates = [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]
    bold_candidates = [Path(r"C:\Windows\Fonts\msyhbd.ttc"), Path(r"C:\Windows\Fonts\msyh.ttc")]
    regular = next(path for path in regular_candidates if path.exists())
    bold = next(path for path in bold_candidates if path.exists())
    pdfmetrics.registerFont(TTFont("YaHei", str(regular)))
    pdfmetrics.registerFont(TTFont("YaHeiBold", str(bold)))


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def create_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="YaHeiBold", fontSize=24, leading=32, textColor=colors.HexColor("#10253F"), alignment=TA_LEFT, spaceAfter=10),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="YaHei", fontSize=11, leading=18, textColor=colors.HexColor("#526174"), spaceAfter=12),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="YaHeiBold", fontSize=17, leading=24, textColor=colors.HexColor("#0D6570"), spaceBefore=7, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="YaHeiBold", fontSize=12.5, leading=18, textColor=colors.HexColor("#173C62"), spaceBefore=5, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="YaHei", fontSize=9.2, leading=15, textColor=colors.HexColor("#26384D"), spaceAfter=5),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="YaHei", fontSize=7.4, leading=11, textColor=colors.HexColor("#526174"), spaceAfter=2),
        "paper_title": ParagraphStyle("paper_title", parent=base["BodyText"], fontName="YaHeiBold", fontSize=9.4, leading=14, textColor=colors.HexColor("#12385B"), spaceAfter=2),
        "paper_body": ParagraphStyle("paper_body", parent=base["BodyText"], fontName="YaHei", fontSize=8.2, leading=12.5, textColor=colors.HexColor("#36485C"), spaceAfter=0),
        "center": ParagraphStyle("center", parent=base["BodyText"], fontName="YaHei", fontSize=8.5, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#526174")),
    }


def paper_card(entry: dict[str, str], takeaway: str, fit: int, styles) -> Table:
    stars = "●" * fit + "○" * (5 - fit)
    title = Paragraph(
        f'<a href="{esc(entry["url"])}"><b>{esc(entry["title"])}</b></a>',
        styles["paper_title"],
    )
    meta = Paragraph(
        f'arXiv:{entry["id"]} · {entry["date"]} · 项目适配 {stars} · '
        f'<a href="{esc(entry["pdf"])}">原始 PDF</a>',
        styles["small"],
    )
    body = Paragraph(esc(takeaway), styles["paper_body"])
    table = Table([[title], [meta], [body]], colWidths=[170 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FA")),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#C8D2DC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, 1), 2),
            ]
        )
    )
    return table


def add_page_number(c: canvas.Canvas, doc: SimpleDocTemplate) -> None:
    c.saveState()
    width, _ = A4
    c.setStrokeColor(colors.HexColor("#D3DAE2"))
    c.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
    c.setFont("YaHei", 7.5)
    c.setFillColor(colors.HexColor("#667487"))
    c.drawString(20 * mm, 9 * mm, "2026 Agent Skills 论文图谱 · 截至 2026-08-27")
    c.drawRightString(width - 20 * mm, 9 * mm, str(doc.page))
    c.restoreState()


def build_pdf() -> None:
    register_fonts()
    entries = load_arxiv_entries()
    styles = create_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    selected_ids = []
    for category in CATEGORIES:
        selected_ids.extend(paper_id for paper_id, _, _ in category["papers"])
    unique_selected = list(dict.fromkeys(selected_ids))
    missing = [paper_id for paper_id in unique_selected if paper_id not in entries]
    if missing:
        raise RuntimeError(f"Missing metadata for: {missing}")

    # The official API query is deliberately broad. It is preserved as a candidate
    # index; the main report is the manually screened representative set.
    pool = sorted(entries.values(), key=lambda row: (row["date"], row["id"]), reverse=True)
    api_pool = [row for row in pool if row["source"] == "arXiv API"]
    cross_pool = [row for row in pool if row["source"] == "交叉补充"]
    monthly = Counter(int(row["date"][5:7]) for row in api_pool if row["date"][:4] == "2026")

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=19 * mm,
        title="2026 Agent Skills 论文全景图谱与 Deep Research 选型指南",
        author="Codex",
        subject="2026 Agent Skills landscape",
    )
    story = []

    story += [
        Spacer(1, 10 * mm),
        Paragraph("2026 Agent Skills 论文全景图谱", styles["title"]),
        Paragraph("领域分类、代表论文、原文链接与 Deep Research 项目选型", styles["subtitle"]),
        Spacer(1, 5 * mm),
    ]
    summary_table = Table(
        [
            [Paragraph("截止日期", styles["small"]), Paragraph("主报告精选", styles["small"]), Paragraph("宽检索＋交叉补充", styles["small"])],
            [Paragraph(CUTOFF, styles["h2"]), Paragraph(f"{len(unique_selected)} 篇", styles["h2"]), Paragraph(f"{len(pool)} 篇候选", styles["h2"])],
        ],
        colWidths=[54 * mm, 54 * mm, 54 * mm],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EFF2")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#BAC6D1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D3DAE2")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story += [
        summary_table,
        Spacer(1, 8 * mm),
        Paragraph(
            "范围说明：主清单只收录 2026 年、能够从 arXiv/OpenReview/作者维护索引核验、并与 Agent Skill 的表示、获取、检索、执行、进化或治理直接相关的代表工作。具身机器人、GUI 和科学工作流只保留与可复用 Agent Skill 明确相连的少量邻接论文。绝大多数为快速更新的预印本，不能把出现次数当作学术共识或成熟度。",
            styles["body"],
        ),
        Paragraph(
            '<b>主要来源：</b> '
            '<a href="https://export.arxiv.org/api/query">arXiv 官方 API</a> · '
            '<a href="https://arxiv.org/abs/2605.07358">A Comprehensive Survey on Agent Skills</a> · '
            '<a href="https://arxiv.org/abs/2602.12430">Agent Skills for Large Language Models</a> · '
            '<a href="https://github.com/JayLZhou/Awesome-Agent-Skills">Awesome Agent Skills</a>',
            styles["small"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("如何阅读这份报告", styles["h2"]),
        Paragraph(
            f"正文中的“项目适配”是针对当前 Evidence-Driven Deep Research Agent 的工程判断：●越多，越适合直接影响现有架构。它不是论文质量或学术影响力评分。附录中的 {len(pool)} 条为宽检索与交叉补充候选，只做题录核验，不等价于逐篇推荐。",
            styles["body"],
        ),
        PageBreak(),
        Paragraph("一、技术生命周期图谱", styles["h1"]),
        LifecycleDiagram(170 * mm),
        Spacer(1, 4 * mm),
        Paragraph("2026 年的结构性变化", styles["h2"]),
        Paragraph(
            "研究重点已经从“给 Agent 一段更好的提示词”转向技能制品工程：Skill 需要明确触发条件、依赖资源、执行入口、质量信号、权限边界和版本生命周期。最活跃的三条线是自进化、规模化检索和安全治理；最缺乏共识的仍是 Skill 的统一表示、跨 Harness 可移植性，以及没有明确成功标签时的低样本验证。",
            styles["body"],
        ),
        Paragraph("arXiv 宽检索池的月度增长", styles["h2"]),
        MonthBars(monthly, 170 * mm),
        Paragraph(
            "注：图中统计的是官方 API 对 all:\"agent skill\" 的宽检索结果，不是严格意义的论文产量。它主要用来显示 2026 年中后期该主题的爆发式增长。",
            styles["small"],
        ),
        PageBreak(),
        Paragraph("二、按领域精选的代表论文", styles["h1"]),
    ]

    for category in CATEGORIES:
        story.append(Paragraph(category["title"], styles["h1"]))
        story.append(Paragraph(f'<b>核心问题：</b>{esc(category["question"])}', styles["body"]))
        for paper_id, takeaway, fit in category["papers"]:
            story.append(paper_card(entries[paper_id], takeaway, fit, styles))
            story.append(Spacer(1, 2.4 * mm))

    story += [
        PageBreak(),
        Paragraph("三、针对你的 Deep Research 项目的阅读与落地顺序", styles["h1"]),
        Paragraph(
            "你的约束是完整任务样本少、成败难二值化。因此最优路线不是大规模 Trace 蒸馏，而是：局部弱信号诊断 → Episode Memory → CandidatePatch → 历史任务成对回放 → Canary → Stable Skill。下面按工程先后排序。",
            styles["body"],
        ),
    ]
    priority_rows = [[Paragraph("优先级", styles["small"]), Paragraph("论文", styles["small"]), Paragraph("在项目中的作用", styles["small"])]]
    for paper_id, priority, reason in PRIORITY:
        entry = entries[paper_id]
        priority_rows.append(
            [
                Paragraph(priority, styles["paper_title"]),
                Paragraph(f'<a href="{entry["url"]}">{esc(entry["title"])}</a><br/><font size="7">arXiv:{paper_id}</font>', styles["paper_body"]),
                Paragraph(esc(reason), styles["paper_body"]),
            ]
        )
    priority_table = Table(priority_rows, colWidths=[18 * mm, 66 * mm, 86 * mm], repeatRows=1)
    priority_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDECF0")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C3CDD7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story += [priority_table, Spacer(1, 5 * mm)]
    for heading, text in [
        ("P0 最小闭环", "先增加 SkillSignal、EpisodeMemory、CandidatePatch、ReplayEvaluator 和 RejectedPatchBuffer。单次严重且可复现的确定性故障可以直接通过回归测试晋级；一般策略必须保留适用条件。"),
        ("P1 规模化", "当候选 Skill 数量增加后，再引入 query-conditioned routing、图关系检索、来源绑定、去重、Skill shadowing 监控与 Canary 发布。"),
        ("暂缓", "暂不优先采用依赖大规模 on-policy 训练、自博弈或大量高质量 verifier 数据的方法。它们适合作为后续研究分支，而不是当前工程主线。"),
    ]:
        story.append(Paragraph(heading, styles["h2"]))
        story.append(Paragraph(text, styles["body"]))

    story += [
        PageBreak(),
        Paragraph("四、阅读路线：从 12 篇开始", styles["h1"]),
    ]
    for index, (paper_id, priority, reason) in enumerate(PRIORITY, 1):
        entry = entries[paper_id]
        story.append(
            KeepTogether(
                [
                    Paragraph(f'{index}. <a href="{entry["url"]}">{esc(entry["title"])}</a>', styles["h2"]),
                    Paragraph(f'arXiv:{paper_id} · {reason}', styles["body"]),
                ]
            )
        )

    story += [
        PageBreak(),
        Paragraph("附录 A：题录宽检索与交叉补充候选池", styles["h1"]),
        Paragraph(
            f'检索式为 all:"agent skill"，按提交日期倒序抓取并限定 2026 年，获得 {len(api_pool)} 条；再从综述索引和项目相关检索中补充 {len(cross_pool)} 条，共 {len(pool)} 条。该附录用于尽可能减少漏检，包含部分只在摘要中提及 Agent Skill 的邻接论文，因此不应把每一条都视作核心 Skill 论文。点击标题可打开原文。',
            styles["body"],
        ),
    ]
    appendix_rows = [[Paragraph("日期", styles["small"]), Paragraph("arXiv", styles["small"]), Paragraph("来源", styles["small"]), Paragraph("标题", styles["small"])]]
    for row in pool:
        appendix_rows.append(
            [
                Paragraph(row["date"], styles["small"]),
                Paragraph(row["id"], styles["small"]),
                Paragraph(row["source"], styles["small"]),
                Paragraph(f'<a href="{row["url"]}">{esc(row["title"])}</a>', styles["small"]),
            ]
        )
    appendix_table = Table(appendix_rows, colWidths=[21 * mm, 23 * mm, 19 * mm, 107 * mm], repeatRows=1)
    appendix_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3EDF0")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D4DBE3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(appendix_table)
    story += [
        PageBreak(),
        Paragraph("附录 B：检索方法、边界与更新建议", styles["h1"]),
        Paragraph("检索方法", styles["h2"]),
        Paragraph(
            "使用 arXiv 官方 Atom API 获取 all:\"agent skill\" 的前两页结果，覆盖 2026-01-01 至 2026-08-27；再与两份 2026 Agent Skills 综述、作者维护的 Awesome Agent Skills 索引及已知 Deep Research 邻接论文交叉核验。主报告由人工按标题和摘要筛选，附录保留宽检索池。",
            styles["body"],
        ),
        Paragraph("边界", styles["h2"]),
        Paragraph(
            "这不是系统综述意义上的 PRISMA 检索，也没有覆盖所有非英文数据库、闭源技术报告和 2026 年尚未公开的会议录。arXiv 论文可能改题、撤回或更新实验。领域中大量结果发表于 2026 年 6-8 月，引用数和复现实证尚不足，因此正文以技术代表性和项目适配度筛选，而不按引用排名。",
            styles["body"],
        ),
        Paragraph("建议更新频率", styles["h2"]),
        Paragraph(
            "如果用于项目技术选型，建议每月增量检索一次；新论文先进入 Candidate Reading Pool，只有出现公开代码、明确 benchmark、可复现实验或与项目当前问题直接对应时，才提升为核心阅读。",
            styles["body"],
        ),
    ]

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
