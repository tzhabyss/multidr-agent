from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
NOTE_DIR = ROOT / "output" / "notes"
OUTPUT_DIR = ROOT / "output" / "pdf" / "chapter-notes"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")

DOCUMENTS = [
    ("SkillX", NOTE_DIR / "SkillX-飞书章节阅读笔记.md", OUTPUT_DIR / "SkillX-逐章解析笔记.pdf"),
    ("CoEvoSkills", NOTE_DIR / "CoEvoSkills-飞书章节阅读笔记.md", OUTPUT_DIR / "CoEvoSkills-逐章解析笔记.pdf"),
    ("SkillOpt", NOTE_DIR / "SkillOpt-飞书章节阅读笔记.md", OUTPUT_DIR / "SkillOpt-逐章解析笔记.pdf"),
]


def register_fonts() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    pdfmetrics.registerFont(TTFont("NoteSans", str(FONT_PATH)))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName="NoteSans", fontSize=9.2,
            leading=14.2, textColor=colors.HexColor("#1F2937"), spaceAfter=4.2,
            wordWrap="CJK", allowWidows=0, allowOrphans=0,
        ),
        "h1": ParagraphStyle(
            "H1CN", parent=base["Heading1"], fontName="NoteSans", fontSize=20,
            leading=27, textColor=colors.HexColor("#123B5D"), spaceBefore=11,
            spaceAfter=10, keepWithNext=True, wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2CN", parent=base["Heading2"], fontName="NoteSans", fontSize=15,
            leading=21, textColor=colors.HexColor("#0F6B78"), spaceBefore=10,
            spaceAfter=6, keepWithNext=True, wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "H3CN", parent=base["Heading3"], fontName="NoteSans", fontSize=11.5,
            leading=17, textColor=colors.HexColor("#B45309"), spaceBefore=7,
            spaceAfter=4, keepWithNext=True, wordWrap="CJK",
        ),
        "h4": ParagraphStyle(
            "H4CN", parent=base["Heading4"], fontName="NoteSans", fontSize=10,
            leading=15, textColor=colors.HexColor("#374151"), spaceBefore=5,
            spaceAfter=3, keepWithNext=True, wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "QuoteCN", parent=base["BodyText"], fontName="NoteSans", fontSize=8.8,
            leading=13.8, textColor=colors.HexColor("#385466"), backColor=colors.HexColor("#EDF6F8"),
            borderColor=colors.HexColor("#55A7B4"), borderWidth=0.8, borderPadding=6,
            leftIndent=8, rightIndent=3, spaceBefore=3, spaceAfter=6, wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletCN", parent=base["BodyText"], fontName="NoteSans", fontSize=9.1,
            leading=14, textColor=colors.HexColor("#1F2937"), leftIndent=14,
            firstLineIndent=-8, spaceAfter=2.5, wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeCN", parent=base["Code"], fontName="NoteSans", fontSize=7.8,
            leading=11.5, textColor=colors.HexColor("#263238"), backColor=colors.HexColor("#F3F4F6"),
            borderColor=colors.HexColor("#D1D5DB"), borderWidth=0.5, borderPadding=6,
            leftIndent=4, rightIndent=4, spaceBefore=3, spaceAfter=6, wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN", parent=base["BodyText"], fontName="NoteSans", fontSize=7.5,
            leading=10.5, textColor=colors.white, alignment=TA_LEFT, wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "TableCellCN", parent=base["BodyText"], fontName="NoteSans", fontSize=7.2,
            leading=10.4, textColor=colors.HexColor("#24313D"), wordWrap="CJK",
        ),
        "cover_title": ParagraphStyle(
            "CoverTitleCN", parent=base["Title"], fontName="NoteSans", fontSize=25,
            leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#123B5D"),
            spaceAfter=12, wordWrap="CJK",
        ),
        "cover_sub": ParagraphStyle(
            "CoverSubCN", parent=base["BodyText"], fontName="NoteSans", fontSize=11,
            leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#607584"),
            wordWrap="CJK",
        ),
    }


def inline_markup(text: str) -> str:
    value = html.escape(text.strip(), quote=False)
    value = re.sub(r"`([^`]+)`", r'<font color="#9A3412">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = value.replace("  ", " &#160;")
    return value


def table_rows(lines: list[str], styles: dict[str, ParagraphStyle]) -> list[list[Paragraph]]:
    parsed: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        parsed.append(cells)
    if not parsed:
        return []
    width = max(len(row) for row in parsed)
    for row in parsed:
        row.extend([""] * (width - len(row)))
    return [
        [Paragraph(inline_markup(cell), styles["table_head"] if i == 0 else styles["table_cell"]) for cell in row]
        for i, row in enumerate(parsed)
    ]


def make_table(lines: list[str], styles: dict[str, ParagraphStyle], available_width: float) -> Table | None:
    rows = table_rows(lines, styles)
    if not rows:
        return None
    column_count = len(rows[0])
    raw_rows = [[re.sub(r"[*`]", "", c.strip()) for c in line.strip().strip("|").split("|")] for line in lines if "---" not in line]
    scores: list[float] = []
    for col in range(column_count):
        max_len = max((len(row[col]) if col < len(row) else 0 for row in raw_rows), default=1)
        scores.append(max(7.0, min(28.0, max_len ** 0.72)))
    total = sum(scores)
    widths = [available_width * score / total for score in scores]
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B78")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BAC7CC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8F9")]),
    ]))
    return table


def markdown_to_story(text: str, styles: dict[str, ParagraphStyle], available_width: float) -> list:
    lines = text.splitlines()
    story: list = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(piece.strip() for piece in paragraph if piece.strip())
            if joined:
                story.append(Paragraph(inline_markup(joined), styles["body"]))
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                code_text = "<br/>".join(html.escape(part, quote=False).replace(" ", "&#160;") for part in code)
                story.append(Paragraph(code_text or "&#160;", styles["code"]))
                code.clear()
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                block.append(lines[i])
                i += 1
            tbl = make_table(block, styles, available_width)
            if tbl:
                story.extend([tbl, Spacer(1, 6)])
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            style = styles[{1: "h1", 2: "h2", 3: "h3"}.get(level, "h4")]
            story.append(Paragraph(inline_markup(heading.group(2)), style))
            i += 1
            continue
        if re.fullmatch(r"-{3,}|_{3,}|\*{3,}", stripped):
            flush_paragraph()
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#B5C4CA"), spaceBefore=4, spaceAfter=7))
            i += 1
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            quote_parts: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_parts.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline_markup(" ".join(quote_parts)), styles["quote"]))
            continue
        bullet = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        numbered = re.match(r"^(\s*)\d+[.)]\s+(.*)$", line)
        if bullet or numbered:
            flush_paragraph()
            match = bullet or numbered
            indent = min(18, len(match.group(1)) * 2)
            marker = "•" if bullet else "○"
            local = ParagraphStyle("NestedBullet", parent=styles["bullet"], leftIndent=14 + indent)
            story.append(Paragraph(f"{marker}&#160;&#160;{inline_markup(match.group(2))}", local))
            i += 1
            continue
        if not stripped:
            flush_paragraph()
            story.append(Spacer(1, 2.5))
            i += 1
            continue
        paragraph.append(stripped)
        i += 1
    flush_paragraph()
    if in_code and code:
        code_text = "<br/>".join(html.escape(part, quote=False).replace(" ", "&#160;") for part in code)
        story.append(Paragraph(code_text, styles["code"]))
    return story


def page_decor(canvas: Canvas, doc: SimpleDocTemplate, short_name: str) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D7E1E5"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("NoteSans", 7.5)
    canvas.setFillColor(colors.HexColor("#71818B"))
    canvas.drawString(18 * mm, height - 11 * mm, f"{short_name} · 逐章节解析笔记")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_document(short_name: str, source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(destination), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=19 * mm, bottomMargin=17 * mm, title=f"{short_name} 论文逐章节解析笔记",
        author="Deep Research Agent",
    )
    story = [
        Spacer(1, 38 * mm),
        Paragraph(f"{short_name}<br/>论文逐章节解析笔记", styles["cover_title"]),
        HRFlowable(width="42%", thickness=1.2, color=colors.HexColor("#2C8C99"), spaceBefore=6, spaceAfter=16),
        Paragraph("飞书笔记版 · 保留论文结构、实验数据、局限性与项目启示", styles["cover_sub"]),
        Spacer(1, 86 * mm),
        Paragraph("由原始 Markdown 逐章笔记排版生成", styles["cover_sub"]),
        PageBreak(),
    ]
    story.extend(markdown_to_story(text, styles, A4[0] - 36 * mm))
    callback = lambda canvas, document: page_decor(canvas, document, short_name)
    doc.build(story, onFirstPage=callback, onLaterPages=callback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the three Feishu-ready chapter notes as PDFs.")
    parser.add_argument("--only", choices=[item[0] for item in DOCUMENTS], help="Render only one document")
    args = parser.parse_args()
    register_fonts()
    selected = [item for item in DOCUMENTS if not args.only or item[0] == args.only]
    for short_name, source, destination in selected:
        if not source.exists():
            raise FileNotFoundError(source)
        build_document(short_name, source, destination)
        print(destination)


if __name__ == "__main__":
    main()
