"""Build the six-page submission technical report from its Markdown source."""

from __future__ import annotations

import html
import re
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "submission" / "TECHNICAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "deepresearch-technical-report.pdf"

INK = colors.HexColor("#162033")
MUTED = colors.HexColor("#536079")
ACCENT = colors.HexColor("#126E82")
ACCENT_LIGHT = colors.HexColor("#E8F3F5")
GRID = colors.HexColor("#C9D3DD")
CODE_BG = colors.HexColor("#F3F6F8")


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    fallback = Path(r"C:\Windows\Fonts\simhei.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSansBold", str(bold)))
        return "ReportSans", "ReportSansBold"
    if fallback.exists():
        pdfmetrics.registerFont(TTFont("ReportSans", str(fallback)))
        return "ReportSans", "ReportSans"
    raise FileNotFoundError("No Chinese font found for PDF generation")


REGULAR, BOLD = register_fonts()


def inline_markup(value: str) -> str:
    escaped = html.escape(value.strip())
    escaped = re.sub(
        r"`([^`]+)`",
        rf'<font name="{BOLD}" color="#126E82">\1</font>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", rf'<font name="{BOLD}">\1</font>', escaped)
    return escaped


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=BOLD,
            fontSize=21,
            leading=28,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=7 * mm,
        ),
        "meta": ParagraphStyle(
            "ReportMeta",
            parent=base["Normal"],
            fontName=REGULAR,
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=7 * mm,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=base["Heading1"],
            fontName=BOLD,
            fontSize=14.5,
            leading=19,
            textColor=INK,
            spaceBefore=1.5 * mm,
            spaceAfter=3.5 * mm,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontName=BOLD,
            fontSize=11.5,
            leading=15.5,
            textColor=ACCENT,
            spaceBefore=2.5 * mm,
            spaceAfter=2 * mm,
        ),
        "h3": ParagraphStyle(
            "ReportH3",
            parent=base["Heading3"],
            fontName=BOLD,
            fontSize=9.8,
            leading=13.2,
            textColor=INK,
            spaceBefore=2 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=8.35,
            leading=12.2,
            textColor=INK,
            wordWrap="CJK",
            alignment=TA_LEFT,
            spaceAfter=1.8 * mm,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=8.25,
            leading=11.8,
            leftIndent=4 * mm,
            firstLineIndent=-3.2 * mm,
            textColor=INK,
            wordWrap="CJK",
            spaceAfter=1.1 * mm,
        ),
        "table": ParagraphStyle(
            "ReportTable",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=7.25,
            leading=9.6,
            textColor=INK,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "ReportTableHeader",
            parent=base["BodyText"],
            fontName=BOLD,
            fontSize=7.25,
            leading=9.6,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=base["Code"],
            fontName=REGULAR,
            fontSize=7.2,
            leading=9.4,
            textColor=INK,
            leftIndent=2.5 * mm,
            rightIndent=2.5 * mm,
            borderPadding=4,
            backColor=CODE_BG,
            borderColor=GRID,
            borderWidth=0.4,
            borderRadius=2,
            spaceBefore=1 * mm,
            spaceAfter=2.2 * mm,
        ),
    }


def table_flowable(rows: list[list[str]], style_map: dict[str, ParagraphStyle]) -> Table:
    column_count = len(rows[0])
    available = A4[0] - 30 * mm
    if column_count == 3:
        widths = [available * 0.21, available * 0.35, available * 0.44]
    elif column_count == 2:
        widths = [available * 0.34, available * 0.66]
    else:
        widths = [available / column_count] * column_count
    rendered: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        cell_style = style_map["table_header"] if row_index == 0 else style_map["table"]
        rendered.append([Paragraph(inline_markup(cell), cell_style) for cell in row])
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ACCENT_LIGHT]),
            ]
        )
    )
    return table


def markdown_story(markdown: str) -> list[object]:
    style_map = styles()
    lines = markdown.splitlines()
    story: list[object] = []
    paragraph_buffer: list[str] = []
    index = 0
    first_heading = True

    def flush_paragraph() -> None:
        if paragraph_buffer:
            story.append(
                Paragraph(inline_markup(" ".join(paragraph_buffer)), style_map["body"])
            )
            paragraph_buffer.clear()

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if stripped == "<!-- PAGEBREAK -->":
            flush_paragraph()
            story.append(PageBreak())
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code_lines), style_map["code"], maxLineLength=110))
            index += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            parsed = [
                [cell.strip() for cell in line.strip("|").split("|")]
                for line in table_lines
            ]
            rows = [
                row
                for row in parsed
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)
            ]
            story.append(table_flowable(rows, style_map))
            story.append(Spacer(1, 2.2 * mm))
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            text = stripped[2:]
            if first_heading:
                story.append(Spacer(1, 13 * mm))
                story.append(Paragraph(inline_markup(text), style_map["title"]))
                first_heading = False
            else:
                story.append(Paragraph(inline_markup(text), style_map["h1"]))
            index += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            style_name = "h1" if re.match(r"## \d+\.", stripped) else "h2"
            story.append(Paragraph(inline_markup(stripped[3:]), style_map[style_name]))
            index += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), style_map["h3"]))
            index += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            story.append(
                Paragraph("• " + inline_markup(stripped[2:]), style_map["bullet"])
            )
            index += 1
            continue
        if re.match(r"^\d+\.\s", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), style_map["bullet"]))
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if first_heading is False and stripped.startswith("版本："):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), style_map["meta"]))
            index += 1
            continue
        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    return story


def page_decor(canvas, document) -> None:  # noqa: ANN001
    page = canvas.getPageNumber()
    width, height = A4
    canvas.saveState()
    if page > 1:
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(0.7)
        canvas.line(15 * mm, height - 13 * mm, width - 15 * mm, height - 13 * mm)
        canvas.setFont(REGULAR, 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(15 * mm, height - 10.2 * mm, "Evidence-Driven Deep Research Agent")
    canvas.setFont(REGULAR, 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 15 * mm, 10 * mm, f"{page} / 6")
    canvas.restoreState()


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="Evidence-Driven Deep Research Agent 技术报告",
        author="Deep Research Agent Project",
        subject="系统架构、关键设计、评测、失败案例与改进路线",
    )
    document.build(
        markdown_story(SOURCE.read_text(encoding="utf-8")),
        onFirstPage=page_decor,
        onLaterPages=page_decor,
    )
    pages = len(PdfReader(str(OUTPUT)).pages)
    if pages > 6:
        raise RuntimeError(f"technical report exceeds six-page limit: {pages}")
    if pages != 6:
        raise RuntimeError(f"technical report must render as six controlled pages: {pages}")
    return OUTPUT


if __name__ == "__main__":
    print(build())
