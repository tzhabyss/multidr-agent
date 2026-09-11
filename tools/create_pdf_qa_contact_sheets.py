from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RENDER_ROOT = ROOT / "tmp" / "pdfs" / "agent-skills-rendered"


def main() -> None:
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)
    for report_dir in sorted(path for path in RENDER_ROOT.iterdir() if path.is_dir()):
        pages = sorted(report_dir.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        if not pages:
            continue
        columns = 4
        thumb_w, thumb_h = 220, 311
        cell_w, cell_h = 240, 345
        rows = (len(pages) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "#D9E2E6")
        draw = ImageDraw.Draw(sheet)
        for index, page_path in enumerate(pages):
            with Image.open(page_path) as page:
                thumb = page.convert("RGB")
                thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            col, row = index % columns, index // columns
            x = col * cell_w + (cell_w - thumb.width) // 2
            y = row * cell_h + 5
            sheet.paste(thumb, (x, y))
            label = f"Page {index + 1}"
            draw.text((col * cell_w + 8, row * cell_h + 318), label, font=font, fill="#12324A")
        sheet.save(RENDER_ROOT / f"{report_dir.name}_contact.png", optimize=True)


if __name__ == "__main__":
    main()
