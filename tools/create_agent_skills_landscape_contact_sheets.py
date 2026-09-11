from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "pdfs" / "2026-agent-skills-landscape-rendered"
DEST = ROOT / "tmp" / "pdfs" / "2026-agent-skills-landscape-contact-sheets"


def page_no(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
    pages = sorted(SOURCE.glob("page-*.png"), key=page_no)
    for batch_index in range(0, len(pages), 8):
        batch = pages[batch_index : batch_index + 8]
        thumbs = []
        for path in batch:
            image = Image.open(path).convert("RGB")
            image.thumbnail((340, 480), Image.Resampling.LANCZOS)
            thumbs.append((path, image.copy()))
        cols, gap, label_h = 4, 18, 34
        thumb_w = 340
        thumb_h = max(image.height for _, image in thumbs)
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (gap + cols * (thumb_w + gap), gap + rows * (thumb_h + label_h + gap)), "#d9dde3")
        draw = ImageDraw.Draw(sheet)
        for index, (path, image) in enumerate(thumbs):
            row, col = divmod(index, cols)
            x = gap + col * (thumb_w + gap)
            y = gap + row * (thumb_h + label_h + gap)
            draw.rectangle((x - 1, y - 1, x + thumb_w, y + thumb_h), fill="white", outline="#98a1ad")
            sheet.paste(image, (x, y))
            draw.text((x, y + thumb_h + 5), f"第 {page_no(path)} 页", font=font, fill="#172033")
        output = DEST / f"contact-{batch_index // 8 + 1}.png"
        sheet.save(output, optimize=True)
        print(output)


if __name__ == "__main__":
    main()
