from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RENDERED = ROOT / "tmp" / "pdfs" / "self-evolving-deepresearch" / "rendered"
OUT = ROOT / "tmp" / "pdfs" / "self-evolving-deepresearch" / "contact-sheets"


def page_number(path: Path) -> int:
    return int(path.stem.rsplit("-", 1)[-1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
    for folder in sorted(p for p in RENDERED.iterdir() if p.is_dir()):
        pages = sorted(folder.glob("page-*.png"), key=page_number)
        thumb_w = 330
        label_h = 34
        gap = 18
        cols = 4
        thumbs = []
        for page in pages:
            image = Image.open(page).convert("RGB")
            height = round(image.height * thumb_w / image.width)
            image.thumbnail((thumb_w, height), Image.Resampling.LANCZOS)
            thumbs.append((page, image.copy()))
        thumb_h = max(image.height for _, image in thumbs)
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new(
            "RGB",
            (gap + cols * (thumb_w + gap), gap + rows * (thumb_h + label_h + gap)),
            "#d9dde3",
        )
        draw = ImageDraw.Draw(sheet)
        for idx, (page, image) in enumerate(thumbs):
            row, col = divmod(idx, cols)
            x = gap + col * (thumb_w + gap)
            y = gap + row * (thumb_h + label_h + gap)
            draw.rectangle((x - 1, y - 1, x + thumb_w, y + thumb_h), fill="white", outline="#98a1ad")
            sheet.paste(image, (x, y))
            draw.text((x, y + thumb_h + 5), f"第 {page_number(page)} 页", font=font, fill="#172033")
        output = OUT / f"{folder.name}_contact.png"
        sheet.save(output, optimize=True)
        print(output)


if __name__ == "__main__":
    main()
