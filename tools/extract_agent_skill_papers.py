from __future__ import annotations

import json
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "tmp" / "pdfs" / "agent-skills-papers"
TEXT_DIR = SOURCE_DIR / "text"


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf")):
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text(x_tolerance=2, y_tolerance=3) or "").replace("\x00", "")
                pages.append(f"\n\n===== PAGE {index} =====\n\n{text}")
            page_count = len(pdf.pages)
            metadata = dict(pdf.metadata or {})
        text_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        text_path.write_text("".join(pages), encoding="utf-8")
        manifest.append(
            {
                "paper_id": pdf_path.stem,
                "pdf": str(pdf_path.relative_to(ROOT)),
                "text": str(text_path.relative_to(ROOT)),
                "pages": page_count,
                "metadata": metadata,
            }
        )
    (TEXT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
