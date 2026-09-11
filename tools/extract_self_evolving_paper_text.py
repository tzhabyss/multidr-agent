from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "tmp" / "pdfs" / "self-evolving-deepresearch"
PAPER_IDS = (
    "2605.23904",
    "2512.23760",
    "2508.06433",
    "2608.01913",
    "2604.01687",
    "2607.15257",
    "2607.07676",
    "2606.03024",
)


def main() -> None:
    text_dir = PAPER_ROOT / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for paper_id in PAPER_IDS:
        pdf_path = PAPER_ROOT / f"{paper_id}.pdf"
        reader = PdfReader(str(pdf_path))
        page_texts: list[str] = []
        for page_number, page in enumerate(reader.pages, 1):
            page_texts.append(
                f"\n\n===== PAGE {page_number} =====\n\n"
                + (page.extract_text() or "")
            )
        (text_dir / f"{paper_id}.txt").write_text(
            "".join(page_texts), encoding="utf-8"
        )
        metadata = {
            str(key).lstrip("/"): str(value)
            for key, value in (reader.metadata or {}).items()
        }
        manifest.append(
            {
                "paper_id": paper_id,
                "pages": len(reader.pages),
                "metadata": metadata,
                "pdf": str(pdf_path),
            }
        )
    (text_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
