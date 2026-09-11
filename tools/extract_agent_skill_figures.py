from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "tmp" / "pdfs" / "agent-skills-papers"
OUT_ROOT = ROOT / "docs" / "paper-reading" / "agent-skills" / "individual"
RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDFTOPPM = Path(
    r"C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe"
)


def safe_members(archive: tarfile.TarFile, destination: Path):
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root == target or root in target.parents:
            yield member


def convert_raster(source: Path, destination: Path) -> bool:
    try:
        with Image.open(source) as image:
            width, height = image.size
            if width < 200 and height < 200:
                return False
            image.convert("RGB").save(destination, "PNG", optimize=True)
        return True
    except Exception:
        return False


def convert_pdf(source: Path, destination: Path) -> bool:
    try:
        prefix = destination.with_suffix("")
        subprocess.run(
            [
                str(PDFTOPPM),
                "-f",
                "1",
                "-singlefile",
                "-r",
                "180",
                "-png",
                str(source),
                str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return destination.exists()
    except Exception:
        return False


def main() -> None:
    for archive_path in sorted(SOURCE_DIR.glob("*.tar")):
        paper_id = archive_path.stem
        extraction_dir = SOURCE_DIR / "source" / paper_id
        images_dir = OUT_ROOT / paper_id / "images"
        extraction_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive_path, "r:*") as archive:
                archive.extractall(extraction_dir, members=safe_members(archive, extraction_dir))
        except tarfile.TarError:
            continue

        records = []
        counter = 0
        candidates = [
            path
            for path in extraction_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in RASTER_EXTENSIONS.union({".pdf"})
            and not any(part.lower().startswith("logo") for part in path.parts)
        ]
        candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
        for source in candidates:
            counter += 1
            relative = source.relative_to(extraction_dir)
            safe_stem = "_".join(relative.with_suffix("").parts)[-100:]
            destination = images_dir / f"{counter:03d}_{safe_stem}.png"
            converted = (
                convert_pdf(source, destination)
                if source.suffix.lower() == ".pdf"
                else convert_raster(source, destination)
            )
            if not converted:
                counter -= 1
                continue
            records.append(
                {
                    "file": destination.name,
                    "source": str(relative),
                    "bytes": destination.stat().st_size,
                }
            )

        lines = [f"# {paper_id} 图片索引", "", f"总计：{len(records)} 张", ""]
        for record in records:
            lines.extend(
                [
                    f"- 文件：`{record['file']}`",
                    f"  - 来源：`{record['source']}`",
                    f"  - 大小：{record['bytes'] / 1024:.1f} KB",
                ]
            )
        (images_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
