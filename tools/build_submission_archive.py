"""Create a clean source archive using the repository's Git ignore rules."""

from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "submission" / "deepresearch-source.zip"
FORBIDDEN_PARTS = {".git", ".runtime", ".test-runtime", "artifacts", ".venv"}
FORBIDDEN_SUFFIXES = {".sqlite", ".db"}


def candidate_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = [Path(item) for item in result.stdout.decode("utf-8").split("\0") if item]
    selected: list[Path] = []
    for relative in paths:
        if FORBIDDEN_PARTS.intersection(relative.parts):
            raise RuntimeError(f"forbidden path entered submission: {relative}")
        if relative.name == ".env" or relative.suffix.casefold() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"forbidden credential/runtime file: {relative}")
        absolute = ROOT / relative
        if absolute.is_file():
            selected.append(relative)
    return sorted(selected, key=lambda item: item.as_posix())


def build() -> Path:
    files = candidate_files()
    if not files:
        raise RuntimeError("submission file list is empty")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    checksums: list[str] = []
    with zipfile.ZipFile(
        OUTPUT,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in files:
            content = (ROOT / relative).read_bytes()
            name = relative.as_posix()
            archive.writestr(name, content)
            checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
        archive.writestr("SUBMISSION_SHA256SUMS.txt", "\n".join(checksums) + "\n")
    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"archive CRC validation failed: {bad}")
    return OUTPUT


if __name__ == "__main__":
    print(build())
