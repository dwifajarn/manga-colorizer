"""Package colorized pages into a single CBZ (comic archive) and/or PDF.

CBZ = a ZIP of images in reading order, opened by comic readers (Tachiyomi,
CDisplayEx, YACReader, and most e-readers).
PDF = built with Pillow, images placed one per page at full size.

Both keep the natural page order (page_2 before page_10).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(path: Path) -> list:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in _NUM_RE.split(path.name)
    ]


def _gather(directory: Path, exts: Iterable[str]) -> list[Path]:
    exts = {e.lower() for e in exts}
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=_natural_key)


def make_cbz(image_dir: Path | str, out_path: Path | str,
             exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".webp"),
             title: str = "", files: list[Path] | None = None) -> Path:
    """Zip the given images (or all images under ``image_dir``) into a .cbz."""
    image_dir = Path(image_dir)
    out_path = Path(out_path)
    if files is None:
        files = _gather(image_dir, exts)
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if title:
            zf.writestr("ComicInfo.xml", _comic_info(title, len(files)))
        for i, f in enumerate(files, 1):
            # Prefix with zero-padded index so any reader keeps the order.
            arcname = f"{i:04d}{f.suffix.lower()}"
            zf.write(f, arcname)
    print(f"[cbz] wrote {out_path} ({len(files)} pages)")
    return out_path


def make_pdf(image_dir: Path | str, out_path: Path | str,
             exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".webp"),
             files: list[Path] | None = None) -> Path:
    """Combine the given images (or all under ``image_dir``) into a single PDF."""
    from PIL import Image

    image_dir = Path(image_dir)
    out_path = Path(out_path)
    if files is None:
        files = _gather(image_dir, exts)
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in files:
        img = Image.open(f).convert("RGB")
        frames.append(img)
    frames[0].save(out_path, "PDF", save_all=True, append_images=frames[1:])
    for img in frames:
        img.close()
    print(f"[pdf] wrote {out_path} ({len(files)} pages)")
    return out_path


def _comic_info(title: str, page_count: int) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<ComicInfo>\n"
        f"  <Title>{title}</Title>\n"
        "  <LanguageISO>id</LanguageISO>\n"
        f"  <PageCount>{page_count}</PageCount>\n"
        "</ComicInfo>\n"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python src/package.py <image_dir> <out.{cbz|pdf}>")
        raise SystemExit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if dst.suffix.lower() == ".pdf":
        make_pdf(src, dst)
    else:
        make_cbz(src, dst)
