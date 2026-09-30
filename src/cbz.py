"""Read comic archives (CBZ) and pack colorized pages back into a CBZ.

A **CBZ** is just a ZIP of page images in reading order. Reading is simple:
extract the image entries, drop non-image files (e.g. ``ComicInfo.xml``), and
sort pages naturally (``page_2`` before ``page_10``).

Supported inputs:
    .cbz  — standard comic archive (ZIP)
    .zip  — same format, treated identically
(CBR/RAR is NOT supported without extra tooling; tell the user to convert.)

Public API:
    extract_cbz(archive, dest_dir)  -> list[Path]   (page images, in order)
    is_cbz(path)                    -> bool
    pack_cbz(image_files, out_path, title="") -> Path   (re-use package.make_cbz)
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable

_NUM_RE = re.compile(r"(\d+)")

# Image extensions we accept inside an archive.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Archive extensions we treat as comic archives.
CBZ_EXTS = {".cbz", ".zip"}


def is_cbz(path: Path | str) -> bool:
    """True if ``path`` looks like a comic archive we can read."""
    return Path(path).suffix.lower() in CBZ_EXTS


def _natural_key(name: str) -> list:
    """Sort key so '2.png' comes before '10.png'."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in _NUM_RE.split(name)
    ]


def _image_entries(names: Iterable[str]) -> list[str]:
    out = []
    for n in names:
        if Path(n).suffix.lower() in IMAGE_EXTS and not Path(n).name.startswith("."):
            out.append(n)
    return sorted(out, key=_natural_key)


def list_cbz_pages(archive: Path | str) -> list[str]:
    """Return the image entry names inside a CBZ, in reading order."""
    archive = Path(archive)
    with zipfile.ZipFile(archive, "r") as zf:
        return _image_entries(zf.namelist())


def extract_cbz(archive: Path | str, dest_dir: Path | str) -> list[Path]:
    """Extract page images from ``archive`` into ``dest_dir``.

    Pages are written as ``{index:04d}{ext}`` so their on-disk order matches
    the archive order and stays stable. Returns the list of written files.
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with zipfile.ZipFile(archive, "r") as zf:
        pages = _image_entries(zf.namelist())
        if not pages:
            raise ValueError(f"No images found inside {archive.name}")
        for i, name in enumerate(pages, 1):
            ext = Path(name).suffix.lower()
            target = dest_dir / f"{i:04d}{ext}"
            with zf.open(name) as src, open(target, "wb") as out:
                out.write(src.read())
            written.append(target)
    print(f"[cbz] extracted {len(written)} pages from {archive.name}")
    return written


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python src/cbz.py <archive.cbz> <dest_dir>")
        raise SystemExit(2)
    files = extract_cbz(sys.argv[1], sys.argv[2])
    for f in files[:10]:
        print(" ", f.name)
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
