"""Batch colorization: process a whole folder (e.g. one chapter).

Reads every supported image in the input directory (sorted naturally so
chapter pages keep their order) and writes a colorized copy into the output
directory, preserving the relative path.

The colorization engine is selected via ``settings["engine"]``:
    "comicnet" -> ColorComicNet (recommended for CPU / manga)
    "zhang"    -> Zhang et al. ECCV16 (photo model, +optional enhancement)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cancel as _cancel
from colorize import Colorizer, load_settings

# Regex to sort names like "page_2.png" before "page_10.png".
_NUM_RE = re.compile(r"(\d+)")


@dataclass
class ArchiveResult:
    """Result for one colorized comic archive (.cbz/.zip).

    ``pages_dir`` is the isolated folder holding the colorized page images,
    ``produced`` is the list of those image files (natural order). The CLI uses
    these to build the per-archive CBZ/PDF in the output root.
    """

    archive: Path
    pages_dir: Path
    produced: list[Path] = field(default_factory=list)


def _natural_key(path: Path) -> list:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in _NUM_RE.split(path.name)
    ]


def collect_images(directory: Path, exts: Iterable[str]) -> list[Path]:
    exts = {e.lower() for e in exts}
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(files, key=_natural_key)


def collect_archives(directory: Path | str) -> list[Path]:
    """Find comic archives (.cbz/.zip) directly under ``directory``."""
    from cbz import CBZ_EXTS

    directory = Path(directory)
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in CBZ_EXTS
    ]
    return sorted(files, key=_natural_key)


def build_engine(settings: dict):
    """Instantiate the colorizer selected by ``settings['engine']``."""
    engine = str(settings.get("engine", "mangacolv2")).lower()
    if engine == "mangacolv2":
        from mangacolv2_engine import MangaColV2Colorizer

        return MangaColV2Colorizer(settings)
    if engine == "comicnet":
        from comicnet_engine import ComicNetColorizer

        return ComicNetColorizer(settings)
    if engine == "zhang":
        return Colorizer(settings)
    raise ValueError(
        f"Unknown engine: {engine!r} (use 'mangacolv2', 'comicnet' or 'zhang')"
    )


def colorize_folder(
    input_dir: Path | str,
    output_dir: Path | str,
    colorizer=None,
    on_progress: Callable[[int, int, Path], None] | None = None,
    upscale: bool | None = None,
) -> tuple[int, int, list[Path]]:
    """Colorize every image in ``input_dir`` into ``output_dir``.

    If ``upscale`` (defaults to ``settings['upscale']``) is true, each colorized
    page is also run through the Real-ESRGAN upscaler.

    Returns ``(success, failed, output_files)`` where ``output_files`` lists the
    colorized files that were produced (in natural order), so callers can pack
    exactly those into a CBZ/PDF.
    """
    settings = (getattr(colorizer, "settings", None) or load_settings())
    exts = settings.get("supported_ext", [".png", ".jpg", ".jpeg"])
    suffix = settings.get("output_suffix", "_colored")

    do_upscale = settings.get("upscale", False) if upscale is None else upscale

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = collect_images(input_dir, exts)
    if not files:
        print(f"[batch] no images found in {input_dir}")
        return 0, 0, []

    if colorizer is None:
        colorizer = build_engine(settings)

    upscaler = None
    if do_upscale:
        from upscale import Upscaler

        upscaler = Upscaler(settings)
        print(f"[batch] upscaling enabled (x{upscaler.factor})")

    ok = failed = 0
    produced: list[Path] = []
    total = len(files)
    cancelled = False
    for idx, src in enumerate(files, 1):
        if _cancel.is_cancelled():
            cancelled = True
            print("[batch] cancelled by user.")
            break
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_name(f"{rel.stem}{suffix}{rel.suffix}")
        try:
            colorizer.colorize_file(src, dst)
            if upscaler is not None:
                upscaler.upscale_file(dst, dst, factor=upscaler.factor)
            ok += 1
            produced.append(dst)
            status = "OK"
        except Exception as exc:  # noqa: BLE001
            failed += 1
            status = f"FAIL ({exc})"

        print(f"[batch] ({idx}/{total}) {rel}  ->  {status}")
        if on_progress:
            on_progress(idx, total, src)

    if cancelled:
        print(f"[batch] stopped early. success={ok} failed={failed}")
    else:
        print(f"[batch] done. success={ok} failed={failed}")
    return ok, failed, produced


def colorize_archive(
    archive: Path | str,
    output_dir: Path | str,
    colorizer=None,
    upscale: bool | None = None,
    keep_pages: bool = False,
    repack_cbz: bool = True,
    pages_dir: Path | str | None = None,
) -> tuple[int, int, ArchiveResult]:
    """Colorize a single CBZ, writing its pages into their own subfolder.

    Steps: extract pages into a temporary folder -> colorize them into
    ``pages_dir`` (an isolated per-archive folder so pages from different
    archives never collide) -> optionally pack them into
    ``<name>_colorized.cbz`` next to the other outputs.

    The extracted B&W pages live in a temporary directory so they never clutter
    ``output/``; it is always removed when done (set ``keep_pages=True`` to keep
    it, e.g. for debugging).

    ``pages_dir`` defaults to ``output_dir/<archive stem>/`` — the colorized
    pages of each archive go there, keeping multi-archive runs isolated.

    Returns ``(success, failed, ArchiveResult)`` where the result carries the
    archive, its pages folder and the colorized page files.
    """
    import shutil
    import tempfile

    from cbz import extract_cbz

    settings = (getattr(colorizer, "settings", None) or load_settings())
    archive = Path(archive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if pages_dir is None:
        pages_dir = output_dir / archive.stem
    pages_dir = Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Extract into a temp dir so the intermediate B&W pages don't pile up in
    # output/ (e.g. output/<name>_pages/). Removed at the end regardless.
    if keep_pages:
        pages_in = output_dir / f"{archive.stem}_pages"
        pages_in.mkdir(parents=True, exist_ok=True)
        remove_pages = False
    else:
        # System temp dir: never touches output/, so the user only sees results.
        pages_in = Path(tempfile.mkdtemp(prefix=f"{archive.stem}_pages_"))
        remove_pages = True

    try:
        extract_cbz(archive, pages_in)

        ok, failed, produced = colorize_folder(
            pages_in, pages_dir, colorizer=colorizer, upscale=upscale
        )

        if produced and repack_cbz:
            from package import make_cbz

            out_cbz = output_dir / f"{archive.stem}_colorized.cbz"
            make_cbz(pages_dir, out_cbz, files=produced, title=archive.stem)
    finally:
        if remove_pages:
            shutil.rmtree(pages_in, ignore_errors=True)

    return ok, failed, ArchiveResult(archive=archive, pages_dir=pages_dir,
                                     produced=produced)


def colorize_input(
    input_dir: Path | str,
    output_dir: Path | str,
    colorizer=None,
    upscale: bool | None = None,
    repack_cbz: bool = True,
) -> tuple[int, int, list[ArchiveResult], list[Path]]:
    """Colorize everything found in ``input_dir``.

    Handles **both** loose images and comic archives (.cbz/.zip).

    - Each archive is colorized into its **own** subfolder under ``output_dir``
      (``<name>/``), so pages from different archives never collide. When
      ``repack_cbz`` is True each archive also gets ``<name>_colorized.cbz``.
    - Loose images are colorized flat into ``output_dir``.

    Returns ``(success, failed, archive_results, loose_files)`` so the caller can
    build the right per-archive (and/or combined) CBZ/PDF outputs.
    """
    settings = (getattr(colorizer, "settings", None) or load_settings())
    if colorizer is None:
        colorizer = build_engine(settings)

    total_ok = total_failed = 0
    archive_results: list[ArchiveResult] = []
    loose_files: list[Path] = []

    archives = collect_archives(input_dir)
    exts = settings.get("supported_ext", [".png", ".jpg", ".jpeg"])
    images = collect_images(Path(input_dir), exts)

    if not archives and not images:
        print(f"[batch] nothing to do in {input_dir} "
              "(put images or a .cbz there)")
        return 0, 0, [], []

    # 1) Comic archives — each into its own output subfolder.
    for arc in archives:
        if _cancel.is_cancelled():
            print("[batch] cancelled by user.")
            return total_ok, total_failed, archive_results, loose_files
        print(f"[batch] archive: {arc.name}")
        ok, failed, result = colorize_archive(
            arc, output_dir, colorizer=colorizer, upscale=upscale,
            repack_cbz=repack_cbz,
        )
        total_ok += ok
        total_failed += failed
        archive_results.append(result)

    # 2) Loose images — flat into the output folder (combined later).
    if images and not _cancel.is_cancelled():
        ok, failed, produced = colorize_folder(
            input_dir, output_dir, colorizer=colorizer, upscale=upscale
        )
        total_ok += ok
        total_failed += failed
        loose_files.extend(produced)

    return total_ok, total_failed, archive_results, loose_files


if __name__ == "__main__":
    import sys

    settings = load_settings()
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(settings["input_dir"])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(settings["output_dir"])
    colorize_folder(in_dir, out_dir)