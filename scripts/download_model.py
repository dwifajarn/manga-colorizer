"""Download colorization & upscaling models into ``models/``.

Engines:
    mangacolv2 (default) — qweasdd manga-colorization-v2 GAN (generator + FFDNet
                           denoiser). Best quality/CPU trade-off for manga.
    comicnet             — ColorComicNet (HF Space 1plus1/MangaColorization).
    zhang                — Zhang et al. ECCV16 photo colorizer.
    upscale              — Real-ESRGAN x2plus ONNX (for the upscaling stage).

Usage:
    python scripts/download_model.py                    # everything
    python scripts/download_model.py --engine mangacolv2
    python scripts/download_model.py --engine upscale
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

SPECS = {
    "mangacolv2": [
        {
            "url": "https://huggingface.co/Kaiser41/manga-colorization-v2/resolve/main/generator.zip",
            "dest": MODELS / "mangacolv2" / "generator.zip",
            "sha256": None,
            "min_bytes": 50_000_000,
        },
        {
            "url": "https://huggingface.co/Kaiser41/manga-colorization-v2/resolve/main/net_rgb.pth",
            "dest": MODELS / "mangacolv2" / "net_rgb.pth",
            "sha256": None,
            "min_bytes": 1_000_000,
        },
    ],
    "comicnet": [
        {
            "url": "https://huggingface.co/spaces/1plus1/MangaColorization/resolve/main/weights/colorizer.pth",
            "dest": MODELS / "comicnet" / "colorizer.pth",
            "sha256": None,
            "min_bytes": 10_000_000,
        },
    ],
    "zhang": [
        {
            "url": "https://huggingface.co/ckpt/colorization/resolve/main/colorization_release_v2-9b330a0b.pth",
            "dest": MODELS / "colorization_release_v2.pth",
            "sha256": "9B330A0BAE53F4DED77B1E23DEFBF78BEAA09C10EBC4C4999E8E4F4A160B93F9",
            "min_bytes": 50_000_000,
        },
    ],
    "upscale": [
        {
            "url": "https://huggingface.co/tamnvcc/RealESRGAN-onnx/resolve/main/onnx/RealESRGAN_x2plus.fp16.onnx",
            "dest": MODELS / "upscale" / "RealESRGAN_x2plus.onnx",
            "sha256": None,
            "min_bytes": 10_000_000,
        },
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _download(url: str, dest: Path) -> bool:
    try:
        print(f"[model] downloading {dest.name}\n        {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
            total = resp.length or 0
            done = 0
            chunk = 1024 * 256
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                fh.write(data)
                done += len(data)
                if total:
                    pct = done * 100 // total
                    sys.stdout.write(f"\r        {pct:3d}%  ({done/1e6:.1f} MB)")
                    sys.stdout.flush()
        sys.stdout.write("\n")
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        print(f"[model] failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def _ensure(spec: dict) -> bool:
    dest: Path = spec["dest"]
    min_bytes = spec.get("min_bytes", 100_000)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        sha = spec.get("sha256")
        if sha and _sha256(dest) != sha:
            print(f"[model] {dest.name} checksum mismatch -> re-download")
        else:
            print(f"[model] {dest.name} already present ({dest.stat().st_size/1e6:.1f} MB)")
            return True
    if not _download(spec["url"], dest):
        print(f"[model] could not fetch {dest.name}; download it manually.")
        return False
    if dest.stat().st_size < min_bytes:
        print(f"[model] WARNING: {dest.name} is smaller than expected "
              f"({dest.stat().st_size} bytes).")
    sha = spec.get("sha256")
    if sha and _sha256(dest) != sha:
        print(f"[model] WARNING: {dest.name} checksum differs.")
    print(f"[model] {dest.name} OK ({dest.stat().st_size/1e6:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download colorization & upscale models.")
    ap.add_argument("--engine",
                    choices=list(SPECS) + ["all"], default="all",
                    help="which model group(s) to fetch (default: all)")
    args = ap.parse_args()

    groups = list(SPECS) if args.engine == "all" else [args.engine]
    results = {}
    for g in groups:
        print(f"\n=== {g} ===")
        ok = True
        for spec in SPECS[g]:
            ok = _ensure(spec) and ok
        results[g] = ok

    print()
    all_ok = True
    for g, ok in results.items():
        print(f"[model] {g:11} : {'OK' if ok else 'FAILED'}")
        all_ok = all_ok and ok
    if all_ok:
        print("\n[model] Ready. Run:  python src/cli.py check")
        return 0
    print("\n[model] Some files failed. See messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
