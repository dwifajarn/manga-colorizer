"""manga-colorization-v2 engine (qweasdd, GAN-based).

Wraps the vendored repo at ``src/mangacolv2_repo`` (MIT-ish, see repo). This is a
GAN colorizer specialised for manga: fast on CPU and produces clean flat colour
with shading — generally much better than the generic photo models for comics.

Weights (downloaded by scripts/download_model.py into ``models/mangacolv2/``):
    generator.zip   -> the GAN generator
    net_rgb.pth     -> FFDNet denoiser (used to clean the input before colorizing)

The vendored ``MangaColorizator`` expects the denoiser weights at the relative
path ``denoising/models/net_rgb.pth``; we copy them there on first use.

Public API mirrors the other engines:
    MangaColV2Colorizer.colorize_image(np.ndarray) -> np.ndarray
    MangaColV2Colorizer.colorize_file(Path, Path)  -> bool
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
_REPO = ROOT / "src" / "mangacolv2_repo"

# Make the vendored repo importable.
import sys as _sys
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))


class MangaColV2Colorizer:
    """Load-then-colorize wrapper around manga-colorization-v2."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        from colorize import load_settings

        self.settings = settings or load_settings()
        model_dir = ROOT / self.settings.get("mangacolv2_model_dir", "models/mangacolv2")
        gen_path = model_dir / self.settings.get("mangacolv2_generator", "generator.zip")
        den_path = model_dir / self.settings.get("mangacolv2_denoiser", "net_rgb.pth")

        if not gen_path.exists():
            raise FileNotFoundError(
                f"generator weights missing: {gen_path}\n"
                "Run:  python scripts/download_model.py --engine mangacolv2"
            )

        # The vendored denoiser loads from a relative path; ensure it's present.
        target_den = _REPO / "denoising" / "models" / "net_rgb.pth"
        target_den.parent.mkdir(parents=True, exist_ok=True)
        if den_path.exists() and (not target_den.exists()
                                  or target_den.stat().st_size != den_path.stat().st_size):
            shutil.copy2(den_path, target_den)

        self.size = int(self.settings.get("mangacolv2_size", 576))
        if self.size % 32 != 0:
            self.size = 576

        self.apply_denoise = bool(self.settings.get("mangacolv2_denoise", True))
        self.denoise_sigma = int(self.settings.get("mangacolv2_denoise_sigma", 25))

        device = self.settings.get("device", "cpu")
        # The vendored code branches on the literal string 'cuda'.
        import torch
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        from colorizator import MangaColorizator

        self._colorizer = MangaColorizator(
            self.device,
            generator_path=str(gen_path),
            extractor_path="",  # not needed at inference time
        )

    # ------------------------------------------------------------------
    def colorize_image(self, bgr: np.ndarray) -> np.ndarray:
        """Colorize a BGR / grayscale image, returning BGR uint8."""
        if bgr.ndim == 2:
            rgb = np.repeat(bgr[:, :, None], 3, axis=2)
        elif bgr.shape[2] == 4:
            rgb = np.dstack([bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]])
        else:
            # BGR -> RGB (the repo works in RGB / float [0,1]).
            rgb = np.dstack([bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]])

        # Normalise to [0,1] float as the vendored code expects.
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

        self._colorizer.set_image(
            rgb,
            size=self.size,
            apply_denoise=self.apply_denoise,
            denoise_sigma=self.denoise_sigma,
        )
        out = self._colorizer.colorize()  # RGB float [0,1]
        out = np.clip(out, 0.0, 1.0)
        out_u8 = (out * 255.0).astype(np.uint8)

        # The repo resizes the image during processing; restore original size.
        orig_h, orig_w = rgb.shape[:2]
        if out_u8.shape[:2] != (orig_h, orig_w):
            if cv2 is not None:
                out_u8 = cv2.resize(out_u8, (orig_w, orig_h),
                                    interpolation=cv2.INTER_LANCZOS4)
            else:
                from PIL import Image
                out_u8 = np.asarray(
                    Image.fromarray(out_u8).resize((orig_w, orig_h), Image.LANCZOS)
                )
        out_bgr = out_u8[:, :, ::-1].copy()  # RGB -> BGR

        # Optional clean-up: protect black line art & de-speckle colours.
        if self.settings.get("postprocess", True):
            try:
                from enhance import postprocess_color

                # Grayscale guide = original luminance.
                if cv2 is not None:
                    gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
                else:
                    gray = None
                out_bgr = postprocess_color(out_bgr, self.settings, gray=gray)
            except Exception as exc:  # noqa: BLE001
                print(f"[mangacolv2] postprocess skipped: {exc}")

        return out_bgr

    # ------------------------------------------------------------------
    def colorize_file(self, src: Path | str, dst: Path | str) -> bool:
        src, dst = Path(src), Path(dst)
        img = self._imread(src)
        if img is None:
            raise ValueError(f"Could not read image: {src}")
        out = self.colorize_image(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            return bool(cv2.imwrite(str(dst), out))
        from PIL import Image
        Image.fromarray(out[:, :, ::-1]).save(dst)
        return True

    @staticmethod
    def _imread(path: Path) -> np.ndarray | None:
        if cv2 is not None:
            return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))[:, :, ::-1]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python src/mangacolv2_engine.py <in> <out>")
        raise SystemExit(2)
    c = MangaColV2Colorizer()
    c.colorize_file(sys.argv[1], sys.argv[2])
    print(f"done -> {sys.argv[2]}")
