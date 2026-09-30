"""ColorComicNet engine (from the ``1plus1/MangaColorization`` HF Space).

A small, comic/anime-specific colorization network (~32 MB) that runs well on
CPU thanks to weight re-parameterization (``model.fuse()``). Unlike the generic
Zhang et al. photo model, it is trained on comics, so it does not smear
"burik" speckle on manga line art.

The network code lives in ``src/comicnet/`` (fetched from the Space, apache-2.0).
Weights: ``models/comicnet/colorizer.pth`` (via scripts/download_model.py).

Public API mirrors ``colorize.Colorizer``:
    ComicNetColorizer.colorize_image(np.ndarray) -> np.ndarray
    ComicNetColorizer.colorize_file(Path, Path)  -> bool
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

# Make the vendored comicnet package importable.
_HERE = Path(__file__).resolve().parent
import sys as _sys
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from comicnet.colorizer import ColorComicNet, MODEL_CFG  # noqa: E402
from comicnet.utils import smart_padding, remove_padding  # noqa: E402
ROOT = _HERE.parent


class ComicNetColorizer:
    """Load-then-colorize wrapper around ColorComicNet."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        from colorize import load_settings

        self.settings = settings or load_settings()
        model_dir = ROOT / self.settings.get("comicnet_model_dir", "models/comicnet")
        model_path = model_dir / self.settings.get("comicnet_model_file", "colorizer.pth")

        if not model_path.exists():
            raise FileNotFoundError(
                f"ColorComicNet weights missing: {model_path}\n"
                "Run:  python scripts/download_model.py"
            )

        self.device = torch.device(self.settings.get("device", "cpu"))
        self.divisor = int(self.settings.get("comicnet_divisor", 64))

        self.model = ColorComicNet(**MODEL_CFG).to(self.device)
        state = torch.load(str(model_path), map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[comicnet] note: {len(missing)} missing keys")
        if unexpected:
            print(f"[comicnet] note: {len(unexpected)} unexpected keys")
        try:
            self.model.fuse()  # fuse re-param convs -> faster CPU inference
        except Exception as exc:  # noqa: BLE001
            print(f"[comicnet] fuse skipped: {exc}")
        self.model.eval()

    # ------------------------------------------------------------------
    def _to_input_tensor(self, gray: np.ndarray) -> tuple[torch.Tensor, tuple]:
        # Build a 3-channel [0,1]->[-1,1] tensor from grayscale.
        t = torch.from_numpy(gray.astype(np.float32) / 255.0)
        t = t[None, None].repeat(1, 3, 1, 1)  # (1,3,H,W)
        t = (t - 0.5) / 0.5  # normalize mean .5 std .5
        t, padding = smart_padding(t, divisor=self.divisor)
        return t.to(self.device), padding

    def colorize_image(self, bgr: np.ndarray) -> np.ndarray:
        """Colorize a BGR / grayscale image, returning BGR uint8."""
        if bgr.ndim == 2:
            gray = bgr
        elif bgr.shape[2] == 4:
            rgb = bgr[:, :, :3]
            gray = self._to_gray(rgb)
        else:
            gray = self._to_gray(bgr)

        t, padding = self._to_input_tensor(gray)
        with torch.no_grad():
            out = self.model(t)
        out = remove_padding(out, padding)
        out = (out + 1.0) / 2.0  # -> [0,1]
        out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb = (out * 255.0).astype(np.uint8)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR

    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray(bgr: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        r, g, b = (bgr[:, :, 2].astype(np.float32), bgr[:, :, 1].astype(np.float32),
                   bgr[:, :, 0].astype(np.float32))
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

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
        print("usage: python src/comicnet_engine.py <in> <out>")
        raise SystemExit(2)
    c = ComicNetColorizer()
    c.colorize_file(sys.argv[1], sys.argv[2])
    print(f"done -> {sys.argv[2]}")
