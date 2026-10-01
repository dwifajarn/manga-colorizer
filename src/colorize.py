"""Core colorization engine (PyTorch).

Uses the official *Colorful Image Colorization* model (Zhang, Isola & Efros,
ECCV 2016) in PyTorch form (``colorization_release_v2.pth``). It predicts an
``ab`` color field from the ``L`` (luminance) channel and recombines them into
a full-color image. Runs on CPU.

Network definition mirrors the reference implementation
(https://github.com/richzhang/colorization, BSD-2-Clause), kept self-contained
so there are no extra imports.

Public API:
    Colorizer                            -> loads weights once, reuse for many.
    Colorizer.colorize_image(np.ndarray) -> np.ndarray
    Colorizer.colorize_file(Path, Path)  -> bool
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import cv2
except ImportError:  # cv2 optional (faster resize + Lab conversion)
    cv2 = None

try:
    from enhance import preprocess_gray, postprocess_color
except ImportError:  # enhance is optional
    preprocess_gray = None
    postprocess_color = None

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = ROOT / "config" / "settings.json"

# Lab normalization constants (from BaseColor in the reference repo).
L_CENT, L_NORM, AB_NORM = 50.0, 100.0, 110.0


def load_settings(path: Path | str = DEFAULT_SETTINGS) -> dict[str, Any]:
    """Load settings.json, falling back to sensible defaults."""
    path = Path(path)
    defaults: dict[str, Any] = {
        "model_dir": "models",
        "model_file": "colorization_release_v2.pth",
        "long_side": 512,
        "device": "cpu",
        "supported_ext": [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"],
    }
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            defaults.update(json.load(fh))
    return defaults


def configure_threads(settings: dict[str, Any] | None = None) -> int:
    """Set torch / BLAS / onnxruntime intra-op threads before loading a model.

    The pipeline is sequential, but the compute is threaded internally. This
    applies ``settings['threads_per_worker']`` (default: all cores), unless an
    ``OMP_NUM_THREADS`` environment override asks for fewer. Call it *before*
    the model is built so the thread pools pick up the value.

    Returns the thread count used.
    """
    import os

    settings = settings or {}
    cores = os.cpu_count() or 1
    threads = settings.get("threads_per_worker")
    if threads is None:
        threads = cores
    threads = max(1, int(threads))

    env = os.environ.get("OMP_NUM_THREADS")
    if env and env.isdigit() and int(env) > 0:
        threads = min(threads, int(env))

    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:  # noqa: BLE001 - torch optional at this layer
        pass

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("ORT_NUM_THREADS", str(threads))
    return threads


# ---------------------------------------------------------------------------
# Network definition (ECCV16 colorization, Zhang et al.)
# ---------------------------------------------------------------------------
class _Eccv16(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d) -> None:
        super().__init__()
        self.l_cent, self.l_norm, self.ab_norm = L_CENT, L_NORM, AB_NORM

        model1 = [
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(True), norm_layer(64),
        ]
        model2 = [
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(True), norm_layer(128),
        ]
        model3 = [
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1), nn.ReLU(True), norm_layer(256),
        ]
        model4 = [
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), norm_layer(512),
        ]
        model5 = [
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2), nn.ReLU(True), norm_layer(512),
        ]
        model6 = [
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2), nn.ReLU(True), norm_layer(512),
        ]
        model7 = [
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), norm_layer(512),
        ]
        model8 = [
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 313, 1, 1, 0),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, 1, 1, 0, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def normalize_l(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.l_cent) / self.l_norm

    def unnormalize_ab(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.ab_norm

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))
        return self.unnormalize_ab(self.upsample4(out_reg))


# ---------------------------------------------------------------------------
class Colorizer:
    """Load-then-colorize wrapper around the PyTorch ECCV16 colorization net."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings = settings or load_settings()
        model_dir = ROOT / self.settings["model_dir"]
        model_path = model_dir / self.settings["model_file"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file missing: {model_path}\n"
                "Run:  python scripts/download_model.py"
            )

        self.device = torch.device(self.settings.get("device", "cpu"))
        self.long_side = int(self.settings.get("long_side", 512))

        self.net = _Eccv16().to(self.device).eval()
        state = torch.load(str(model_path), map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Model weights incomplete; missing {len(missing)} keys: {missing[:8]}")
        if unexpected:
            print(f"[colorize] note: ignored {len(unexpected)} unexpected keys")

    # ------------------------------------------------------------------
    def _resize(self, img: np.ndarray, size: tuple[int, int], interp: int) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(img, size, interpolation=interp)
        from PIL import Image
        pil = Image.fromarray(img)
        resample = Image.BILINEAR if interp == 1 else Image.NEAREST
        return np.asarray(pil.resize(size, resample))

    @staticmethod
    def _to_gray(bgr: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        r, g, b = (bgr[:, :, 2].astype(np.float32), bgr[:, :, 1].astype(np.float32),
                   bgr[:, :, 0].astype(np.float32))
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    @staticmethod
    def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        try:
            from skimage import color as skcolor
            rgb = skcolor.lab2rgb(lab.astype(np.float32) / 255.0)
            return (rgb[:, :, ::-1] * 255.0).astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Need OpenCV or scikit-image to convert Lab->BGR. "
                "Install with: pip install opencv-python"
            ) from exc

    # ------------------------------------------------------------------
    def colorize_image(self, bgr: np.ndarray) -> np.ndarray:
        """Colorize a BGR / grayscale image and return a BGR uint8 image."""
        if bgr.ndim == 2:
            gray = bgr
        elif bgr.shape[2] == 4:
            gray = self._to_gray(bgr[:, :, :3])
        else:
            gray = self._to_gray(bgr)

        # --- Pre-process: clean screenshot noise / normalise contrast. ---
        if preprocess_gray is not None and self.settings.get("preprocess", True):
            gray = preprocess_gray(gray, self.settings)

        orig_h, orig_w = gray.shape[:2]

        # Resize so long side == long_side for speed; keep multiples of 8 for
        # the encoder/decoder (three stride-2 stages).
        scale = self.long_side / max(orig_h, orig_w)
        new_w = max(8, (int(round(orig_w * scale)) // 8) * 8)
        new_h = max(8, (int(round(orig_h * scale)) // 8) * 8)

        gray_resized = self._resize(gray, (new_w, new_h), 1).astype(np.float32)
        l_channel = (gray_resized / 255.0) * 100.0  # L in [0, 100]

        with torch.no_grad():
            l_t = torch.from_numpy(l_channel).float()[None, None].to(self.device)
            ab = self.net(l_t)[0].cpu().numpy().transpose(1, 2, 0)  # (H,W,2) in [-110,110]

        # Build an OpenCV Lab image (uint8: L in 0..255, a/b in 0..255 offset 128).
        lab = np.empty((new_h, new_w, 3), np.float32)
        lab[:, :, 0] = l_channel / 100.0 * 255.0
        lab[:, :, 1] = ab[:, :, 0] + 128.0
        lab[:, :, 2] = ab[:, :, 1] + 128.0
        lab = np.clip(lab, 0, 255).astype(np.uint8)

        colorized = self._lab_to_bgr(lab)
        if (new_h, new_w) != (orig_h, orig_w):
            colorized = self._resize(colorized, (orig_w, orig_h), 1)
        colorized = colorized.astype(np.uint8)

        # --- Post-process: protect lines, de-speckle, saturation boost. ---
        if postprocess_color is not None and self.settings.get("postprocess", True):
            colorized = postprocess_color(colorized, self.settings, gray=gray)

        return colorized

    # ------------------------------------------------------------------
    def colorize_file(self, src: Path | str, dst: Path | str) -> bool:
        """Read ``src`` from disk, colorize, and write to ``dst``."""
        src, dst = Path(src), Path(dst)
        img = self._imread(src)
        if img is None:
            raise ValueError(f"Could not read image: {src}")
        out = self.colorize_image(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        return bool(self._imwrite(dst, out))

    @staticmethod
    def _imread(path: Path) -> np.ndarray | None:
        if cv2 is not None:
            return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        from PIL import Image
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        return arr[:, :, ::-1]

    @staticmethod
    def _imwrite(path: Path, bgr: np.ndarray) -> bool:
        if cv2 is not None:
            return cv2.imwrite(str(path), bgr)
        from PIL import Image
        Image.fromarray(bgr[:, :, ::-1]).save(path)
        return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python src/colorize.py <input_image> <output_image>")
        raise SystemExit(2)
    c = Colorizer()
    c.colorize_file(sys.argv[1], sys.argv[2])
    print(f"done -> {sys.argv[2]}")
