"""ONNX-based image upscaling (Real-ESRGAN x2plus) for anime/manga.

Runs on CPU via onnxruntime with **tiling**, so large pages stay within RAM.
The model is RGB-float, expects values in [0,1], and outputs x2 resolution.

Pipeline-friendly: works on a BGR uint8 image (OpenCV convention) and returns
a BGR uint8 image at ``factor`` times the size.

Reference: Real-ESRGAN (Wang et al., 2021). ONNX export: tamnvcc/RealESRGAN-onnx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent


class Upscaler:
    """Lazy-loading ONNX Real-ESRGAN upscaler with tiling."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.factor = int(settings.get("upscale_factor", 2))
        self.tile = int(settings.get("upscale_tile", 256))
        self.max_side = int(settings.get("upscale_max_side", 2600))
        self._sess = None

        model = settings.get("upscale_model", "models/upscale/RealESRGAN_x2plus.onnx")
        self.model_path = ROOT / model

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Upscale model missing: {self.model_path}\n"
                    "Run:  python scripts/download_model.py --engine upscale"
                )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 0  # use all cores
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(
                str(self.model_path), sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._sess

    # ------------------------------------------------------------------
    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """Upscale a single RGB float32 [0,1] HWC array (no tiling)."""
        sess = self._session()
        h, w = rgb.shape[:2]
        # The model downsamples/upsamples internally; pad to a multiple of 4
        # (and at least 8) so the Reshape nodes are valid, then crop after.
        mult = 8
        ph = (mult - h % mult) % mult
        pw = (mult - w % mult) % mult
        if ph or pw:
            padded = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
        else:
            padded = rgb
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(None, {"input": inp})[0]
        out = out[0].transpose(1, 2, 0)
        # Crop back to the true upscaled size.
        out = out[: h * self.factor, : w * self.factor]
        return np.clip(out, 0.0, 1.0)

    def _run_tiled(self, rgb: np.ndarray) -> np.ndarray:
        """Tiled inference to limit peak memory on large pages."""
        h, w = rgb.shape[:2]
        f = self.factor
        tile = self.tile
        out = np.zeros((h * f, w * f, 3), np.float32)

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y1, x1 = min(y + tile, h), min(x + tile, w)
                patch = rgb[y:y1, x:x1]
                up = self._run(patch)
                out[y * f:y1 * f, x * f:x1 * f] = up[: (y1 - y) * f, : (x1 - x) * f]
        return out

    # ------------------------------------------------------------------
    def upscale_image(self, bgr: np.ndarray, factor: int | None = None) -> np.ndarray:
        """Upscale a BGR uint8 image. Returns BGR uint8 at ``factor``× size."""
        factor = factor or self.factor
        if bgr.ndim == 2:
            bgr = np.repeat(bgr[:, :, None], 3, axis=2)
        elif bgr.shape[2] == 4:
            bgr = bgr[:, :, :3]

        h, w = bgr.shape[:2]
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0

        if max(rgb.shape[:2]) > self.tile:
            out = self._run_tiled(rgb)
        else:
            out = self._run(rgb)

        out_u8 = (out * 255.0).round().astype(np.uint8)
        out_bgr = out_u8[:, :, ::-1].copy()

        # If a different factor was requested than the model's native, resize.
        native = self.factor
        if factor != native:
            target = (w * factor, h * factor)
            if cv2 is not None:
                out_bgr = cv2.resize(out_bgr, target, interpolation=cv2.INTER_LANCZOS4)
            else:
                from PIL import Image
                out_bgr = np.asarray(
                    Image.fromarray(out_bgr[:, :, ::-1]).resize(target, Image.LANCZOS)
                )[:, :, ::-1]
        return out_bgr

    # ------------------------------------------------------------------
    def upscale_file(self, src: Path | str, dst: Path | str,
                     factor: int | None = None) -> bool:
        src, dst = Path(src), Path(dst)
        if cv2 is not None:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        else:
            from PIL import Image
            with Image.open(src) as im:
                img = np.asarray(im.convert("RGB"))[:, :, ::-1]
        if img is None:
            raise ValueError(f"Could not read image: {src}")
        out = self.upscale_image(img, factor=factor)
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically: to a temp file in the same folder, then replace.
        # This avoids races where a reader (or the very same path being read
        # and written) can see a stale/partial file.
        import os
        import tempfile

        suffix = dst.suffix or ".png"
        fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), suffix=suffix)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            if cv2 is not None:
                ok = bool(cv2.imwrite(str(tmp_path), out))
            else:
                from PIL import Image
                Image.fromarray(out[:, :, ::-1]).save(tmp_path)
                ok = True
            if not ok:
                raise IOError("imwrite failed")
            os.replace(str(tmp_path), str(dst))
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python src/upscale.py <in> <out>")
        raise SystemExit(2)
    s = {"upscale_model": "models/upscale/RealESRGAN_x2plus.onnx"}
    up = Upscaler(s)
    up.upscale_file(sys.argv[1], sys.argv[2])
    print(f"done -> {sys.argv[2]}")
