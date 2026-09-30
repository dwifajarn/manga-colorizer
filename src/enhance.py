"""Image enhancement for manga colorization.

Two stages:

1. ``preprocess_gray`` — clean a (possibly screenshot) grayscale page before
   colorization: contrast / white balance normalisation and light denoise.

2. ``postprocess_color`` — make the colorized result clean and vivid:
   * saturation / vibrance boost first,
   * then edge-aware smoothing of the ``ab`` channels (guided filter) to kill
     the speckle ("burik") while keeping line edges sharp,
   * finally keep paper/whites truly white (de-tint near-white pixels).

All functions operate on OpenCV BGR / grayscale ``np.ndarray``.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def preprocess_gray(gray: np.ndarray, settings: dict) -> np.ndarray:
    """Clean a grayscale page before colorization. Returns uint8 grayscale.

    Steps (all optional / configurable):
      1. robust contrast normalisation (paper -> white, ink -> black)
      2. screentone cleaning — low-pass the halftone dots so the colorizer is
         not confused by the dot pattern (major cause of "burik")
      3. light denoise (median) to remove anti-aliasing / JPEG grain
      4. gamma lift
    """
    if not settings.get("preprocess", True):
        return gray

    if gray.ndim == 3:
        gray = (0.299 * gray[:, :, 2] + 0.587 * gray[:, :, 1]
                + 0.114 * gray[:, :, 0]).astype(np.uint8)

    work = gray.astype(np.float32)

    # 1) Robust contrast normalisation: paper -> white, ink -> black.
    lo = float(settings.get("pre_black_point", 2.0))
    hi = float(settings.get("pre_white_point", 98.0))
    p_lo, p_hi = np.percentile(work, [lo, hi])
    if p_hi - p_lo < 1e-3:
        p_lo, p_hi = 0.0, 255.0
    work = (work - p_lo) * (255.0 / (p_hi - p_lo))
    work = np.clip(work, 0, 255)

    # 2) Screentone cleaning: halftone dots are high-frequency; a small blur
    #    merges them into smooth grey so the model sees flat shading instead.
    if settings.get("pre_clean_screentone", True) and cv2 is not None:
        k = int(settings.get("pre_screentone_ksize", 3))
        if k % 2 == 0:
            k += 1
        if k >= 3:
            work = cv2.GaussianBlur(work, (k, k), 0).astype(np.float32)
        # Re-sharpen edges so line art survives the blur.
        if settings.get("pre_sharpen_lines", True):
            blur = cv2.GaussianBlur(work, (0, 0), 1.0)
            work = cv2.addWeighted(work, 1.6, blur, -0.6, 0).astype(np.float32)
            work = np.clip(work, 0, 255)

    # 3) Light denoise to remove screenshot anti-aliasing / JPEG grain.
    if settings.get("pre_denoise", True) and cv2 is not None:
        k = int(settings.get("pre_median_ksize", 3))
        if k % 2 == 0:
            k += 1
        if k >= 3:
            work = cv2.medianBlur(work.astype(np.uint8), k).astype(np.float32)

    # 4) Gamma to lift midtones.
    gamma = float(settings.get("pre_gamma", 1.0))
    if abs(gamma - 1.0) > 1e-3:
        work = 255.0 * np.power(np.clip(work, 0, 255) / 255.0, gamma)

    return np.clip(work, 0, 255).astype(np.uint8)


def extract_lineart(gray: np.ndarray, settings: dict | None = None) -> np.ndarray:
    """Extract a clean binary-ish line map from a grayscale page.

    Returns a float32 mask in [0,1] where 1.0 = ink/line. Uses adaptive
    thresholding plus a mild morphological close so faint lines are kept
    without grabbing screentone.
    """
    settings = settings or {}
    if cv2 is None:
        # crude fallback: darkness = 1 - normalised
        g = gray.astype(np.float32) / 255.0
        return np.clip(1.0 - g, 0, 1)

    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    block = int(settings.get("line_block_size", 15))
    if block % 2 == 0:
        block += 1
    C = float(settings.get("line_C", 8))
    binary = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, C
    )
    # Also consider strongly dark pixels regardless of local contrast.
    dark = (g < float(settings.get("line_dark_threshold", 100))).astype(np.uint8) * 255
    mask = cv2.bitwise_or(binary, dark)
    # Remove tiny specks (screentone residue) but keep connected lines.
    k = int(settings.get("line_open_ksize", 2))
    if k >= 2:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask.astype(np.float32) / 255.0)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing (He et al. guided filter).

    ``guide`` and ``src`` are float32 in [0, 255]. Returns filtered ``src``.
    We use a small radius so weak colour gradients survive but speckle is gone.
    """
    if cv2 is None:
        return src
    d = radius * 2 + 1
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (d, d))
    mean_p = cv2.boxFilter(src, cv2.CV_32F, (d, d))
    corr_I = cv2.boxFilter(guide * guide, cv2.CV_32F, (d, d))
    corr_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (d, d))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (d, d))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (d, d))
    return mean_a * guide + mean_b


def postprocess_color(bgr: np.ndarray, settings: dict,
                      gray: np.ndarray | None = None) -> np.ndarray:
    """Clean up a colorized image: protect lines, de-speckle, boost saturation.

    Parameters
    ----------
    bgr : colorized image (BGR uint8).
    settings : configuration dict.
    gray : the original (preprocessed) grayscale page used as guidance for
           line detection. If ``None``, it is derived from ``bgr``.
    """
    if not settings.get("postprocess", True):
        return bgr
    if cv2 is None or bgr.ndim != 3:
        return bgr

    if gray is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = bgr.shape[:2]
    if gray.shape[:2] != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)

    # ---- Mode A: line overlay (keeps colours, re-ink the black outlines). ----
    if settings.get("post_mode", "line_overlay") == "line_overlay":
        out = _line_overlay(bgr, gray, settings)
        if settings.get("post_white_preserve", True):
            out = _detint_white(out, settings)
        return out

    # ---- Mode B: LAB-based protection + de-speckle (legacy). ----
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1].copy()
    b = lab[:, :, 2].copy()

    # --- 1) Saturation / vibrance boost. ---
    sat = float(settings.get("post_saturation", 1.35))
    a = (a - 128.0) * sat + 128.0
    b = (b - 128.0) * sat + 128.0

    # --- 2) Edge-aware smoothing of ab -> kill speckle / "burik". ---
    guide = L
    radius = int(settings.get("post_smooth_radius", 6))
    eps = float(settings.get("post_smooth_eps", 0.02)) * (255.0 * 255.0)
    a = _guided_filter(guide, a, radius, eps)
    b = _guided_filter(guide, b, radius, eps)

    # --- 3) Line-art protection: keep ink/outlines neutral (black). ---
    if settings.get("post_protect_lines", True):
        dark_thr = float(settings.get("post_line_dark", 90))
        soft = float(settings.get("post_line_soft", 40))
        mask = np.clip((dark_thr + soft - gray.astype(np.float32)) / max(1.0, soft), 0.0, 1.0)
        k = int(settings.get("post_line_dilate", 1))
        if k > 0:
            mask = cv2.dilate(mask, np.ones((k * 2 + 1, k * 2 + 1), np.float32))
        a = a * (1.0 - mask) + 128.0 * mask
        b = b * (1.0 - mask) + 128.0 * mask

    # --- 4) Screentone / texture desaturation. ---
    if settings.get("post_desat_texture", True):
        strength = float(settings.get("post_texture_strength", 0.6))
        ksize = int(settings.get("post_texture_ksize", 5))
        if ksize % 2 == 0:
            ksize += 1
        mean = cv2.blur(gray.astype(np.float32), (ksize, ksize))
        sq = cv2.blur((gray.astype(np.float32)) ** 2, (ksize, ksize))
        var = np.clip(sq - mean * mean, 0, None)
        thr = float(settings.get("post_texture_threshold", 180.0))
        tex = np.clip(var / max(1.0, thr), 0.0, 1.0) * strength
        a = a * (1.0 - tex) + 128.0 * tex
        b = b * (1.0 - tex) + 128.0 * tex

    max_dev = float(settings.get("post_max_dev", 80.0))
    a = np.clip(a, 128.0 - max_dev, 128.0 + max_dev)
    b = np.clip(b, 128.0 - max_dev, 128.0 + max_dev)

    lab[:, :, 1] = a
    lab[:, :, 2] = b
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    if settings.get("post_white_preserve", True):
        out = _detint_white(out, settings)
    return out


def _line_overlay(bgr: np.ndarray, gray: np.ndarray, settings: dict) -> np.ndarray:
    """Re-ink the black line art over the colorized image AND restore paper.

    Two things happen, both driven by the *original* grayscale luminance:

    * **Line re-ink** — where the source is dark (ink), blend the colour toward
      the original grey value, so outlines are crisp and black again.
    * **Paper restore** — where the source is near-white (paper), blend toward
      pure white, which removes the coloured haze the model adds to blank areas.

    Colour in the midtones (skin, hair, cloth) is left untouched, so nothing is
    washed out.
    """
    g = gray.astype(np.float32)
    out = bgr.astype(np.float32)

    # --- 1) Line re-ink (dark pixels -> ink grey). ---
    dark = float(settings.get("post_line_dark", 90))
    soft = float(settings.get("post_line_soft", 40))
    in_ink = np.clip((dark + soft - g) / max(1.0, soft), 0.0, 1.0)
    in_ink = in_ink * float(settings.get("post_line_strength", 1.0))
    m = in_ink[:, :, None]
    out = out * (1.0 - m) + g[:, :, None] * m

    # --- 2) Paper restore (near-white pixels -> pure white). ---
    if settings.get("post_paper_restore", True):
        w_lo = float(settings.get("post_paper_lo", 228))
        w_hi = float(settings.get("post_paper_hi", 250))
        paper = np.clip((g - w_lo) / max(1.0, (w_hi - w_lo)), 0.0, 1.0)
        p = paper[:, :, None] * float(settings.get("post_paper_strength", 1.0))
        out = out * (1.0 - p) + 255.0 * p

    return np.clip(out, 0, 255).astype(np.uint8)


def _detint_white(bgr: np.ndarray, settings: dict) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    thr = float(settings.get("post_white_threshold", 230))
    fade = np.clip((v - thr) / max(1.0, 255.0 - thr), 0.0, 1.0)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - fade)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
