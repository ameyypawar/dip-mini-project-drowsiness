"""Week-2 image enhancement: size normalization + classical DIP enhancement.

Reused by weeks 3-8 (segmentation, morphology, transforms, classification,
live inference) -- every function here must run on a single grayscale
`np.ndarray` (uint8) unless stated otherwise, so it composes cleanly with
`src.dataset.EyeSample` loading.

Pipeline order, fixed by `preprocess_pipeline`:
    1. normalize_size   -- canonical square shape first, so every later
       step (CLAHE tile grid, segmentation thresholds, morphology,
       Hough transforms) operates on a consistent geometry regardless of
       the MRL crop's native size.
    2. suppress_glare   -- remove eyeglass IR specular hotspots before
       contrast enhancement, so CLAHE does not amplify a saturated blob
       that has not been repaired yet.
    3. apply_gamma      -- global brightness normalization (dark / bad-
       lighting frames).
    4. apply_clahe      -- local contrast enhancement, applied last so it
       acts on an already brightness/glare-corrected, canonically-sized
       image.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Dimension statistics (drives the canonical-size decision -- see
# docs/week2_enhancement.md and notebooks/02_image_enhancement.ipynb).
# ---------------------------------------------------------------------------

def dimension_stats(paths: Iterable[str | Path]) -> dict:
    """Compute width/height/aspect-ratio statistics over a set of image paths.

    Reads only image headers via PIL (`Image.open(...).size`), so this is
    cheap enough to run over the full MRL dataset (84,898 images, ~10s) --
    no sampling needed. Returns a dict with n_images, n_distinct_sizes, and
    min/max/mean/median for width, height, and aspect ratio (width/height).
    """
    widths: list[int] = []
    heights: list[int] = []
    distinct_sizes: set[tuple[int, int]] = set()

    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)
        distinct_sizes.add((w, h))

    if not widths:
        raise ValueError("dimension_stats got zero paths")

    aspect_ratios = [w / h for w, h in zip(widths, heights)]

    def _summary(xs: list[float]) -> dict[str, float]:
        return {
            "min": min(xs),
            "max": max(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
        }

    return {
        "n_images": len(widths),
        "n_distinct_sizes": len(distinct_sizes),
        "width": _summary(widths),
        "height": _summary(heights),
        "aspect_ratio": _summary(aspect_ratios),
    }


# ---------------------------------------------------------------------------
# Size normalization
# ---------------------------------------------------------------------------

def normalize_size(img: np.ndarray, target: int = 64, mode: str = "letterbox") -> np.ndarray:
    """Resize `img` to a canonical `target`x`target` square.

    mode="resize":     plain `cv2.resize` to (target, target) -- distorts
                        aspect ratio for non-square inputs, but every pixel
                        of the original crop is kept.
    mode="letterbox":  resize preserving aspect ratio so the longer side
                        equals `target`, then pad the shorter side with
                        zeros (black) to reach a `target`x`target` canvas
                        -- no distortion, but introduces a black border.

    Interpolation is chosen per-case, not fixed: `cv2.INTER_AREA` for
    shrinking (correct anti-aliasing, avoids moire) and `cv2.INTER_CUBIC`
    for enlarging (smooth upsampling), matching OpenCV's own guidance that
    INTER_AREA is for decimation and INTER_CUBIC/INTER_LINEAR for zooming.
    """
    h, w = img.shape[:2]

    if mode == "resize":
        shrinking = (target < w) or (target < h)
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
        return cv2.resize(img, (target, target), interpolation=interp)

    if mode == "letterbox":
        scale = target / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        canvas_shape = (target, target) if img.ndim == 2 else (target, target, img.shape[2])
        canvas = np.zeros(canvas_shape, dtype=img.dtype)
        y0 = (target - new_h) // 2
        x0 = (target - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    raise ValueError(f"unknown mode {mode!r}, expected 'resize' or 'letterbox'")


# ---------------------------------------------------------------------------
# Contrast / brightness enhancement
# ---------------------------------------------------------------------------

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization.

    Equalizes the histogram locally within `tile_grid` tiles rather than
    globally, so illumination varies across the image (e.g. eyelid shadow
    vs. sclera highlight) don't wash out local contrast. `clip_limit` caps
    per-bin height before redistribution, preventing the noise
    amplification that plain adaptive histogram equalization causes in
    near-uniform regions.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def apply_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction via a precomputed 256-entry lookup table.

    output = 255 * (input / 255) ** (1 / gamma). gamma > 1 brightens
    (lifts shadow detail in dark/bad-lighting IR frames); gamma < 1
    darkens. Implemented as `cv2.LUT` over a table built once, not a
    per-pixel power call, since gamma correction on 8-bit images only has
    256 possible outputs.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


# ---------------------------------------------------------------------------
# Glare / specular-highlight suppression (eyeglass IR reflections)
# ---------------------------------------------------------------------------

def suppress_glare(
    img: np.ndarray,
    threshold: int = 235,
    dilate_px: int = 3,
    inpaint_radius: int = 5,
    method: str = "telea",
) -> np.ndarray:
    """Suppress IR specular highlights (eyeglass reflections) via threshold + inpaint.

    Classical 3-step DIP method:
      1. Global intensity threshold: eyeglass hotspots in IR eye crops are
         near-saturated pixels (close to 255) -- `img >= threshold`
         isolates candidate specular-highlight pixels as a binary mask.
      2. Morphological dilation (elliptical structuring element) grows the
         mask by `dilate_px` to cover the highlight's soft halo/bloom, so
         inpainting doesn't leave a bright ring at the mask boundary.
      3. PDE-based inpainting (`cv2.INPAINT_TELEA`, Telea 2004, or
         `cv2.INPAINT_NS`, Navier-Stokes/Bertalmio 2001) reconstructs the
         masked region from surrounding non-masked texture by propagating
         isophotes (edge/level lines) inward -- plausible iris/sclera
         texture replaces the blown-out hotspot instead of leaving a flat
         patch.

    No-op (returns `img` unchanged) if no pixel exceeds `threshold`.
    """
    mask = (img >= threshold).astype(np.uint8) * 255
    if not mask.any():
        return img

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, kernel)

    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, inpaint_radius, flag)


# ---------------------------------------------------------------------------
# Composed canonical pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    """Explicit, documented config for `preprocess_pipeline`.

    Defaults are the Week-2 chosen values -- see docs/week2_enhancement.md
    for the measurement/evidence behind each one.
    """

    target_size: int = 64
    resize_mode: str = "resize"          # 'resize' or 'letterbox'
    glare_suppression: bool = True
    glare_threshold: int = 235
    glare_dilate_px: int = 3
    glare_inpaint_radius: int = 5
    gamma: float = 1.5
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (4, 4)  # 16x16px tiles at 64x64 -- (8,8) degenerates here, see docs/week2_enhancement.md


DEFAULT_CONFIG = PreprocessConfig()


def preprocess_pipeline(img: np.ndarray, cfg: PreprocessConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Canonical Week-2 pipeline: normalize_size -> suppress_glare -> apply_gamma -> apply_clahe.

    This is the single entry point weeks 3-8 should call to turn a raw
    variable-size MRL grayscale crop (or a live webcam eye ROI) into the
    canonical enhanced image that segmentation/morphology/transforms/
    classification all assume as input.
    """
    out = normalize_size(img, target=cfg.target_size, mode=cfg.resize_mode)
    if cfg.glare_suppression:
        out = suppress_glare(
            out,
            threshold=cfg.glare_threshold,
            dilate_px=cfg.glare_dilate_px,
            inpaint_radius=cfg.glare_inpaint_radius,
        )
    out = apply_gamma(out, gamma=cfg.gamma)
    out = apply_clahe(out, clip_limit=cfg.clahe_clip_limit, tile_grid=cfg.clahe_tile_grid)
    return out
