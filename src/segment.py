"""Week-3 segmentation: threshold-based iris/pupil isolation on enhanced eye crops.

Intensity assumption (drives every function below): in these MRL near-IR eye
crops, after `src.preprocess.preprocess_pipeline` (CLAHE + gamma + glare
suppression) has run, the **pupil/iris is the darkest structure** in the
frame and the **sclera/skin is comparatively bright** -- IR illumination
reflects strongly off sclera and skin, weakly off the pupil. So "foreground"
in every mask produced here means *dark* pixels, and every `cv2.threshold`/
`cv2.adaptiveThreshold` call below uses the `*_INV` variant (pixels **below**
the threshold become the 255 foreground).

On a **closed** eye there is no iris/pupil to find -- the frame is mostly
uniform eyelid skin with thin dark eyelash/crease strokes. Thresholding still
runs (it cannot know eye state), but the region-isolation step
(`isolate_dark_region`) enforces a **minimum area fraction**: eyelash strokes
are thin and small, so they fail the area test and the function honestly
returns an **empty mask** rather than fabricating an iris that is not there.
That emptiness is itself the open/closed signal downstream weeks consume.

**Measured caveat (see docs/week3_segmentation.md):** MRL crops are already
tightly bound around the eye, so a plain Otsu split of the *whole* 64x64
frame lands near the histogram median regardless of eye state (measured
mean dark-area-fraction: 0.501 closed vs. 0.497 open, over all 6000 working-
set images -- not distinguishable) -- Otsu's global sclera/iris split is not,
by itself, a pupil isolator on this dataset. Region isolation therefore
anchors on a **stricter, Otsu-derived threshold**
(`otsu_value * pupil_threshold_scale`, scale < 1) to isolate the pupil's
darker core specifically, before applying the largest-component + min-area
test. This recovers a real, if imperfect, open/closed gap (measured
degenerate-rate gap ~0.18, naive same-signal accuracy ~59% at the chosen
defaults) -- reported honestly as a modest signal, not a solved classifier;
Week 4's shape features (not just area) carry more of the real separating
power, see docs/week4_morphology.md.

Pipeline for one image: `otsu_threshold` or `adaptive_threshold` (raw binary
mask, general sclera/iris split, used for the method comparison) ->
`isolate_dark_region` on the stricter pupil-anchored mask (largest-component
pupil/iris candidate, or empty on closed eyes) -> `segment_eye` composes all
of this into one result object. Week 4 (`src.morphology`) takes the mask
from here and cleans it further (opening/closing/hole-filling) before
feature extraction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def otsu_threshold(img: np.ndarray) -> tuple[float, np.ndarray]:
    """Global Otsu thresholding, dark-foreground convention.

    Otsu picks the threshold that minimizes intra-class intensity variance
    over the *whole image* histogram -- appropriate here because CLAHE
    (Week 2) has already normalized local contrast, so a single global split
    between "dark iris/pupil" and "bright sclera/skin" is meaningful across
    the whole 64x64 crop, not just a sub-region.

    Returns (threshold_value, binary_mask) where `binary_mask` is uint8
    {0, 255} and 255 marks pixels *darker* than the Otsu threshold.
    """
    thresh_val, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(thresh_val), mask


def adaptive_threshold(
    img: np.ndarray,
    method: str = "gaussian",
    block_size: int = 15,
    C: int = 3,
) -> np.ndarray:
    """Local adaptive thresholding, dark-foreground convention.

    Each pixel is compared against a statistic (mean or Gaussian-weighted
    mean) of its own `block_size`x`block_size` neighbourhood minus `C`, so
    the threshold varies spatially -- intended to help when illumination is
    uneven across the crop. `method="mean"` uses `ADAPTIVE_THRESH_MEAN_C`
    (unweighted local mean); `method="gaussian"` uses
    `ADAPTIVE_THRESH_GAUSSIAN_C` (Gaussian-weighted local mean, smoother
    response to neighbourhood edges). `block_size` must be odd and >= 3.
    """
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError(f"block_size must be odd and >= 3, got {block_size}")
    flag = cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(
        img, 255, flag, cv2.THRESH_BINARY_INV, block_size, C
    )


# ---------------------------------------------------------------------------
# Region isolation
# ---------------------------------------------------------------------------

def isolate_dark_region(mask: np.ndarray, min_area_frac: float = 0.03) -> tuple[np.ndarray, dict]:
    """Isolate the single largest connected dark component as the iris/pupil candidate.

    Runs 8-connectivity connected-component labelling on `mask` (a binary
    {0,255} dark-foreground mask from `otsu_threshold`/`adaptive_threshold`)
    and keeps only the **largest** component, provided its area clears
    `min_area_frac` of the image area. Rejecting small components is the
    honest handling of closed eyes: eyelash strokes and shadow specks are
    small and thin and will not pass this test, so the returned mask is
    **all-zero** -- there is no iris to isolate, and no component is forced
    to stand in for one.

    Returns (region_mask, info) where info has keys: n_components_raw
    (components in the input mask before filtering), largest_area (pixel
    count of the kept component, 0 if none kept), is_degenerate (True if no
    component cleared the area threshold).
    """
    h, w = mask.shape[:2]
    min_area = min_area_frac * h * w

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # label 0 is background; candidates are labels 1..n_labels-1
    n_components_raw = n_labels - 1

    region_mask = np.zeros_like(mask)
    largest_area = 0
    largest_label = -1
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = lbl

    is_degenerate = largest_label == -1 or largest_area < min_area
    if not is_degenerate:
        region_mask[labels == largest_label] = 255
    else:
        largest_area = 0

    info = {
        "n_components_raw": n_components_raw,
        "largest_area": int(largest_area),
        "is_degenerate": bool(is_degenerate),
    }
    return region_mask, info


# ---------------------------------------------------------------------------
# Composed segmentation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentConfig:
    """Config for `segment_eye`. Defaults are the Week-3 chosen values --
    see docs/week3_segmentation.md for the quantitative comparison behind
    the method choice.
    """

    method: str = "otsu"  # 'otsu' | 'adaptive_mean' | 'adaptive_gaussian'
    adaptive_block_size: int = 15
    adaptive_C: int = 3
    pupil_threshold_scale: float = 0.7  # scales the Otsu value down to anchor on the pupil's darker core
    min_region_area_frac: float = 0.01


DEFAULT_SEGMENT_CONFIG = SegmentConfig()


@dataclass(frozen=True)
class SegmentResult:
    raw_mask: np.ndarray       # general sclera/iris split (method comparison), dark=255
    pupil_mask_raw: np.ndarray  # stricter Otsu-anchored dark-core mask, before region isolation
    region_mask: np.ndarray    # largest-component isolated candidate from pupil_mask_raw, empty if degenerate
    threshold_value: float | None  # Otsu's picked value for raw_mask; None for adaptive methods
    otsu_value: float          # always computed (pupil-isolation anchor), regardless of cfg.method
    is_degenerate: bool
    dark_area_frac: float      # region_mask foreground fraction of image area (0.0 if degenerate)


def segment_eye(img: np.ndarray, cfg: SegmentConfig = DEFAULT_SEGMENT_CONFIG) -> SegmentResult:
    """Threshold + isolate the dark iris/pupil region on one preprocessed eye crop.

    `img` should already be `src.preprocess.preprocess_pipeline` output
    (64x64 uint8 grayscale). `raw_mask` is the general sclera/iris split
    (Otsu or adaptive, per `cfg.method`) used for the Week-3 method
    comparison. Region isolation is anchored on Otsu specifically
    (`otsu_value * cfg.pupil_threshold_scale`) regardless of `cfg.method`,
    because Otsu yields one global scalar to scale down toward the pupil's
    darker core -- adaptive thresholds are local per-pixel and have no
    single value to anchor on. See module docstring for the measured
    justification.
    """
    threshold_value: float | None = None
    otsu_value, otsu_mask = otsu_threshold(img)
    if cfg.method == "otsu":
        threshold_value = otsu_value
        raw_mask = otsu_mask
    elif cfg.method == "adaptive_mean":
        raw_mask = adaptive_threshold(img, method="mean", block_size=cfg.adaptive_block_size, C=cfg.adaptive_C)
    elif cfg.method == "adaptive_gaussian":
        raw_mask = adaptive_threshold(img, method="gaussian", block_size=cfg.adaptive_block_size, C=cfg.adaptive_C)
    else:
        raise ValueError(f"unknown method {cfg.method!r}")

    strict_val = otsu_value * cfg.pupil_threshold_scale
    pupil_mask_raw = (img < strict_val).astype(np.uint8) * 255

    region_mask, info = isolate_dark_region(pupil_mask_raw, min_area_frac=cfg.min_region_area_frac)
    h, w = img.shape[:2]
    dark_area_frac = info["largest_area"] / (h * w)

    return SegmentResult(
        raw_mask=raw_mask,
        pupil_mask_raw=pupil_mask_raw,
        region_mask=region_mask,
        threshold_value=threshold_value,
        otsu_value=otsu_value,
        is_degenerate=info["is_degenerate"],
        dark_area_frac=dark_area_frac,
    )


# ---------------------------------------------------------------------------
# Quantitative method comparison (drives the Week-3 method choice)
# ---------------------------------------------------------------------------

def compare_methods_on_image(img: np.ndarray, cfg: SegmentConfig = DEFAULT_SEGMENT_CONFIG) -> dict:
    """Run otsu / adaptive_mean / adaptive_gaussian on one image, return per-method stats.

    Used to aggregate quantitative comparisons (fragmentation, area
    fraction, timing) across the working set -- see
    scripts/compare_segmentation_methods.py and
    docs/week3_segmentation.md.
    """
    out = {}
    for method in ("otsu", "adaptive_mean", "adaptive_gaussian"):
        t0 = time.perf_counter()
        result = segment_eye(img, SegmentConfig(
            method=method,
            adaptive_block_size=cfg.adaptive_block_size,
            adaptive_C=cfg.adaptive_C,
            min_region_area_frac=cfg.min_region_area_frac,
        ))
        elapsed = time.perf_counter() - t0
        raw_labels = cv2.connectedComponentsWithStats(result.raw_mask, connectivity=8)[0] - 1
        out[method] = {
            "n_components_raw": raw_labels,
            "dark_area_frac": result.dark_area_frac,
            "is_degenerate": result.is_degenerate,
            "elapsed_s": elapsed,
        }
    return out
