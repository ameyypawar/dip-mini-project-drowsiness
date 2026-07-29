"""Week-4 morphological operations: clean Week-3 segmentation masks, extract shape features.

Week 3 (`src.segment`) produces `pupil_mask_raw` -- a stricter, Otsu-anchored
dark-core threshold mask -- which is typically **fragmented**: measured mean
5.2 connected components per image on an 800-image sample (84% of images
have more than one component), from noise, eyelash strands crossing the
pupil boundary, and residual specular texture. Morphology exists precisely
to clean this up before any shape feature is trusted:

    1. **Opening** (erosion -> dilation) strips thin spurs and small speckle
       components that do not survive the structuring element -- measured
       to cut mean component count roughly in half (5.25 -> 2.58) at the
       chosen 3x3 ellipse, while retaining 66% of foreground area (a 5x5
       element over-erodes, down to 34% area retained -- eats real pupil
       pixels, not just noise).
    2. **Closing** (dilation -> erosion) bridges small gaps/nearby fragments
       of what should be one pupil blob -- measured to further reduce mean
       component count (2.58 -> 2.23 at a 5x5 ellipse) while only mildly
       inflating area (1.05x), i.e. it merges nearby pieces rather than
       swallowing unrelated dark regions.
    3. **Hole filling** patches interior gaps in an otherwise-solid blob
       (e.g. a missed specular pixel inside the pupil). Measured to affect
       only 5/800 sampled images here, because Week 2's glare suppression
       already inpaints most specular hotspots before segmentation ever
       runs -- included for completeness and for images that still slip
       through, not because it is doing heavy lifting on this dataset.

`clean_mask` composes exactly these three steps in this order. Erosion and
dilation are also exposed standalone (`erode`, `dilate`) since either may be
useful in isolation (e.g. a live Week-7 webcam ROI with different noise
characteristics).

Feature extraction (`extract_mask_features`) runs on the **cleaned** mask,
not the raw Week-3 one -- this is the shape evidence Week 6's classifier
consumes. A cleaned mask with zero surviving components (closed eye, no
qualifying dark core) is the honest degenerate case: every feature is
reported as 0.0 and `is_degenerate=1`, rather than fabricating shape
statistics for a region that was never found.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# Structuring elements and primitive operations
# ---------------------------------------------------------------------------

_SHAPE_FLAGS = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


def structuring_element(shape: str = "ellipse", size: int = 3) -> np.ndarray:
    """Build a `size`x`size` structuring element. `shape` in {'ellipse', 'rect', 'cross'}."""
    if shape not in _SHAPE_FLAGS:
        raise ValueError(f"unknown shape {shape!r}, expected one of {list(_SHAPE_FLAGS)}")
    return cv2.getStructuringElement(_SHAPE_FLAGS[shape], (size, size))


def opening(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Erosion then dilation -- strips components/protrusions smaller than `kernel`."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def closing(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Dilation then erosion -- bridges gaps/nearby fragments within `kernel`'s reach."""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def erode(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Shrinks foreground regions -- removes boundary pixels not fully covered by `kernel`."""
    return cv2.erode(mask, kernel)


def dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Grows foreground regions -- adds boundary pixels touched by `kernel`."""
    return cv2.dilate(mask, kernel)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior background holes fully enclosed by foreground.

    `scipy.ndimage.binary_fill_holes` floods from outside the frame border
    inward; any background pixel it cannot reach is, by definition, enclosed
    by foreground and gets filled. Returns uint8 {0, 255}.
    """
    filled = binary_fill_holes(mask > 0)
    return (filled.astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Composed cleanup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MorphConfig:
    """Config for `clean_mask`. Defaults are the Week-4 chosen values -- see
    docs/week4_morphology.md for the structuring-element size/shape sweep
    behind each one.
    """

    open_shape: str = "ellipse"
    open_size: int = 3
    close_shape: str = "ellipse"
    close_size: int = 5
    do_fill_holes: bool = True


DEFAULT_MORPH_CONFIG = MorphConfig()


def clean_mask(mask: np.ndarray, cfg: MorphConfig = DEFAULT_MORPH_CONFIG) -> np.ndarray:
    """Canonical Week-4 cleanup: opening -> closing -> hole filling.

    Order matters: opening runs first to remove speckle/noise *before*
    closing has a chance to bridge that same noise into the main blob;
    hole filling runs last so it patches whatever gaps remain in the
    already-opened-and-closed result.
    """
    out = opening(mask, structuring_element(cfg.open_shape, cfg.open_size))
    out = closing(out, structuring_element(cfg.close_shape, cfg.close_size))
    if cfg.do_fill_holes:
        out = fill_holes(out)
    return out


# ---------------------------------------------------------------------------
# Feature extraction (feeds the Week-6 classifier)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "n_components",
    "dark_area_ratio",
    "largest_area",
    "largest_area_ratio",
    "filled_area",
    "filled_area_ratio",
    "centroid_row_norm",
    "centroid_col_norm",
    "eccentricity",
    "solidity",
    "extent",
    "bbox_aspect_ratio",
    "vertical_extent_norm",
    "is_degenerate",
]


def _zero_features(n_components: int = 0) -> dict:
    """Degenerate-case feature row: no qualifying dark region found."""
    return {
        "n_components": n_components,
        "dark_area_ratio": 0.0,
        "largest_area": 0.0,
        "largest_area_ratio": 0.0,
        "filled_area": 0.0,
        "filled_area_ratio": 0.0,
        "centroid_row_norm": 0.0,
        "centroid_col_norm": 0.0,
        "eccentricity": 0.0,
        "solidity": 0.0,
        "extent": 0.0,
        "bbox_aspect_ratio": 0.0,
        "vertical_extent_norm": 0.0,
        "is_degenerate": 1.0,
    }


def extract_mask_features(mask: np.ndarray, min_area_frac: float = 0.01) -> dict:
    """Shape/position features of the cleaned dark-region mask, via `skimage.measure.regionprops`.

    `mask` should already be `clean_mask` output. Labels all foreground
    components (8-connectivity via skimage's default `connectivity=2` on a
    2D image); if none clear `min_area_frac` of the frame area, returns the
    honest degenerate row (all-zero, `is_degenerate=1`) -- the closed-eye
    signal. Otherwise computes shape statistics on the **largest** surviving
    component (the pupil/iris candidate) while `n_components` and
    `dark_area_ratio` still summarize the *whole* cleaned mask (all
    surviving components, not just the largest), since fragmentation itself
    is diagnostic.

    `vertical_extent_norm` (bounding-box height / image height) is the
    eyelid-aperture proxy: how tall the isolated dark region stands in the
    frame.
    """
    h, w = mask.shape[:2]
    total_area = h * w
    min_area = min_area_frac * total_area

    labels = label(mask > 0, connectivity=2)
    regions = [r for r in regionprops(labels) if r.area >= min_area]

    if not regions:
        return _zero_features(n_components=0)

    dark_area_ratio = sum(r.area for r in regions) / total_area
    largest = max(regions, key=lambda r: r.area)

    minr, minc, maxr, maxc = largest.bbox
    bbox_h = maxr - minr
    bbox_w = maxc - minc
    cy, cx = largest.centroid

    return {
        "n_components": float(len(regions)),
        "dark_area_ratio": float(dark_area_ratio),
        "largest_area": float(largest.area),
        "largest_area_ratio": float(largest.area / total_area),
        "filled_area": float(largest.area_filled),
        "filled_area_ratio": float(largest.area_filled / total_area),
        "centroid_row_norm": float(cy / h),
        "centroid_col_norm": float(cx / w),
        "eccentricity": float(largest.eccentricity),
        "solidity": float(largest.solidity),
        "extent": float(largest.extent),
        "bbox_aspect_ratio": float(bbox_w / bbox_h) if bbox_h > 0 else 0.0,
        "vertical_extent_norm": float(bbox_h / h),
        "is_degenerate": 0.0,
    }
