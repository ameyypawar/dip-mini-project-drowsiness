"""Task 8 -- Binary morphological processing: erosion and dilation.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Masks are uint8 {0, 255}, 255 = foreground (dark pupil/iris region).
Thin wrappers over OpenCV's erode/dilate (default borders: erosion treats
outside-image as foreground, dilation as background, which keeps the
erosion/dilation duality exact), plus an independent set-theoretic NumPy
reference implementation used to verify the wrappers, plus measurement
helpers. Reuses src.morphology (structuring_element, opening, closing).
"""
from __future__ import annotations

import cv2
import numpy as np

from src import morphology as morph

SE_SPECS = {
    "square3": ("rect", 3),
    "cross3": ("cross", 3),
    "square5": ("rect", 5),
}


def to_binary(img: np.ndarray, threshold: float, dark_foreground: bool = True) -> np.ndarray:
    """Grayscale (converted if 3-channel) -> {0,255} mask. Dark pixels
    (< threshold) are foreground by default, matching the pupil convention."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (img < threshold) if dark_foreground else (img >= threshold)
    return (m.astype(np.uint8)) * 255


def make_se(shape: str, size: int) -> np.ndarray:
    """shape in {'square'/'rect', 'cross'}; returns uint8 size x size SE."""
    return morph.structuring_element("rect" if shape == "square" else shape, size)


def erode(mask: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.erode(mask, se, iterations=iterations)


def dilate(mask: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.dilate(mask, se, iterations=iterations)


# ---- independent set-theoretic reference (fit / hit) ----------------------

def erode_ref(mask: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """A (-) B = {z : B_z subset of A}; AND of shifted copies. Outside = fg."""
    a = mask > 0
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    for _ in range(iterations):
        p = np.pad(a, ((ph, ph), (pw, pw)), constant_values=True)
        out = np.ones_like(a)
        for i, j in zip(*np.nonzero(se)):
            out &= p[i:i + a.shape[0], j:j + a.shape[1]]
        a = out
    return a.astype(np.uint8) * 255


def dilate_ref(mask: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """A (+) B = {z : reflect(B)_z hits A}; OR of shifted copies. Outside = bg."""
    a = mask > 0
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    for _ in range(iterations):
        p = np.pad(a, ((ph, ph), (pw, pw)), constant_values=False)
        out = np.zeros_like(a)
        for i, j in zip(*np.nonzero(se)):
            out |= p[i:i + a.shape[0], j:j + a.shape[1]]
        a = out
    return a.astype(np.uint8) * 255


# ---- measurements ---------------------------------------------------------

def fg_count(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def area_fraction(mask: np.ndarray) -> float:
    return fg_count(mask) / mask.size


def n_components(mask: np.ndarray, connectivity: int = 8) -> int:
    return int(cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=connectivity)[0]) - 1


def component_areas(mask: np.ndarray, connectivity: int = 8) -> list[int]:
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=connectivity)
    return sorted(int(s) for s in stats[1:, cv2.CC_STAT_AREA])


def boundary_length(mask: np.ndarray) -> int:
    """Number of foreground pixels with at least one 4-neighbour background
    pixel (image border counts as background)."""
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    inner = cv2.erode(mask, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return int(((mask > 0) & (inner == 0)).sum())


def thin_pixels(mask: np.ndarray, se: np.ndarray) -> int:
    """|A| - |open(A)|: pixels in structures narrower than the SE."""
    return fg_count(mask) - fg_count(morph.opening(mask, se))


def gap_pixels(mask: np.ndarray, se: np.ndarray) -> int:
    """|close(A)| - |A|: background gap pixels the SE can bridge."""
    return fg_count(morph.closing(mask, se)) - fg_count(mask)


def measure(mask: np.ndarray, ref: np.ndarray | None = None) -> dict:
    ref = mask if ref is None else ref
    c = fg_count(mask)
    r = fg_count(ref)
    return {
        "fg_pixels": c,
        "area_fraction": round(area_fraction(mask), 5),
        "pct_change": round(100.0 * (c - r) / r, 2) if r else 0.0,
        "components": n_components(mask),
        "boundary_len": boundary_length(mask),
    }


def iterations_to_vanish(mask: np.ndarray, se: np.ndarray, max_iter: int = 150) -> int | None:
    """First erosion iteration at which no foreground remains (None if > max_iter)."""
    cur = mask
    for k in range(1, max_iter + 1):
        cur = cv2.erode(cur, se)
        if not cur.any():
            return k
    return None


def component_vanish_iterations(mask: np.ndarray, se: np.ndarray, max_iter: int = 150) -> list[int | None]:
    """Per original component (sorted by area), the erosion iteration at which it disappears."""
    n, lab = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        comp = ((lab == i).astype(np.uint8)) * 255
        out.append(iterations_to_vanish(comp, se, max_iter))
    return sorted(out, key=lambda v: (v is None, v))


def merge_iterations(mask: np.ndarray, se: np.ndarray, max_iter: int = 40) -> dict:
    """Dilation iteration at which component count first drops below the
    original, and the iteration it first reaches 1 (None if never)."""
    n0 = n_components(mask)
    first = one = None
    cur = mask
    for k in range(1, max_iter + 1):
        cur = cv2.dilate(cur, se)
        n = n_components(cur)
        if first is None and n < n0:
            first = k
        if n <= 1:
            one = k
            break
    return {"n0": n0, "first_merge_iter": first, "single_iter": one}


def duality_check(mask: np.ndarray, se: np.ndarray, iterations: int = 1) -> dict:
    """erode(A, B) == NOT dilate(NOT A, reflect(B)); also verifies OpenCV
    against the NumPy set-definition reference. Returns differing-pixel counts."""
    comp = 255 - mask
    se_r = se[::-1, ::-1].copy()
    lhs = erode(mask, se, iterations)
    rhs = 255 - dilate(comp, se_r, iterations)
    lhs2 = dilate(mask, se, iterations)
    rhs2 = 255 - erode(comp, se_r, iterations)
    return {
        "erode_vs_dual_diff": int((lhs != rhs).sum()),
        "dilate_vs_dual_diff": int((lhs2 != rhs2).sum()),
        "erode_cv_vs_ref_diff": int((lhs != erode_ref(mask, se, iterations)).sum()),
        "dilate_cv_vs_ref_diff": int((lhs2 != dilate_ref(mask, se, iterations)).sum()),
    }
