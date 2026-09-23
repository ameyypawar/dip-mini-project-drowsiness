"""Task 5 -- Sharpening Filters, Filter Comparison, and Module Integration.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Extends -- does not duplicate -- Weeks 3 and 4:
  - Task 3 (`src.enhancement`): negative/log/gamma/contrast-stretch/hist-eq,
    plus `noise_proxy` (reused below).
  - Task 4 (`src.spatial_filtering`): noise injection, mean/Gaussian/median
    smoothing, Laplacian sharpening, unsharp masking, high-boost filtering,
    PSNR/SSIM/edge-preservation/timing metrics.

New in this module:
  1. Gradient-based sharpening (Sobel-derived) -- `gradient_sharpen`. Not
     present in Task 4: distinct from the Laplacian (2nd-derivative, signed)
     and unsharp/high-boost (low-pass-subtracted) mechanisms.
  2. `mse` -- Mean Squared Error. Task 4's `spatial_filtering` module has
     PSNR/SSIM/edge-preservation but not raw MSE, and Task 5 sub-task 2
     explicitly requires MSE alongside PSNR in the comparison table.
  3. `noise_amplification_ratio` -- measures (rather than asserts) how much
     a sharpening filter amplifies noise, reusing `src.enhancement.noise_proxy`.
  4. Thin re-export wrappers around Task 4's Laplacian / unsharp / high-boost
     sharpening (imported, not reimplemented), so this module is Task 5's
     single coherent entry point for "sharpening filters".
  5. `SMOOTH_DISPATCH` / `SHARPEN_DISPATCH` / `FILTER_DISPATCH` name->callable
     tables consumed by `scripts/enhance_app.py`'s `--filter` flag, mirroring
     `src.enhancement.TECHNIQUE_DISPATCH`'s existing pattern.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from src.enhancement import noise_proxy
from src.spatial_filtering import (
    gaussian_filter,
    high_boost,
    laplacian_sharpen,
    mean_filter,
    median_filter,
    unsharp_mask,
)

__all__ = [
    "gradient_sharpen",
    "laplacian_sharpen",
    "unsharp_mask",
    "high_boost",
    "mse",
    "noise_amplification_ratio",
    "SMOOTH_DISPATCH",
    "SHARPEN_DISPATCH",
    "FILTER_DISPATCH",
]


# ---------------------------------------------------------------------------
# 1. Gradient-based sharpening (new -- Sobel-derived, not in Task 4)
# ---------------------------------------------------------------------------

def gradient_sharpen(img: np.ndarray, ksize: int = 3, amount: float = 1.0) -> np.ndarray:
    """Sharpen via Sobel gradient magnitude: output = original + amount * |grad|.

    Mechanism distinct from both Task 4 sharpeners: `laplacian_sharpen` adds
    a signed 2nd-derivative response in a single convolution pass;
    `unsharp_mask`/`high_boost` subtract a low-pass blur from the original
    to get a signed high-frequency mask. Here the correction term is a
    *first*-derivative gradient MAGNITUDE (`sqrt(Gx^2+Gy^2)` via
    `cv2.Sobel`, always >= 0 -- the same building block `sobel_edge` in
    `src.edge_segmentation` uses for edge detection) added back onto the
    original, so every edge is brightened unconditionally rather than
    receiving a signed correction. `amount` scales how much gradient
    magnitude is added back; `ksize` is the Sobel kernel size (odd).
    Computed in float64 and clipped once at the end (same convention as
    `laplacian_sharpen`), so overshoot at strong edges saturates to 255
    instead of wrapping.
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"ksize must be odd and >= 1, got {ksize}")
    img_f = img.astype(np.float64)
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    sharpened = img_f + amount * grad_mag
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Metrics new to Task 5
# ---------------------------------------------------------------------------

def mse(clean: np.ndarray, test: np.ndarray) -> float:
    """Mean Squared Error between `clean` and `test` (uint8, same shape).

    Added here rather than in `src.spatial_filtering` because Task 5
    sub-task 2 explicitly asks for MSE alongside PSNR in the comparison
    table; `psnr = 10*log10(255^2/mse)` so the two always agree in
    direction (lower MSE = higher PSNR) -- MSE is reported as the required
    raw error measure, PSNR (`src.spatial_filtering.psnr`) as its dB form.
    """
    diff = clean.astype(np.float64) - test.astype(np.float64)
    return float(np.mean(diff ** 2))


def noise_amplification_ratio(clean: np.ndarray, sharpened: np.ndarray) -> float:
    """How much a sharpening filter amplifies noise, measured rather than
    just asserted (Task 5 sub-task 1).

    Ratio of `src.enhancement.noise_proxy` (std-dev of the residual
    `img - median_filter_3x3(img)` -- high-frequency content a 3x3 median
    treats as noise rather than real structure) on `sharpened` over the
    same proxy on `clean`. 1.0 = no change in high-frequency content;
    >1.0 = sharpening increased it by that factor (noise amplification);
    <1.0 would mean the filter net-suppressed high-frequency content.
    """
    base = noise_proxy(clean)
    if base <= 0:
        return float("inf") if noise_proxy(sharpened) > 0 else 1.0
    return noise_proxy(sharpened) / base


# ---------------------------------------------------------------------------
# 3. Dispatch tables consumed by scripts/enhance_app.py's --filter flag
# ---------------------------------------------------------------------------

SMOOTH_DISPATCH: dict[str, Callable[..., np.ndarray]] = {
    "mean": mean_filter,
    "gaussian": gaussian_filter,
    "median": median_filter,
}

SHARPEN_DISPATCH: dict[str, Callable[..., np.ndarray]] = {
    "laplacian": lambda img, **_kw: laplacian_sharpen(img),
    "unsharp": unsharp_mask,
    "high_boost": high_boost,
    "gradient": gradient_sharpen,
}

# Full table -- name -> callable(img, **kwargs) -> img -- used by
# scripts/enhance_app.py so its CLI never needs an if/elif ladder
# duplicating this module's API (same pattern as
# src.enhancement.TECHNIQUE_DISPATCH).
FILTER_DISPATCH: dict[str, Callable[..., np.ndarray]] = {**SMOOTH_DISPATCH, **SHARPEN_DISPATCH}
