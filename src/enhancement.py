"""Task 3 -- Image Enhancement in the Spatial Domain.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

All techniques operate on a single-channel `uint8` `np.ndarray`, matching the
convention used throughout `src.preprocess` / `src.segment` /
`src.spatial_filtering`. Nothing here modifies those modules; this file is
additive and self-contained.

Contents:
  A. Intensity transformations -- negative, log_transform, gamma_correction,
                                   contrast_stretch
  B. Histogram analysis         -- compute_histogram, histogram_stats
  C. Histogram equalization     -- equalize_histogram_cv (OpenCV),
                                    equalize_histogram_manual (own CDF impl.)
  D. Image arithmetic           -- add_images, add_constant, subtract_images,
                                    average_images, make_noisy_copies
  E. Comparison                 -- mean_intensity, rms_contrast, noise_proxy,
                                    compare_techniques

Every intensity-transform / arithmetic parameter (gamma, log constant `c`,
stretch percentiles/range, blend weight, brightness offset, noise sigma) is a
keyword argument with a sensible default, so the CLI (`scripts/enhance_app.py`)
and the report notebook can override it without touching this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# A. Intensity transformations
# ---------------------------------------------------------------------------

def negative(img: np.ndarray) -> np.ndarray:
    """Image negative: s = (L-1) - r, L = 256 for 8-bit images.

    No user-tunable parameter (the transform is fixed by definition); kept
    as a one-argument function for symmetry with the other techniques.
    """
    return (255 - img.astype(np.int16)).astype(np.uint8)


def log_transform(img: np.ndarray, c: float | None = None) -> np.ndarray:
    """Log transformation: s = c * log(1 + r).

    `c` is user-modifiable; if omitted it is auto-scaled so the image's own
    max intensity maps to 255 (`c = 255 / log(1 + max(r))`), which keeps the
    transform sane across differently-exposed IR crops instead of clipping
    or under-using the output range.
    """
    img_f = img.astype(np.float64)
    if c is None:
        peak = img_f.max()
        c = 255.0 / np.log1p(peak) if peak > 0 else 1.0
    out = c * np.log1p(img_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """Power-law (gamma) transformation: s = c * r^gamma, r normalized to [0,1].

    Both `gamma` and `c` are user-modifiable. gamma < 1 brightens (expands
    the dark end -- useful for underexposed IR crops); gamma > 1 darkens /
    increases contrast in bright regions.
    """
    img_norm = img.astype(np.float64) / 255.0
    out = c * np.power(img_norm, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_stretch(
    img: np.ndarray,
    r_min: float | None = None,
    r_max: float | None = None,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    out_min: int = 0,
    out_max: int = 255,
) -> np.ndarray:
    """Piecewise-linear (min-max) contrast stretching.

    `r_min`/`r_max` (the input clip range) are user-modifiable directly; if
    left `None` they default to the `low_pct`/`high_pct` percentiles of the
    image itself (robust to a handful of saturated/hot pixels, which small
    IR crops often have from glare). `out_min`/`out_max` set the output
    range and are also user-modifiable.
    """
    if r_min is None:
        r_min = float(np.percentile(img, low_pct))
    if r_max is None:
        r_max = float(np.percentile(img, high_pct))
    if r_max <= r_min:
        return img.copy()
    out = (img.astype(np.float64) - r_min) / (r_max - r_min)
    out = out * (out_max - out_min) + out_min
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# B. Histogram analysis
# ---------------------------------------------------------------------------

def compute_histogram(img: np.ndarray, bins: int = 256, range_: tuple[int, int] = (0, 256)):
    """Return (hist, bin_edges) for `img`, `bins` and `range_` user-modifiable."""
    hist, edges = np.histogram(img.ravel(), bins=bins, range=range_)
    return hist, edges


def histogram_stats(img: np.ndarray) -> dict:
    """Summary stats used to compare a histogram before/after enhancement."""
    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": int(img.min()),
        "max": int(img.max()),
    }


# ---------------------------------------------------------------------------
# C. Histogram equalization
# ---------------------------------------------------------------------------

def equalize_histogram_cv(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization via OpenCV's `cv2.equalizeHist`."""
    return cv2.equalizeHist(img)


def equalize_histogram_manual(img: np.ndarray) -> np.ndarray:
    """Own implementation of global histogram equalization (no cv2 call).

    Standard CDF-mapping algorithm: build the 256-bin histogram, take its
    cumulative sum, normalize so the smallest non-zero CDF value maps to 0
    and the max maps to 255, then apply the resulting lookup table.
    """
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return img.copy()
    cdf_min = nonzero.min()
    total = img.size
    denom = total - cdf_min
    if denom <= 0:
        return img.copy()
    lut = np.round((cdf - cdf_min) / denom * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[img]


# ---------------------------------------------------------------------------
# D. Image arithmetic
# ---------------------------------------------------------------------------

def _match_shape(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if img.shape == ref.shape:
        return img
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)


def add_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Weighted image addition / blend: out = alpha*img1 + (1-alpha)*img2.

    `alpha` is user-modifiable. Application: brightness boost via overlay,
    or compositing two exposures/frames (e.g. blending an IR frame with a
    brightened copy of itself to lift shadow detail without full gamma
    correction).
    """
    img2 = _match_shape(img2, img1)
    out = alpha * img1.astype(np.float64) + (1 - alpha) * img2.astype(np.float64)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_constant(img: np.ndarray, value: int = 40) -> np.ndarray:
    """Scalar addition: out = img + value (clipped to [0,255]).

    `value` is user-modifiable (may be negative to darken). Application:
    simple global brightness compensation for underexposed IR frames.
    """
    return np.clip(img.astype(np.int16) + int(value), 0, 255).astype(np.uint8)


def subtract_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Absolute difference: out = |img1 - img2|.

    Application: change / motion detection between two frames -- e.g. the
    difference between successive eye-region frames highlights the eyelid
    movement of a blink or a drowsiness-onset closure, while a static
    background/iris region cancels to near zero.
    """
    img2 = _match_shape(img2, img1)
    return cv2.absdiff(img1, img2)


def average_images(images: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise mean of N same-size images.

    Application: noise reduction by averaging N acquisitions of the
    (approximately) same scene -- independent zero-mean sensor noise
    averages toward zero as N grows (~1/sqrt(N) reduction in noise std),
    while the shared signal (the eye) is preserved.
    """
    if not images:
        raise ValueError("average_images got an empty list")
    stack = np.stack([im.astype(np.float64) for im in images], axis=0)
    return np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)


def make_noisy_copies(img: np.ndarray, n: int = 5, sigma: float = 15.0, seed: int | None = 0) -> list[np.ndarray]:
    """Synthesize `n` independently Gaussian-noisy copies of `img`.

    Used to demonstrate frame-averaging noise reduction on this dataset,
    which has no true multi-frame burst of the same eye. `seed` makes the
    demo reproducible.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        noise = rng.normal(0.0, sigma, img.shape)
        out.append(np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8))
    return out


# ---------------------------------------------------------------------------
# E. Comparison (brightness / contrast / noise, per technique per image)
# ---------------------------------------------------------------------------

def mean_intensity(img: np.ndarray) -> float:
    """Brightness proxy: mean pixel intensity."""
    return float(img.mean())


def rms_contrast(img: np.ndarray) -> float:
    """Contrast proxy: RMS / standard-deviation contrast (intensity spread)."""
    return float(img.std())


def noise_proxy(img: np.ndarray, ksize: int = 3) -> float:
    """Noise proxy: std-dev of the residual (img - median_filter(img)).

    A 3x3 median filter removes most real structure while leaving impulse /
    high-frequency sensor noise and enhancement-amplified graininess in the
    residual, so its std-dev tracks noise level rather than genuine detail.
    """
    med = cv2.medianBlur(img, ksize)
    residual = img.astype(np.float64) - med.astype(np.float64)
    return float(residual.std())


@dataclass
class TechniqueResult:
    name: str
    image: np.ndarray
    brightness: float
    contrast: float
    noise: float


# Default technique registry: name -> callable(img) -> img. Shared by the
# comparison routine and the CLI app so both stay in sync.
DEFAULT_TECHNIQUES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "original": lambda im: im,
    "negative": negative,
    "log": log_transform,
    "gamma_0.5": lambda im: gamma_correction(im, gamma=0.5),
    "gamma_1.5": lambda im: gamma_correction(im, gamma=1.5),
    "contrast_stretch": contrast_stretch,
    "hist_eq": equalize_histogram_cv,
}


def compare_techniques(
    img: np.ndarray,
    techniques: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> list[TechniqueResult]:
    """Apply each technique in `techniques` (default: DEFAULT_TECHNIQUES) to
    `img` and score brightness / contrast / noise for each output.
    """
    techniques = techniques or DEFAULT_TECHNIQUES
    results = []
    for name, fn in techniques.items():
        out = fn(img)
        results.append(
            TechniqueResult(
                name=name,
                image=out,
                brightness=mean_intensity(out),
                contrast=rms_contrast(out),
                noise=noise_proxy(out),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Dispatch table used by scripts/enhance_app.py (name -> function), so the
# CLI never needs an if/elif ladder duplicating this module's API.
# ---------------------------------------------------------------------------

TECHNIQUE_DISPATCH: dict[str, Callable[..., np.ndarray]] = {
    "negative": negative,
    "log": log_transform,
    "gamma": gamma_correction,
    "contrast_stretch": contrast_stretch,
    "hist_eq": equalize_histogram_cv,
    "hist_eq_manual": equalize_histogram_manual,
    "add_constant": add_constant,
}
