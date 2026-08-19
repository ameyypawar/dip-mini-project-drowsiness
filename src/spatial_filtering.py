"""Task 4 -- Spatial Domain Filtering: smoothing, sharpening, high-boost.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

This module extends the pipeline with a spatial-filtering stage intended to
sit *between* raw capture and the Week-3 segmentation module
(`src.segment`): noisy/degraded eye crops go in, a denoised (and optionally
sharpened) crop comes out, and that crop is what segmentation should
actually threshold. Nothing here modifies `src.preprocess` or `src.segment`
-- this module is additive and is meant to be composed by the caller
(typically: denoise -> `src.preprocess.preprocess_pipeline` -> `src.segment`).

Contents:
  1. Noise injection    -- `add_salt_pepper_noise`, `add_gaussian_noise`
  2. Smoothing filters   -- `mean_filter`, `gaussian_filter`, `median_filter`
  3. Sharpening          -- `laplacian_sharpen`, `unsharp_mask`
  4. High-boost filtering -- `high_boost`
  5. Metrics             -- `psnr`, `ssim_score`, `edge_preservation_index`
  6. Timing harness      -- `time_filter`

All image filters take/return single-channel `uint8` `np.ndarray`, matching
the convention used throughout `src.preprocess` / `src.segment`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# 1. Noise injection
# ---------------------------------------------------------------------------

def add_salt_pepper_noise(
    img: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: int | None = 0,
) -> np.ndarray:
    """Add impulse (salt-and-pepper) noise to a grayscale uint8 image.

    `amount` is the fraction of *all* pixels replaced by an impulse
    (0.05 = 5%% of pixels touched). `salt_vs_pepper` splits that fraction
    between salt (255, `salt_vs_pepper` share) and pepper (0, the rest).
    Pixel *locations* for salt and pepper are drawn without replacement
    from disjoint index pools, so a pixel is never hit twice. `seed` is
    forwarded to a private `np.random.default_rng` so every call is
    reproducible independent of global numpy random state.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    n_pixels = h * w
    n_noisy = int(round(amount * n_pixels))
    if n_noisy == 0:
        return out

    flat_idx = rng.choice(n_pixels, size=n_noisy, replace=False)
    n_salt = int(round(n_noisy * salt_vs_pepper))
    salt_idx = flat_idx[:n_salt]
    pepper_idx = flat_idx[n_salt:]

    flat = out.reshape(-1)
    flat[salt_idx] = 255
    flat[pepper_idx] = 0
    return out.reshape(img.shape)


def add_gaussian_noise(
    img: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Add additive white Gaussian noise to a grayscale uint8 image.

    Noise ~ N(mean, sigma^2) is added in float space, then the result is
    clipped back to [0, 255] and cast to uint8 -- clipping (not wrapping)
    matches how a real sensor saturates. `seed` is forwarded to a private
    `np.random.default_rng` for reproducibility.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=sigma, size=img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Smoothing filters
# ---------------------------------------------------------------------------

def mean_filter(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean/average (box) filter: replace each pixel by the unweighted mean
    of its `k`x`k` neighbourhood. Thin wrapper over `cv2.blur`. Cheapest of
    the three smoothing filters but treats every neighbour equally,
    including outlier/impulse pixels -- it spreads impulse noise into a
    grey smear rather than removing it.
    """
    if k % 2 == 0 or k < 1:
        raise ValueError(f"k must be odd and >= 1, got {k}")
    return cv2.blur(img, (k, k))


def gaussian_filter(img: np.ndarray, k: int = 3, sigma: float = 0.0) -> np.ndarray:
    """Gaussian smoothing: `k`x`k` kernel weighted by a 2D Gaussian
    (weights fall off with distance from the centre pixel). Thin wrapper
    over `cv2.GaussianBlur`. `sigma=0` tells OpenCV to derive sigma from
    `k` (`0.3*((k-1)*0.5 - 1) + 0.8`, its standard heuristic). Like the
    mean filter it averages neighbours (so it still smears impulse spikes
    rather than rejecting them), but centre-weighting preserves edges
    somewhat better than a flat box for the same kernel size.
    """
    if k % 2 == 0 or k < 1:
        raise ValueError(f"k must be odd and >= 1, got {k}")
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)


def median_filter(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Median filter: replace each pixel by the median of its `k`x`k`
    neighbourhood. Thin wrapper over `cv2.medianBlur`. Nonlinear -- an
    isolated impulse (salt=255 or pepper=0) is an outlier in its local
    window and the median simply discards it, rather than averaging it in.
    This is the textbook impulse-noise filter; see the measured comparison
    in docs/task4_spatial_filtering.md for how much this matters in
    practice on this dataset.
    """
    if k % 2 == 0 or k < 1:
        raise ValueError(f"k must be odd and >= 1, got {k}")
    return cv2.medianBlur(img, k)


# ---------------------------------------------------------------------------
# 3. Sharpening
# ---------------------------------------------------------------------------

# 4-neighbour discrete Laplacian, standard "sharpen" kernel (centre 5,
# 4-connected neighbours -1, corners 0). Convolving with this kernel is
# algebraically `original - laplacian(original)` in one pass, since the
# discrete Laplacian itself is [[0,-1,0],[-1,4,-1],[0,-1,0]] (sign flipped
# depending on convention) and subtracting an edge-response from the
# original boosts the edges the Laplacian responded to.
LAPLACIAN_SHARPEN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]],
    dtype=np.float32,
)


def laplacian_sharpen(img: np.ndarray, kernel: np.ndarray = LAPLACIAN_SHARPEN_KERNEL) -> np.ndarray:
    """Sharpen via a single Laplacian-derived convolution kernel.

    Equivalent to `output = original - c * laplacian(original)` with the
    Laplacian sign folded into `kernel` (default: 4-connected, centre 5 /
    neighbours -1 -- see `LAPLACIAN_SHARPEN_KERNEL`). `cv2.filter2D` with
    `ddepth=-1` would clip intermediate values at uint8 range *before* the
    convolution finishes accumulating; instead we convolve in float64 and
    clip once at the end, so overshoot/undershoot at edges is handled
    correctly rather than silently truncated mid-computation.
    """
    filtered = cv2.filter2D(img.astype(np.float64), ddepth=-1, kernel=kernel)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def unsharp_mask(img: np.ndarray, k: int = 5, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking: output = original + amount * (original - blurred).

    Classic 3-step recipe: blur the original (`gaussian_filter`) to get a
    low-pass estimate, subtract it from the original to get the "mask"
    (high-frequency detail/edges), then add `amount` times that mask back
    onto the original. `amount=1.0` is the standard unsharp mask; higher
    `amount` exaggerates edges further. See `high_boost` for the
    generalization of this same idea with an independent gain `A` on the
    original term.
    """
    blurred = gaussian_filter(img, k=k, sigma=sigma).astype(np.float64)
    original = img.astype(np.float64)
    mask = original - blurred
    sharpened = original + amount * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. High-boost filtering
# ---------------------------------------------------------------------------

def high_boost(img: np.ndarray, A: float = 1.5, k: int = 5, sigma: float = 1.0) -> np.ndarray:
    """High-boost filter: output = A * original - blurred.

    Rearranged, `A*original - blurred = (A-1)*original + (original -
    blurred) = (A-1)*original + mask`, i.e. the original scaled by
    `(A-1)` plus the same high-frequency "mask" `unsharp_mask` uses --
    high-boost filtering is the strict generalization of unsharp masking,
    with an independent gain `A` on how much of the low-frequency original
    survives alongside the boosted-edge mask.

    Where this lands relative to `unsharp_mask`, worked out precisely
    (not just asserted -- verified against `unsharp_mask`'s output in the
    notebook): `unsharp_mask(amount=1)` computes `original + mask =
    2*original - blurred`. Matching that against `A*original - blurred`
    term-by-term needs **A=2**, not A=1 -- `high_boost(A=2)` is the exact
    pixel-for-pixel equivalent of standard unsharp masking (amount=1).
    **A=1** is the boundary case where the `(A-1)*original` term vanishes
    entirely: output = mask = original - blurred, i.e. a pure high-pass
    edge signal with *no* low-frequency original surviving -- visually a
    near-flat grey field with edge lines, not a recognizable sharpened eye
    crop. This matters for reading the `A` sweep below: A=1 is closer to
    edge detection than to sharpening; A>=1.5 is where the output starts
    looking like a plausibly sharpened image, and A=2 is where it
    converges on plain unsharp masking. Values 0 < A < 1 would suppress
    the original below unity weight (rarely useful; not swept here).
    """
    blurred = gaussian_filter(img, k=k, sigma=sigma).astype(np.float64)
    original = img.astype(np.float64)
    boosted = A * original - blurred
    return np.clip(boosted, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def psnr(clean: np.ndarray, test: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) of `test` against `clean`.
    Thin wrapper over `skimage.metrics.peak_signal_noise_ratio`,
    `data_range=255` (matches the Week-8 compression convention). Returns
    `inf` if `test` is pixel-identical to `clean`.
    """
    return float(peak_signal_noise_ratio(clean, test, data_range=255))


def ssim_score(clean: np.ndarray, test: np.ndarray) -> float:
    """Structural Similarity Index of `test` against `clean`, in [-1, 1]
    (1 = identical). Thin wrapper over
    `skimage.metrics.structural_similarity`, `data_range=255`. For images
    smaller than the default 7x7 SSIM window (some crops here are as
    small as ~55-60px, still well above 7), the default window is fine;
    we do not override `win_size`.
    """
    return float(structural_similarity(clean, test, data_range=255))


def edge_preservation_index(clean: np.ndarray, test: np.ndarray) -> float:
    """Edge-preservation index: Pearson correlation between the Sobel
    gradient-magnitude maps of `clean` and `test`.

    Formula: compute `Gx, Gy` via `cv2.Sobel` (3x3, float64) on each
    image, gradient magnitude `G = sqrt(Gx^2 + Gy^2)`, then return the
    Pearson correlation coefficient between the two flattened magnitude
    maps (`np.corrcoef`). 1.0 = filtered image's edge structure is
    perfectly linearly correlated with the clean original's (edges fully
    preserved, wherever they are and however their magnitude scaled);
    values well below 1 indicate edges were blurred away, shifted, or
    swamped by noise/impulse-response artifacts. This is a *structural*
    edge-fidelity measure, not a raw edge-strength ratio -- a uniformly
    dampened but correctly-located edge map still scores near 1.
    """
    def _grad_mag(im: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(im.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(im.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    g_clean = _grad_mag(clean).ravel()
    g_test = _grad_mag(test).ravel()

    if np.std(g_clean) == 0 or np.std(g_test) == 0:
        return 1.0 if np.allclose(g_clean, g_test) else 0.0

    corr = np.corrcoef(g_clean, g_test)[0, 1]
    return float(corr)


@dataclass(frozen=True)
class QualityMetrics:
    """Bundle of the three fidelity metrics, all measured against the
    clean, noise-free original."""

    psnr: float
    ssim: float
    edge_preservation: float


def evaluate_quality(clean: np.ndarray, test: np.ndarray) -> QualityMetrics:
    """Compute `QualityMetrics` (PSNR, SSIM, edge-preservation index) of
    `test` against `clean` in one call."""
    return QualityMetrics(
        psnr=psnr(clean, test),
        ssim=ssim_score(clean, test),
        edge_preservation=edge_preservation_index(clean, test),
    )


# ---------------------------------------------------------------------------
# 6. Timing harness
# ---------------------------------------------------------------------------

def time_filter(fn: Callable[[np.ndarray], np.ndarray], img: np.ndarray, n_reps: int = 2000) -> float:
    """Average per-call wall-clock cost of `fn(img)`, in microseconds.

    A single `time.time()` around one call to a filter on a ~90x90 image
    measures scheduler/timer noise, not the filter's actual cost (the call
    itself is sub-millisecond). This harness instead runs `fn(img)`
    `n_reps` times back-to-back inside one `time.perf_counter()` bracket
    (perf_counter, not time.time, for monotonic sub-microsecond
    resolution) and divides -- the per-call noise averages out over
    enough repetitions. One untimed warm-up call is made first so
    first-call effects (e.g. lazy allocation) don't bias the measurement.
    Returns microseconds per call.
    """
    fn(img)  # warm-up, untimed
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn(img)
    elapsed = time.perf_counter() - t0
    return (elapsed / n_reps) * 1e6
