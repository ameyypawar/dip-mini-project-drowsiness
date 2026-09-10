"""Task 6 -- Image Segmentation: Edge Detection and Thresholding.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Operates on the same grayscale `uint8` `np.ndarray` convention as
`src.enhancement` / `src.segment`. Additive and self-contained: does not
modify `src.segment`, only wraps it.

Contents:
  A. Edge detection  -- roberts_edge, prewitt_edge (explicit kernels via
                         cv2.filter2D), sobel_edge, laplacian_edge (OpenCV
                         builtins)
  B. Thresholding     -- global_threshold (own fixed-value implementation),
                         otsu_wrapper / adaptive_wrapper (thin wrappers
                         reusing src.segment so the Otsu/adaptive logic is
                         not duplicated)

Foreground convention note: the edge operators return a bright-ridge
magnitude map (edges are bright, flat regions are dark) -- the natural
convention for gradient/Laplacian magnitude. The threshold functions instead
follow `src.segment`'s dark-foreground convention (foreground = pupil/iris,
darker than sclera/skin in these near-IR eye crops), via `THRESH_BINARY_INV`,
so `global_threshold` is directly comparable to `otsu_wrapper` /
`adaptive_wrapper`.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.segment import adaptive_threshold, otsu_threshold


# ---------------------------------------------------------------------------
# A. Edge detection
# ---------------------------------------------------------------------------

def _normalize_mag(mag: np.ndarray) -> np.ndarray:
    """Scale a non-negative float gradient-magnitude map to uint8 [0,255]."""
    peak = mag.max()
    if peak <= 0:
        return np.zeros_like(mag, dtype=np.uint8)
    return np.clip(mag / peak * 255.0, 0, 255).astype(np.uint8)


# Roberts cross-gradient kernels (2x2) -- the smallest possible gradient
# operator, differencing diagonal neighbour pairs. Very sensitive to noise
# since it averages over only 2 pixels per direction.
_ROBERTS_GX = np.array([[1, 0], [0, -1]], dtype=np.float64)
_ROBERTS_GY = np.array([[0, 1], [-1, 0]], dtype=np.float64)


def roberts_edge(img: np.ndarray) -> np.ndarray:
    """Roberts cross-gradient edge magnitude, explicit 2x2 kernels via cv2.filter2D."""
    img_f = img.astype(np.float64)
    gx = cv2.filter2D(img_f, cv2.CV_64F, _ROBERTS_GX)
    gy = cv2.filter2D(img_f, cv2.CV_64F, _ROBERTS_GY)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return _normalize_mag(mag)


# Prewitt kernels (3x3) -- unweighted 3-row/column difference, a wider
# (less noise-sensitive than Roberts) first-derivative operator.
_PREWITT_GX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
_PREWITT_GY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)


def prewitt_edge(img: np.ndarray) -> np.ndarray:
    """Prewitt edge magnitude, explicit 3x3 kernels via cv2.filter2D."""
    img_f = img.astype(np.float64)
    gx = cv2.filter2D(img_f, cv2.CV_64F, _PREWITT_GX)
    gy = cv2.filter2D(img_f, cv2.CV_64F, _PREWITT_GY)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return _normalize_mag(mag)


def sobel_edge(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel edge magnitude via OpenCV's builtin `cv2.Sobel` (center-weighted
    3x3 first-derivative kernel -- more noise-resistant than Prewitt because
    of the extra center-row/column weighting).
    """
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return _normalize_mag(mag)


def laplacian_edge(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Laplacian (second-derivative) edge magnitude via OpenCV's builtin
    `cv2.Laplacian` -- isotropic, but far more noise-sensitive than the
    first-derivative operators since differentiating twice amplifies
    high-frequency noise.
    """
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    return _normalize_mag(np.abs(lap))


# ---------------------------------------------------------------------------
# B. Thresholding
# ---------------------------------------------------------------------------

def global_threshold(img: np.ndarray, thresh: int = 127) -> np.ndarray:
    """Naive fixed-value global threshold, dark-foreground convention.

    Every pixel darker than `thresh` becomes foreground (255), regardless of
    the image's own histogram -- unlike Otsu, this value is not data-driven,
    so it is only as good as the operator's guess of a suitable split point.
    Used as the baseline against which Otsu and adaptive thresholding are
    compared.
    """
    _, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def otsu_wrapper(img: np.ndarray) -> np.ndarray:
    """Thin wrapper around `src.segment.otsu_threshold` (Task 3/Week-3 module) -- returns just the binary mask."""
    _, mask = otsu_threshold(img)
    return mask


def adaptive_wrapper(img: np.ndarray, block_size: int = 15, C: int = 3) -> np.ndarray:
    """Thin wrapper around `src.segment.adaptive_threshold` (Gaussian-weighted local mean)."""
    return adaptive_threshold(img, method="gaussian", block_size=block_size, C=C)
