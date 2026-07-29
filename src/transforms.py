"""Week-5 transforms: DFT/DCT frequency-domain analysis + Hough circle iris fitting.

Reused by Week 6 (feature join) and Week 8 (DCT/JPEG compression). All
functions operate on the canonical 64x64 preprocessed crop produced by
`src.preprocess.preprocess_pipeline` -- a single grayscale uint8
`np.ndarray` -- unless stated otherwise.

Two independent DIP topics live here:
  - Fourier / Cosine transforms -- frequency-domain sharpness (blur) metrics,
    compared against the classical spatial-domain Laplacian-variance baseline.
  - Hough circle transform -- iris fitting. An open eye exposes a roughly
    circular iris; a closed eye does not, so "no circle found" is itself a
    feature, not a failure -- see `detect_iris_circle` and
    docs/week5_transforms.md for the measured open-vs-closed detection gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Fourier (DFT)
# ---------------------------------------------------------------------------

def dft_shifted(img: np.ndarray) -> np.ndarray:
    """Complex 2D DFT of `img` (cast to float32), zero-frequency shifted to center."""
    f = np.fft.fft2(img.astype(np.float32))
    return np.fft.fftshift(f)


def magnitude_spectrum(img: np.ndarray) -> np.ndarray:
    """Log-scaled magnitude spectrum for display: log(1 + |F|), normalized to uint8 [0, 255].

    Log scaling is standard for DFT display -- the DC/low-frequency
    component otherwise dwarfs everything else on a linear scale.
    """
    fshift = dft_shifted(img)
    mag = np.log1p(np.abs(fshift))
    mag -= mag.min()
    peak = mag.max()
    if peak > 0:
        mag = mag / peak
    return (mag * 255).astype(np.uint8)


def _radial_grid(h: int, w: int) -> np.ndarray:
    """Distance (px, float) of every pixel from the shifted-spectrum center (h//2, w//2)."""
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def high_freq_energy_ratio(img: np.ndarray, radius: int = 8) -> float:
    """Fraction of spectral energy (|F|^2) lying outside a central disk of `radius` px.

    `radius=8` is the chosen default for the 64x64 canonical crop -- see
    docs/week5_transforms.md for the sweep that picked it (radius 6-10 all
    separate sharp/blurred comparably; 8 is the middle of that plateau and
    keeps the ratio away from both extremes of 0/1). Sharp, detailed images
    keep proportionally more energy outside the low-frequency core; blur
    (a spatial-domain low-pass) suppresses high frequencies, so blurred
    images score lower.
    """
    fshift = dft_shifted(img)
    power = np.abs(fshift) ** 2
    r = _radial_grid(*img.shape[:2])
    total = power.sum()
    if total == 0:
        return 0.0
    outside = power[r > radius].sum()
    return float(outside / total)


def radial_energy_profile(img: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Radially-averaged power spectrum, binned into `n_bins` concentric annuli.

    Bin `i` covers radius in [i * step, (i + 1) * step) px from the spectrum
    center, where step = max_radius / n_bins and max_radius = half the crop
    diagonal. Each bin holds the *mean* power of pixels falling in it (mean,
    not sum, so bin values are comparable despite the outer annuli covering
    far more pixels than the inner ones). `n_bins=8` keeps this a compact,
    reusable feature vector (see `build_feature_table`) rather than one bin
    per integer pixel radius.
    """
    h, w = img.shape[:2]
    fshift = dft_shifted(img)
    power = np.abs(fshift) ** 2
    r = _radial_grid(h, w)
    max_radius = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    step = max_radius / n_bins
    profile = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        lo, hi = i * step, (i + 1) * step
        mask = (r >= lo) & (r < hi) if i < n_bins - 1 else (r >= lo)
        if mask.any():
            profile[i] = power[mask].mean()
    return profile


# ---------------------------------------------------------------------------
# Cosine (DCT)
# ---------------------------------------------------------------------------

def dct2(img: np.ndarray) -> np.ndarray:
    """2D DCT-II of `img`, computed on float32 (`cv2.dct`'s required dtype)."""
    return cv2.dct(img.astype(np.float32))


def dct_energy_compaction(img: np.ndarray, k: int) -> float:
    """Fraction of total DCT energy (sum of coef^2) held in the top-left k x k block.

    Top-left coefficients are the lowest-frequency ones. A high fraction
    means most information is captured by a handful of low-frequency
    coefficients -- exactly the property JPEG block quantization (Week 8)
    exploits: coefficients outside a small top-left block can be dropped
    with little perceptual loss.
    """
    d = dct2(img)
    total = float((d ** 2).sum())
    if total == 0:
        return 0.0
    k = min(k, d.shape[0], d.shape[1])
    block = float((d[:k, :k] ** 2).sum())
    return block / total


def dct_energy_compaction_curve(img: np.ndarray, ks: Optional[list[int]] = None) -> dict[int, float]:
    """`dct_energy_compaction` evaluated at each k in `ks` (default: 1..min(H, W)).

    Used to plot the energy-retained-vs-coefficients-kept curve (Week 5
    notebook) and directly reusable for the Week 8 JPEG quality sweep.
    """
    h, w = img.shape[:2]
    if ks is None:
        ks = list(range(1, min(h, w) + 1))
    return {k: dct_energy_compaction(img, k) for k in ks}


# ---------------------------------------------------------------------------
# Classical spatial-domain blur baseline
# ---------------------------------------------------------------------------

def laplacian_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian -- classical spatial-domain blur/sharpness metric.

    Low variance -> few strong second-derivative edge responses -> likely
    blurred. Computed on float64 (`cv2.CV_64F`) so negative Laplacian
    responses aren't clipped the way they would be on a uint8 output.
    """
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


# ---------------------------------------------------------------------------
# Hough circle transform -- iris fitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoughConfig:
    """`cv2.HoughCircles` params tuned for the 64x64 canonical crop.

    See docs/week5_transforms.md for the two-stage grid search: an initial
    sweep on the curated 40-image sample (n=20/class, too small -- picked
    configs that turned out to be overfit to sampling noise), then a
    ~3,150-config re-search validated on an 800-image stratified draw
    (400 open / 400 closed) from the working set, scored by
    open-detection-rate minus closed-detection-rate. This config was the
    highest-gap candidate with the higher absolute detection rates among
    near-tied top results (gap=0.2575, open=0.5925, closed=0.3350).
    """

    dp: float = 1.0
    min_dist: int = 64          # crop height -- at most one iris expected per 64x64 crop
    param1: int = 100           # Canny high threshold for HoughCircles' internal edge map
    param2: int = 12            # accumulator threshold -- lower = more permissive detection
    min_radius: int = 6
    max_radius: int = 14
    median_blur_ksize: int = 5  # light denoise before edge detection; 0/1 disables


DEFAULT_HOUGH = HoughConfig()


@dataclass(frozen=True)
class IrisCircle:
    """Result of one `detect_iris_circle` call.

    `found=False` on a closed (or otherwise non-circular) eye is an
    expected, informative outcome, not an error -- see module docstring.
    `n_circles_detected` is the count of candidate circles HoughCircles
    returned before picking the best (highest-accumulator-score) one; it is
    used as a confidence proxy since OpenCV's Python API does not expose
    raw accumulator votes.
    """

    found: bool
    cx: float
    cy: float
    radius: float
    n_circles_detected: int


def detect_iris_circle(img: np.ndarray, cfg: HoughConfig = DEFAULT_HOUGH) -> IrisCircle:
    """Fit the best circle (by HoughCircles' accumulator ranking) to `img`.

    HoughCircles returns candidates ordered by accumulator score,
    strongest first, so `circles[0]` is the best fit.
    """
    src = img
    if cfg.median_blur_ksize > 1:
        src = cv2.medianBlur(src, cfg.median_blur_ksize)

    circles = cv2.HoughCircles(
        src,
        cv2.HOUGH_GRADIENT,
        dp=cfg.dp,
        minDist=cfg.min_dist,
        param1=cfg.param1,
        param2=cfg.param2,
        minRadius=cfg.min_radius,
        maxRadius=cfg.max_radius,
    )
    if circles is None:
        return IrisCircle(found=False, cx=0.0, cy=0.0, radius=0.0, n_circles_detected=0)

    candidates = circles[0]
    cx, cy, r = candidates[0]
    return IrisCircle(
        found=True,
        cx=float(cx),
        cy=float(cy),
        radius=float(r),
        n_circles_detected=int(len(candidates)),
    )


# ---------------------------------------------------------------------------
# Feature table (Week 6 join key: filename)
# ---------------------------------------------------------------------------

# DCT top-left block sizes to report as energy-compaction features. 8 is the
# JPEG block size, included deliberately to set up Week 8.
DCT_KS = (2, 4, 8, 16)

FEATURE_COLUMNS = (
    ["filename", "split", "eye_state"]
    + ["dft_high_freq_ratio"]
    + [f"dft_radial_bin{i}" for i in range(8)]
    + [f"dct_compaction_k{k}" for k in DCT_KS]
    + ["laplacian_variance"]
    + ["hough_found", "hough_cx", "hough_cy", "hough_radius", "hough_n_circles"]
)


def compute_row_features(img: np.ndarray, hough_cfg: HoughConfig = DEFAULT_HOUGH) -> dict:
    """All numeric transform features for one preprocessed 64x64 crop, keyed by column name."""
    row: dict[str, float] = {"dft_high_freq_ratio": high_freq_energy_ratio(img)}

    profile = radial_energy_profile(img, n_bins=8)
    for i, v in enumerate(profile):
        row[f"dft_radial_bin{i}"] = float(v)

    for k in DCT_KS:
        row[f"dct_compaction_k{k}"] = dct_energy_compaction(img, k)

    row["laplacian_variance"] = laplacian_variance(img)

    circle = detect_iris_circle(img, hough_cfg)
    row["hough_found"] = int(circle.found)
    row["hough_cx"] = circle.cx
    row["hough_cy"] = circle.cy
    row["hough_radius"] = circle.radius
    row["hough_n_circles"] = circle.n_circles_detected

    return row


def build_feature_table(
    working_set_csv: str,
    out_csv: str,
    repo_root: str = ".",
    hough_cfg: HoughConfig = DEFAULT_HOUGH,
) -> tuple[int, float]:
    """Compute transform features for every row of `working_set_csv`, write `out_csv`.

    Each input row's image is loaded from `<repo_root>/<source_relpath>`,
    run through `src.preprocess.preprocess_pipeline` (Week-2 canonical
    config), then all transform features are computed. Output columns:
    `FEATURE_COLUMNS` (filename, split, eye_state first, so Week 6 can join
    on `filename`). Returns (n_rows_written, wall_seconds).
    """
    import csv
    import time
    from pathlib import Path

    from src.preprocess import DEFAULT_CONFIG, preprocess_pipeline

    root = Path(repo_root)
    t0 = time.time()
    n_written = 0

    with open(working_set_csv, newline="") as f_in, open(out_csv, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()

        for row in reader:
            img_path = root / row["source_relpath"]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"could not read image: {img_path}")

            enhanced = preprocess_pipeline(img, DEFAULT_CONFIG)
            features = compute_row_features(enhanced, hough_cfg)

            out_row = {
                "filename": row["filename"],
                "split": row["split"],
                "eye_state": row["eye_state"],
            }
            out_row.update(features)
            writer.writerow(out_row)
            n_written += 1

    elapsed = time.time() - t0
    return n_written, elapsed


def main() -> None:
    n, elapsed = build_feature_table(
        working_set_csv="data/working_set.csv",
        out_csv="data/features/transform_features.csv",
    )
    print(f"Wrote data/features/transform_features.csv ({n} rows) in {elapsed:.1f}s "
          f"({elapsed / max(n, 1) * 1000:.2f} ms/image)")


if __name__ == "__main__":
    main()
