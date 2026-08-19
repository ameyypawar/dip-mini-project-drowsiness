#!/usr/bin/env python3
"""Generate all Task 3 (Image Enhancement in the Spatial Domain) figures and
metric tables into results/task3_*.png / results/task3_metrics.csv.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Uses only src.enhancement (new module, Task 3). Does not import or modify
src.preprocess / src.segment / src.spatial_filtering. Everything random is
seeded (seed=0) via src.enhancement.make_noisy_copies, so re-running this
script reproduces byte-identical figures/metrics.

Run: ./.venv/bin/python scripts/make_task3_figures.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import enhancement as enh  # noqa: E402

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The 3 images: spanning good lighting, poor lighting, eyeglasses+reflection
# (per manifest fields), preferring the physically larger crop within each
# condition bucket so pixel detail survives the INTER_NEAREST upscale.
IMAGES = [
    ("good_lighting", ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png",
     "open eye, no glasses, good lighting, no reflection (128x128, largest in bucket)"),
    ("poor_lighting", ROOT / "data/samples/alert/s0014_08297_0_0_1_1_0_02.png",
     "open eye, no glasses, poor lighting (130x130, largest no-glasses poor-lighting crop)"),
    ("glasses_reflection", ROOT / "data/samples/alert/s0035_00133_0_1_1_2_1_02.png",
     "open eye, eyeglasses + strong IR reflection (144x144, largest glasses+reflection crop)"),
]

# Same-subject open/closed pair (subject s0031) for a genuine (not synthetic)
# change/motion-detection subtraction demo.
SUBTRACT_PAIR = (
    ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png",   # open
    ROOT / "data/samples/drowsy/s0031_00171_1_0_0_2_1_02.png",  # same subject, closed
)

UPSCALE_MIN = 300  # smallest dimension after INTER_NEAREST upscale


def load_gray(path: Path, upscale_min: int = UPSCALE_MIN) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max(1, upscale_min // min(h, w))
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return img


def savefig(fig, name: str):
    out = RESULTS_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 1-3: intensity-transform grids (one per image)
# ---------------------------------------------------------------------------

def transform_grid_figure(label: str, img: np.ndarray, fig_num: int):
    techs = [
        ("Original", img),
        ("Negative", enh.negative(img)),
        ("Log", enh.log_transform(img)),
        ("Gamma 0.5", enh.gamma_correction(img, gamma=0.5)),
        ("Gamma 1.5", enh.gamma_correction(img, gamma=1.5)),
        ("Contrast stretch", enh.contrast_stretch(img)),
    ]
    fig, axes = plt.subplots(1, len(techs), figsize=(3 * len(techs), 3.2))
    for ax, (name, out) in zip(axes, techs):
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Figure {fig_num}: Intensity transformations -- {label}", y=1.04)
    savefig(fig, f"task3_transforms_{label}.png")


def histogram_grid_figure(label: str, img: np.ndarray, fig_num: int):
    techs = [
        ("Original", img),
        ("Negative", enh.negative(img)),
        ("Log", enh.log_transform(img)),
        ("Gamma 0.5", enh.gamma_correction(img, gamma=0.5)),
        ("Gamma 1.5", enh.gamma_correction(img, gamma=1.5)),
        ("Contrast stretch", enh.contrast_stretch(img)),
    ]
    fig, axes = plt.subplots(1, len(techs), figsize=(3 * len(techs), 2.6))
    for ax, (name, out) in zip(axes, techs):
        hist, edges = enh.compute_histogram(out)
        ax.bar(edges[:-1], hist, width=1.0, color="#444")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 255)
        ax.set_yticks([])
    fig.suptitle(f"Figure {fig_num}: Histograms before/after transformation -- {label}", y=1.05)
    savefig(fig, f"task3_histograms_{label}.png")


# ---------------------------------------------------------------------------
# Figure 4-6: histogram equalization (per image)
# ---------------------------------------------------------------------------

def histeq_figure(label: str, img: np.ndarray, fig_num: int) -> dict:
    eq_cv = enh.equalize_histogram_cv(img)
    eq_manual = enh.equalize_histogram_manual(img)
    max_abs_diff = int(np.abs(eq_cv.astype(int) - eq_manual.astype(int)).max())

    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(eq_cv, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Equalized (cv2.equalizeHist)")
    axes[0, 1].axis("off")

    h0, e0 = enh.compute_histogram(img)
    h1, e1 = enh.compute_histogram(eq_cv)
    axes[1, 0].bar(e0[:-1], h0, width=1.0, color="#444")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_title("Original histogram")
    axes[1, 1].bar(e1[:-1], h1, width=1.0, color="#444")
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_title("Equalized histogram")
    fig.suptitle(f"Figure {fig_num}: Histogram equalization -- {label}", y=1.02)
    savefig(fig, f"task3_histeq_{label}.png")

    return {
        "label": label,
        "orig_std": enh.rms_contrast(img),
        "eq_std": enh.rms_contrast(eq_cv),
        "orig_mean": enh.mean_intensity(img),
        "eq_mean": enh.mean_intensity(eq_cv),
        "manual_matches_cv_max_abs_diff": max_abs_diff,
    }


# ---------------------------------------------------------------------------
# Figure 7: image arithmetic -- averaging (noise reduction)
# ---------------------------------------------------------------------------

def averaging_figure(img: np.ndarray, fig_num: int) -> dict:
    noisy_copies_5 = enh.make_noisy_copies(img, n=5, sigma=20.0, seed=0)
    noisy_copies_20 = enh.make_noisy_copies(img, n=20, sigma=20.0, seed=0)
    avg5 = enh.average_images(noisy_copies_5)
    avg20 = enh.average_images(noisy_copies_20)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = [
        ("Clean original", img),
        ("Single noisy frame", noisy_copies_5[0]),
        ("Average of 5 noisy frames", avg5),
        ("Average of 20 noisy frames", avg20),
    ]
    for ax, (name, out) in zip(axes, panels):
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Figure {fig_num}: Image averaging for noise reduction (Gaussian sigma=20, seeded)", y=1.05)
    savefig(fig, "task3_arithmetic_averaging.png")

    return {
        "noise_single_frame": enh.noise_proxy(noisy_copies_5[0]),
        "noise_avg5": enh.noise_proxy(avg5),
        "noise_avg20": enh.noise_proxy(avg20),
    }


# ---------------------------------------------------------------------------
# Figure 8: image arithmetic -- subtraction (change / motion detection)
# ---------------------------------------------------------------------------

def subtraction_figure(fig_num: int) -> dict:
    open_img = load_gray(SUBTRACT_PAIR[0])
    closed_img = load_gray(SUBTRACT_PAIR[1])
    diff = enh.subtract_images(open_img, closed_img)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(open_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Frame A: eye open (subject s0031)", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(closed_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Frame B: eye closed (same subject)", fontsize=9)
    axes[1].axis("off")
    axes[2].imshow(diff, cmap="hot", vmin=0, vmax=255)
    axes[2].set_title("|A - B| (change map)", fontsize=9)
    axes[2].axis("off")
    fig.suptitle(f"Figure {fig_num}: Image subtraction for change / motion (blink) detection", y=1.05)
    savefig(fig, "task3_arithmetic_subtraction.png")

    return {"diff_mean": enh.mean_intensity(diff), "diff_max": int(diff.max())}


# ---------------------------------------------------------------------------
# Figure 9: image arithmetic -- addition (brightness / overlay)
# ---------------------------------------------------------------------------

def addition_figure(img: np.ndarray, fig_num: int) -> dict:
    bright_copy = enh.gamma_correction(img, gamma=0.5)
    blended = enh.add_images(img, bright_copy, alpha=0.5)
    const_added = enh.add_constant(img, value=40)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = [
        ("Original", img),
        ("Brightened copy (gamma 0.5)", bright_copy),
        ("Blend: 0.5*orig + 0.5*bright", blended),
        ("Original + 40 (add_constant)", const_added),
    ]
    for ax, (name, out) in zip(axes, panels):
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Figure {fig_num}: Image addition -- brightness boost / overlay compositing", y=1.05)
    savefig(fig, "task3_arithmetic_addition.png")

    return {
        "orig_mean": enh.mean_intensity(img),
        "blend_mean": enh.mean_intensity(blended),
        "const_add_mean": enh.mean_intensity(const_added),
    }


# ---------------------------------------------------------------------------
# Comparison table across the 3 images (brightness / contrast / noise per
# technique), plus a "most suitable technique" pick per image.
# ---------------------------------------------------------------------------

def comparison_table(images: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for label, img in images.items():
        results = enh.compare_techniques(img)
        base = next(r for r in results if r.name == "original")
        for r in results:
            contrast_gain = r.contrast / base.contrast if base.contrast else float("nan")
            noise_gain = r.noise / base.noise if base.noise else float("nan")
            score = contrast_gain / noise_gain if noise_gain else float("nan")
            rows.append({
                "image": label,
                "technique": r.name,
                "brightness": round(r.brightness, 2),
                "contrast": round(r.contrast, 2),
                "noise": round(r.noise, 2),
                "contrast_gain": round(contrast_gain, 3),
                "noise_gain": round(noise_gain, 3),
                "score_contrast_per_noise": round(score, 3),
            })
    return rows


def pick_winners(rows: list[dict]) -> dict[str, dict]:
    """Most suitable technique per image: excludes 'original', requires
    brightness to stay in [30, 225] (not clipped to near-black/near-white),
    then maximizes contrast-gain-per-noise-gain (score_contrast_per_noise).
    """
    winners = {}
    by_image: dict[str, list[dict]] = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r)
    for label, rs in by_image.items():
        candidates = [r for r in rs if r["technique"] != "original" and 30 <= r["brightness"] <= 225]
        if not candidates:
            candidates = [r for r in rs if r["technique"] != "original"]
        winners[label] = max(candidates, key=lambda r: r["score_contrast_per_noise"])
    return winners


def comparison_bar_figure(rows: list[dict], fig_num: int):
    labels = sorted({r["image"] for r in rows})
    techs = [t for t in enh.DEFAULT_TECHNIQUES if t != "original"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics = [("contrast", "Contrast (std-dev)"), ("brightness", "Brightness (mean)"), ("noise", "Noise proxy (residual std)")]
    x = np.arange(len(techs))
    width = 0.25
    for ax, (key, title) in zip(axes, metrics):
        for i, label in enumerate(labels):
            vals = [next(r[key] for r in rows if r["image"] == label and r["technique"] == t) for t in techs]
            ax.bar(x + i * width, vals, width, label=label)
        ax.set_xticks(x + width)
        ax.set_xticklabels(techs, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Figure {fig_num}: Brightness / contrast / noise across techniques, all 3 images", y=1.04)
    savefig(fig, "task3_comparison_bars.png")


def main():
    images = {label: load_gray(path) for label, path, _cond in IMAGES}

    fig_n = 1
    for label, img in images.items():
        transform_grid_figure(label, img, fig_n); fig_n += 1
        histogram_grid_figure(label, img, fig_n); fig_n += 1

    histeq_rows = []
    for label, img in images.items():
        histeq_rows.append(histeq_figure(label, img, fig_n)); fig_n += 1

    good_light_img = images["good_lighting"]
    avg_stats = averaging_figure(good_light_img, fig_n); fig_n += 1
    sub_stats = subtraction_figure(fig_n); fig_n += 1
    add_stats = addition_figure(good_light_img, fig_n); fig_n += 1

    rows = comparison_table(images)
    comparison_bar_figure(rows, fig_n); fig_n += 1

    winners = pick_winners(rows)

    # Write CSV (stdlib csv, no pandas).
    metrics_csv = RESULTS_DIR / "task3_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {metrics_csv}")

    print("\n=== Histogram equalization observations ===")
    for r in histeq_rows:
        print(r)

    print("\n=== Arithmetic stats ===")
    print("averaging:", avg_stats)
    print("subtraction:", sub_stats)
    print("addition:", add_stats)

    print("\n=== Winners (most suitable technique per image) ===")
    for label, w in winners.items():
        print(label, "->", w["technique"], w)

    return {
        "histeq_rows": histeq_rows,
        "avg_stats": avg_stats,
        "sub_stats": sub_stats,
        "add_stats": add_stats,
        "winners": winners,
        "rows": rows,
    }


if __name__ == "__main__":
    main()
