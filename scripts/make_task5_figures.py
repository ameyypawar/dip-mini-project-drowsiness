#!/usr/bin/env python3
"""Generate all Task 5 (Sharpening Filters, Filter Comparison, and Module
Integration) figures and comparison-metrics CSVs into results/task5_*.png /
results/task5_*.csv.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Reuses (does not reimplement):
  - src.spatial_filtering (Task 4): noise injection, mean/Gaussian/median
    smoothing, Laplacian sharpening, high-boost filtering, PSNR/SSIM/
    edge-preservation.
  - src.enhancement (Task 3): histogram equalization, noise_proxy.
  - src.sharpening (Task 5, new): gradient-based sharpening, MSE,
    noise-amplification ratio, dispatch tables.

Same 3 test images as Tasks 3/4/6/7. No pandas; CSV via stdlib csv.

Run: ./.venv/bin/python scripts/make_task5_figures.py
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import enhancement as enh              # noqa: E402
from src import spatial_filtering as sf         # noqa: E402
from src import sharpening as sh                # noqa: E402

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMAGES = [
    ("good_lighting", ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png"),
    ("poor_lighting", ROOT / "data/samples/alert/s0014_08297_0_0_1_1_0_02.png"),
    ("glasses_reflection", ROOT / "data/samples/alert/s0035_00133_0_1_1_2_1_02.png"),
]

UPSCALE_MIN = 300  # matches make_task3/6/7_figures.py so pixel detail survives
NOISE_AMOUNT = 0.05  # salt-and-pepper fraction, matches Task 4's default
HIGH_BOOST_A = 2.0   # == unsharp_mask(amount=1), verified in Task 4


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


def write_csv(rows: list[dict], name: str):
    out = RESULTS_DIR / name
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


# ===========================================================================
# Part 1 (sub-task 1): sharpening filters on CLEAN images -- Laplacian,
# gradient-based (new), high-boost. Original vs each, + noise amplification.
# ===========================================================================

def sharpening_figures():
    fig_n = 1
    metric_rows = []
    amp_by_method = {"laplacian": [], "gradient": [], "high_boost": []}

    for label, path in IMAGES:
        clean = load_gray(path)
        methods = {
            "Laplacian": sh.laplacian_sharpen(clean),
            "Gradient (Sobel)": sh.gradient_sharpen(clean, ksize=3, amount=1.0),
            f"High-Boost (A={HIGH_BOOST_A:g})": sh.high_boost(clean, A=HIGH_BOOST_A, k=5, sigma=1.0),
        }
        key_map = {"Laplacian": "laplacian", "Gradient (Sobel)": "gradient",
                   f"High-Boost (A={HIGH_BOOST_A:g})": "high_boost"}

        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        axes[0].imshow(clean, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")
        for ax, (name, out) in zip(axes[1:], methods.items()):
            m = sh.mse(clean, out)
            p = sf.psnr(clean, out)
            amp = sh.noise_amplification_ratio(clean, out)
            ax.imshow(out, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name}\nMSE={m:.1f}  PSNR={p:.1f}dB\nnoise x{amp:.2f}", fontsize=8.3)
            ax.axis("off")
            metric_rows.append({
                "image": label,
                "method": key_map[name],
                "mse": round(m, 4),
                "psnr_db": round(p, 4),
                "ssim": round(sf.ssim_score(clean, out), 4),
                "edge_preservation": round(sf.edge_preservation_index(clean, out), 4),
                "noise_amplification_ratio": round(amp, 4),
            })
            amp_by_method[key_map[name]].append(amp)
        fig.suptitle(f"Figure {fig_n}: Sharpening filters (clean input) -- {label}", y=1.06)
        savefig(fig, f"task5_sharpening_{label}.png")
        fig_n += 1

    write_csv(metric_rows, "task5_sharpening_metrics.csv")

    # Figure 4: noise amplification bar chart, 3 methods x 3 images.
    labels = [lbl for lbl, _ in IMAGES]
    methods_order = ["laplacian", "gradient", "high_boost"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(methods_order):
        ax.bar(x + i * width, amp_by_method[m], width, label=m)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="no change (x1.0)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Noise amplification ratio (sharpened / clean)")
    ax.set_title(f"Figure {fig_n}: Measured noise amplification by sharpening method")
    ax.legend(fontsize=8)
    savefig(fig, "task5_noise_amplification_bars.png")
    fig_n += 1

    return fig_n, metric_rows, amp_by_method


# ===========================================================================
# Part 2 (sub-task 2): comparison table -- 3 images x {Mean, Gaussian,
# Median, Laplacian, High-Boost} on salt-and-pepper-noisy input, MSE + PSNR.
# ===========================================================================

FILTER_LABELS = ["Mean 3x3", "Gaussian 3x3", "Median 3x3", "Laplacian", f"High-Boost (A={HIGH_BOOST_A:g})"]


def apply_comparison_filter(name: str, noisy: np.ndarray) -> np.ndarray:
    if name == "Mean 3x3":
        return sf.mean_filter(noisy, k=3)
    if name == "Gaussian 3x3":
        return sf.gaussian_filter(noisy, k=3)
    if name == "Median 3x3":
        return sf.median_filter(noisy, k=3)
    if name == "Laplacian":
        return sf.laplacian_sharpen(noisy)
    if name == f"High-Boost (A={HIGH_BOOST_A:g})":
        return sf.high_boost(noisy, A=HIGH_BOOST_A, k=5, sigma=1.0)
    raise ValueError(name)


def comparison_figures(fig_n: int):
    all_rows = []
    for label, path in IMAGES:
        clean = load_gray(path)
        noisy = sf.add_salt_pepper_noise(clean, amount=NOISE_AMOUNT, seed=0)

        outputs = {name: apply_comparison_filter(name, noisy) for name in FILTER_LABELS}

        panels = [("Original", clean), ("Noisy (S&P 5%)", noisy)] + list(outputs.items())
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))
        for ax, (name, out) in zip(axes, panels):
            ax.imshow(out, cmap="gray", vmin=0, vmax=255)
            ax.set_title(name, fontsize=8.6)
            ax.axis("off")
        fig.suptitle(f"Figure {fig_n}: Filter comparison (Mean/Gaussian/Median/Laplacian/High-Boost) -- {label}", y=1.05)
        savefig(fig, f"task5_filters_{label}.png")
        fig_n += 1

        for name, out in outputs.items():
            all_rows.append({
                "image": label,
                "filter": name,
                "mse": round(sh.mse(clean, out), 4),
                "psnr_db": round(sf.psnr(clean, out), 4),
                "ssim": round(sf.ssim_score(clean, out), 4),
                "edge_preservation": round(sf.edge_preservation_index(clean, out), 4),
            })

    write_csv(all_rows, "task5_filter_comparison.csv")

    # Figure: PSNR bar chart, 5 filters x 3 images.
    labels = [lbl for lbl, _ in IMAGES]
    x = np.arange(len(FILTER_LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, label in enumerate(labels):
        vals = [next(r["psnr_db"] for r in all_rows if r["image"] == label and r["filter"] == m) for m in FILTER_LABELS]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(FILTER_LABELS, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("PSNR (dB) vs. clean original")
    ax.set_title(f"Figure {fig_n}: PSNR comparison, all 5 filters x 3 images (salt-and-pepper noisy input)")
    ax.legend(fontsize=8)
    savefig(fig, "task5_comparison_bars.png")
    fig_n += 1

    return fig_n, all_rows


# ===========================================================================
# Part 3 (sub-task 5): enhancement + filtering chain -- order matters.
# enhance-then-filter vs filter-then-enhance, on one image.
# ===========================================================================

CHAIN_IMAGE_LABEL = "poor_lighting"


def chain_demo(fig_n: int):
    label, path = next((lbl, p) for lbl, p in IMAGES if lbl == CHAIN_IMAGE_LABEL)
    clean = load_gray(path)
    noisy = sf.add_salt_pepper_noise(clean, amount=NOISE_AMOUNT, seed=0)

    enhance_then_filter = sf.median_filter(enh.equalize_histogram_cv(noisy), k=3)
    filter_then_enhance = enh.equalize_histogram_cv(sf.median_filter(noisy, k=3))

    rows = [
        {"order": "noisy_baseline", "stage": "noisy (no processing)",
         "mse": round(sh.mse(clean, noisy), 4), "psnr_db": round(sf.psnr(clean, noisy), 4),
         "ssim": round(sf.ssim_score(clean, noisy), 4)},
        {"order": "enhance_then_filter", "stage": "hist_eq -> median3x3",
         "mse": round(sh.mse(clean, enhance_then_filter), 4), "psnr_db": round(sf.psnr(clean, enhance_then_filter), 4),
         "ssim": round(sf.ssim_score(clean, enhance_then_filter), 4)},
        {"order": "filter_then_enhance", "stage": "median3x3 -> hist_eq",
         "mse": round(sh.mse(clean, filter_then_enhance), 4), "psnr_db": round(sf.psnr(clean, filter_then_enhance), 4),
         "ssim": round(sf.ssim_score(clean, filter_then_enhance), 4)},
    ]
    write_csv(rows, "task5_chain_metrics.csv")

    winner = "filter_then_enhance" if rows[2]["psnr_db"] > rows[1]["psnr_db"] else "enhance_then_filter"
    print(f"\nChain order winner (higher PSNR): {winner}")
    for r in rows:
        print(r)

    panels = [
        ("Clean original", clean),
        ("Noisy (S&P 5%)", noisy),
        (f"Enhance->Filter\nPSNR={rows[1]['psnr_db']:.2f}dB", enhance_then_filter),
        (f"Filter->Enhance\nPSNR={rows[2]['psnr_db']:.2f}dB", filter_then_enhance),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (name, out) in zip(axes, panels):
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Figure {fig_n}: Enhancement + filtering chain order comparison -- {label}", y=1.07)
    savefig(fig, "task5_chain_demo.png")
    fig_n += 1

    return fig_n, rows, winner


# ===========================================================================
# Part 4: pipeline block diagram (Acquisition -> Enhancement -> Filtering ->
# Segmentation), matplotlib boxes + arrows.
# ===========================================================================

def _box(ax, cx, cy, w, h, text, sub, facecolor):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0,rounding_size=0.05",
                          linewidth=1.4, edgecolor="#333333", facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy + 0.10, text, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=3)
    ax.text(cx, cy - 0.16, sub, ha="center", va="center", fontsize=7.6, zorder=3)


def _arrow(ax, x1, x2, y, label):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14,
                                  linewidth=1.4, color="#333333", shrinkA=2, shrinkB=2, zorder=2))
    ax.text((x1 + x2) / 2, y + 0.14, label, ha="center", va="bottom", fontsize=7.4, style="italic")


def pipeline_diagram(fig_n: int):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        (1.6, "Acquisition", "IR eye-crop frame\n(Task 2, src.acquisition)", "#e0f3e0"),
        (4.6, "Enhancement", "gamma / hist-eq / CLAHE\n(Task 3, src.enhancement)", "#dbe8fb"),
        (7.6, "Filtering", "median denoise + sharpen\n(Task 4/5, src.spatial_filtering,\nsrc.sharpening)", "#fdf1cf"),
        (10.6, "Segmentation", "Otsu + region growing\n(Task 6/7, src.segment,\nsrc.region_growing)", "#f6dcdc"),
    ]
    y = 1.55
    for cx, title, sub, color in stages:
        _box(ax, cx, y, 2.5, 1.35, title, sub, color)
    for (x1, _, _, _), (x2, _, _, _) in zip(stages[:-1], stages[1:]):
        _arrow(ax, x1 + 1.25, x2 - 1.25, y, "")

    ax.text(6.1, 0.15,
            "Figure {}: Integrated module pipeline -- output of each stage is the input of the next.".format(fig_n),
            ha="center", fontsize=9.5)
    savefig(fig, "task5_pipeline_diagram.png")
    return fig_n + 1


# ===========================================================================
# Main
# ===========================================================================

def main():
    fig_n, sharp_rows, amp = sharpening_figures()
    fig_n, comp_rows = comparison_figures(fig_n)
    fig_n, chain_rows, winner = chain_demo(fig_n)
    fig_n = pipeline_diagram(fig_n)

    print("\n=== Sharpening noise amplification, mean across 3 images ===")
    for m, vals in amp.items():
        print(m, "mean amplification x", round(float(np.mean(vals)), 3))

    print("\n=== Filter comparison, mean PSNR across 3 images ===")
    for name in FILTER_LABELS:
        vals = [r["psnr_db"] for r in comp_rows if r["filter"] == name]
        print(name, "mean PSNR", round(float(np.mean(vals)), 3), "dB")

    print(f"\nTotal figures generated: {fig_n - 1}")


if __name__ == "__main__":
    main()
