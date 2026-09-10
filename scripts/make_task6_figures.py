#!/usr/bin/env python3
"""Generate all Task 6 (Image Segmentation: Edge Detection and Thresholding)
figures and the comparison-metrics table into results/task6_*.png /
results/task6_metrics.csv.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Task 6 sub-task 1 says "use the best output obtained from Task 5" -- Task 5
was never done in this project, so the best Task 3 enhancement output
(histogram equalization, the winning technique on 2 of 3 test images per
results/task3_metrics.csv) is substituted as the input here instead. Same 3
test images as Task 3, loaded and upscaled the same way.

Uses src.edge_segmentation (new module, Task 6) + src.enhancement
(hist_eq only) + src.segment (via edge_segmentation's wrappers). Does not
modify any existing module. No pandas; CSV via stdlib csv.

Run: ./.venv/bin/python scripts/make_task6_figures.py
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

from src import enhancement as enh          # noqa: E402
from src import edge_segmentation as es     # noqa: E402

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Same 3 images as Task 3.
IMAGES = [
    ("good_lighting", ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png"),
    ("poor_lighting", ROOT / "data/samples/alert/s0014_08297_0_0_1_1_0_02.png"),
    ("glasses_reflection", ROOT / "data/samples/alert/s0035_00133_0_1_1_2_1_02.png"),
]

UPSCALE_MIN = 300  # matches make_task3_figures.py so pixel detail survives


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
# Metrics helpers
# ---------------------------------------------------------------------------

# Fixed absolute cutoff (0-255, on each operator's own max-normalized
# magnitude map -- see _normalize_mag in src.edge_segmentation) used to
# binarize every edge map for the component/noise metrics below. Otsu was
# tried first and rejected: gradient-magnitude histograms here are strongly
# unimodal (a huge near-zero bulk + a thin tail of true edges), so Otsu's
# between-class-variance split lands unpredictably far out in that tail --
# measured directly: on good_lighting it picked 13 for Roberts but 91 for
# Sobel/Prewitt (91 is past the 99th percentile, 42), collapsing those masks
# to 1-2 stray pixels instead of real edges. A single fixed cutoff, applied
# identically to every operator's own normalized map, avoids that
# instability and keeps the ratio comparable across operators.
EDGE_BINARIZE_THRESHOLD = 25


def edge_binary_mask(mag: np.ndarray) -> np.ndarray:
    """Fixed-cutoff binarization of an edge-magnitude map (bright-foreground
    convention: edges are the bright ridges) -- see EDGE_BINARIZE_THRESHOLD.
    """
    _, mask = cv2.threshold(mag, EDGE_BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def component_stats(mask: np.ndarray) -> tuple[int, int]:
    """Return (n_components, n_isolated_single_pixel_components) via 8-connectivity."""
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    n_components = n_labels - 1  # exclude background label 0
    if n_components == 0:
        return 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    n_isolated = int(np.sum(areas == 1))
    return n_components, n_isolated


def foreground_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


# ---------------------------------------------------------------------------
# Figures 1-3: original -> enhanced -> 4 edge maps, per image
# ---------------------------------------------------------------------------

def edge_grid_figure(label: str, orig: np.ndarray, enhanced: np.ndarray, edges: dict, fig_num: int):
    panels = [("Original", orig), ("Enhanced (hist_eq)", enhanced)] + list(edges.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for ax, (name, out) in zip(axes, panels):
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Figure {fig_num}: Edge detection -- {label}", y=1.04)
    savefig(fig, f"task6_edges_{label}.png")


# ---------------------------------------------------------------------------
# Figures 4-6: enhanced -> global / Otsu / adaptive binary masks, per image
# ---------------------------------------------------------------------------

def threshold_grid_figure(label: str, enhanced: np.ndarray, masks: dict, fig_num: int):
    panels = [("Enhanced (hist_eq)", enhanced)] + list(masks.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for ax, (name, out) in zip(axes, panels):
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Figure {fig_num}: Thresholding (dark-foreground = pupil/iris) -- {label}", y=1.04)
    savefig(fig, f"task6_thresholds_{label}.png")


# ---------------------------------------------------------------------------
# Figure 7: summary comparison bar chart (all 7 methods x 3 images)
# ---------------------------------------------------------------------------

def summary_bar_figure(rows: list[dict], method_order: list[str], fig_num: int):
    labels = sorted({r["image"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    metrics = [
        ("foreground_ratio", "Foreground / edge pixel ratio"),
        ("n_components", "Connected components (count)"),
        ("noise_proxy_isolated_px", "Noise proxy: isolated single-px components"),
    ]
    x = np.arange(len(method_order))
    width = 0.25
    for ax, (key, title) in zip(axes, metrics):
        for i, label in enumerate(labels):
            vals = [next(r[key] for r in rows if r["image"] == label and r["method"] == m) for m in method_order]
            ax.bar(x + i * width, vals, width, label=label)
        ax.set_xticks(x + width)
        ax.set_xticklabels(method_order, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Figure {fig_num}: Segmentation method comparison, all 3 images", y=1.04)
    savefig(fig, "task6_metrics_bars.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EDGE_METHODS = ["Roberts", "Prewitt", "Sobel", "Laplacian"]
THRESH_METHODS = ["Global", "Otsu", "Adaptive"]
ALL_METHODS = EDGE_METHODS + THRESH_METHODS


def main():
    rows = []
    fig_n = 1

    edge_fig_data = {}
    thresh_fig_data = {}

    for label, path in IMAGES:
        orig = load_gray(path)
        enhanced = enh.equalize_histogram_cv(orig)

        edges = {
            "Roberts": es.roberts_edge(enhanced),
            "Prewitt": es.prewitt_edge(enhanced),
            "Sobel": es.sobel_edge(enhanced),
            "Laplacian": es.laplacian_edge(enhanced),
        }
        edge_fig_data[label] = (orig, enhanced, edges)

        global_mask = es.global_threshold(enhanced)
        otsu_mask = es.otsu_wrapper(enhanced)
        adaptive_mask = es.adaptive_wrapper(enhanced)
        masks = {"Global": global_mask, "Otsu": otsu_mask, "Adaptive": adaptive_mask}
        thresh_fig_data[label] = (enhanced, masks)

        # Metrics: edge methods (Otsu-binarized magnitude map, no IoU -- not
        # a thresholding method).
        for name, mag in edges.items():
            bmask = edge_binary_mask(mag)
            n_comp, n_iso = component_stats(bmask)
            rows.append({
                "image": label,
                "method": name,
                "foreground_ratio": round(foreground_ratio(bmask), 4),
                "n_components": n_comp,
                "noise_proxy_isolated_px": n_iso,
                "iou_vs_otsu": "",
            })

        # Metrics: thresholding methods (IoU measured against this image's
        # own Otsu mask -- Otsu vs itself is trivially 1.0).
        for name, mask in masks.items():
            n_comp, n_iso = component_stats(mask)
            rows.append({
                "image": label,
                "method": name,
                "foreground_ratio": round(foreground_ratio(mask), 4),
                "n_components": n_comp,
                "noise_proxy_isolated_px": n_iso,
                "iou_vs_otsu": round(iou(mask, otsu_mask), 4),
            })

    # Figures 1-3 (edges) and 4-6 (thresholds), in image order.
    for label, _path in IMAGES:
        orig, enhanced, edges = edge_fig_data[label]
        edge_grid_figure(label, orig, enhanced, edges, fig_n); fig_n += 1
    for label, _path in IMAGES:
        enhanced, masks = thresh_fig_data[label]
        threshold_grid_figure(label, enhanced, masks, fig_n); fig_n += 1

    # Figure 7: summary bar chart.
    summary_bar_figure(rows, ALL_METHODS, fig_n); fig_n += 1

    # CSV.
    metrics_csv = RESULTS_DIR / "task6_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {metrics_csv}")

    # Aggregate (mean across the 3 images) per method, to pick winners.
    print("\n=== Per-method mean across 3 images ===")
    agg = {}
    for m in ALL_METHODS:
        m_rows = [r for r in rows if r["method"] == m]
        fr = np.mean([r["foreground_ratio"] for r in m_rows])
        nc = np.mean([r["n_components"] for r in m_rows])
        ni = np.mean([r["noise_proxy_isolated_px"] for r in m_rows])
        iou_vals = [r["iou_vs_otsu"] for r in m_rows if r["iou_vs_otsu"] != ""]
        iou_mean = np.mean(iou_vals) if iou_vals else None
        agg[m] = {"foreground_ratio": fr, "n_components": nc, "noise_proxy_isolated_px": ni, "iou_vs_otsu": iou_mean}
        print(m, agg[m])

    print("\n=== Per-image, per-method rows ===")
    for r in rows:
        print(r)

    return rows, agg


if __name__ == "__main__":
    main()
