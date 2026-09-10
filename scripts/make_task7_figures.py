#!/usr/bin/env python3
"""Generate all Task 7 (Region Growing, Connectivity Comparison, and Region
Splitting/Merging) figures and metrics CSVs into results/task7_*.png /
results/task7_*.csv.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Continues the same pipeline as Task 6: input is the histogram-equalized
(Task 3 winning enhancement) version of the same 3 test images, upscaled the
same way. Uses src.region_growing (new module, Task 7) + src.enhancement
(hist_eq only) + src.segment (otsu_threshold / segment_eye, reused as-is for
a reference pupil mask -- not duplicated). No pandas; CSV via stdlib csv.

Run: ./.venv/bin/python scripts/make_task7_figures.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import enhancement as enh              # noqa: E402
from src import segment as sg                   # noqa: E402
from src import region_growing as rg            # noqa: E402

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Same 3 images as Task 3 / Task 6.
IMAGES = [
    ("good_lighting", ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png"),
    ("poor_lighting", ROOT / "data/samples/alert/s0014_08297_0_0_1_1_0_02.png"),
    ("glasses_reflection", ROOT / "data/samples/alert/s0035_00133_0_1_1_2_1_02.png"),
]

UPSCALE_MIN = 300  # matches make_task3_figures.py / make_task6_figures.py

# Region-growing thresholds: 3 shown in the core figure (Low/Medium/High),
# plus a 4th (Very High) added only in the metrics table to pin down exactly
# where leakage past the reference pupil mask begins.
THRESH_LOW, THRESH_MED, THRESH_HIGH, THRESH_VHIGH = 10, 25, 45, 60
FIGURE_THRESHOLDS = [("Low", THRESH_LOW), ("Medium", THRESH_MED), ("High", THRESH_HIGH)]
TABLE_THRESHOLDS = [("Low", THRESH_LOW), ("Medium", THRESH_MED), ("High", THRESH_HIGH), ("Very High", THRESH_VHIGH)]

CONN_THRESHOLD = THRESH_MED  # representative threshold for the 4- vs 8-connectivity comparison
SPLIT_MIN_SIZE = 20
SPLIT_HOMOGENEITY = 25
TIMING_REPEATS = 30


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
# Figure: Step 1 -- original / grayscale panel with dims + min/max
# ---------------------------------------------------------------------------

def step1_figure(label: str, path: Path, fig_num: int) -> tuple[int, int, int, int]:
    raw_color = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray_native = cv2.cvtColor(raw_color, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray_native.shape[:2]
    gmin, gmax = int(gray_native.min()), int(gray_native.max())

    scale = max(1, UPSCALE_MIN // min(h0, w0))
    if scale > 1:
        raw_color = cv2.resize(raw_color, (w0 * scale, h0 * scale), interpolation=cv2.INTER_NEAREST)
        gray_native = cv2.resize(gray_native, (w0 * scale, h0 * scale), interpolation=cv2.INTER_NEAREST)
    raw_rgb = cv2.cvtColor(raw_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(raw_rgb)
    axes[0].set_title("Original (as loaded, IMREAD_COLOR)", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(gray_native, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale (cvtColor BGR2GRAY)", fontsize=9)
    axes[1].axis("off")
    fig.suptitle(
        f"Figure {fig_num}: Step 1 -- {label}: {h0}x{w0} native, min={gmin}, max={gmax} "
        f"(source file is already single-channel IR, so panels are numerically identical)",
        y=1.06, fontsize=8.3,
    )
    savefig(fig, f"task7_step1_{label}.png")
    return h0, w0, gmin, gmax


# ---------------------------------------------------------------------------
# Figure: region growing at 3 thresholds + manual-seed comparison
# ---------------------------------------------------------------------------

def region_growing_figure(label: str, enhanced: np.ndarray, auto_seed: tuple, manual_seed: tuple,
                           fig_num: int) -> dict:
    metrics = {}
    masks = {}
    for name, T in FIGURE_THRESHOLDS:
        mask, cnt = rg.region_grow(enhanced, auto_seed, T, connectivity=8)
        masks[name] = (mask, cnt, T)
        metrics[name] = cnt

    manual_mask, manual_cnt = rg.region_grow(enhanced, manual_seed, THRESH_MED, connectivity=8)
    metrics["manual_seed_medium"] = manual_cnt

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    seeded = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    axes[0].imshow(seeded)
    axes[0].plot(auto_seed[1], auto_seed[0], "r+", markersize=14, markeredgewidth=2, label="auto seed")
    axes[0].plot(manual_seed[1], manual_seed[0], "bx", markersize=12, markeredgewidth=2, label="manual seed")
    axes[0].set_title("Enhanced + seeds", fontsize=9)
    axes[0].legend(fontsize=6, loc="lower right")
    axes[0].axis("off")

    for ax, name in zip(axes[1:4], ["Low", "Medium", "High"]):
        mask, cnt, T = masks[name]
        ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Auto seed, T={T} ({name})\n{cnt}px ({cnt / enhanced.size:.1%})", fontsize=8.5)
        ax.axis("off")

    axes[4].imshow(manual_mask, cmap="gray", vmin=0, vmax=255)
    axes[4].set_title(f"Manual seed, T={THRESH_MED} (Medium)\n{manual_cnt}px ({manual_cnt / enhanced.size:.1%})", fontsize=8.5)
    axes[4].axis("off")

    fig.suptitle(f"Figure {fig_num}: Region growing at 3 thresholds -- {label}", y=1.06)
    savefig(fig, f"task7_regiongrow_{label}.png")
    return metrics


# ---------------------------------------------------------------------------
# Figure: 4- vs 8-connectivity
# ---------------------------------------------------------------------------

def compactness(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri == 0:
        return 0.0
    return float(4 * np.pi * area / (peri * peri))


def connectivity_figure(label: str, enhanced: np.ndarray, seed: tuple, fig_num: int) -> dict:
    mask4, count4 = rg.region_grow(enhanced, seed, CONN_THRESHOLD, connectivity=4)
    mask8, count8 = rg.region_grow(enhanced, seed, CONN_THRESHOLD, connectivity=8)
    diff = cv2.bitwise_xor(mask4, mask8)
    only8 = int(np.logical_and(mask8 > 0, mask4 == 0).sum())
    only4 = int(np.logical_and(mask4 > 0, mask8 == 0).sum())

    comp4, comp8 = compactness(mask4), compactness(mask8)

    t0 = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        rg.region_grow(enhanced, seed, CONN_THRESHOLD, connectivity=4)
    t4_ms = (time.perf_counter() - t0) / TIMING_REPEATS * 1000
    t0 = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        rg.region_grow(enhanced, seed, CONN_THRESHOLD, connectivity=8)
    t8_ms = (time.perf_counter() - t0) / TIMING_REPEATS * 1000

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].imshow(mask4, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"4-connected\n{count4}px, compactness={comp4:.3f}", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(mask8, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"8-connected\n{count8}px, compactness={comp8:.3f}", fontsize=9)
    axes[1].axis("off")
    axes[2].imshow(diff, cmap="hot", vmin=0, vmax=255)
    axes[2].set_title(f"Difference (XOR)\n{only8} only-8, {only4} only-4", fontsize=9)
    axes[2].axis("off")
    fig.suptitle(f"Figure {fig_num}: 4- vs 8-connectivity, T={CONN_THRESHOLD} -- {label}", y=1.06)
    savefig(fig, f"task7_connectivity_{label}.png")

    return {
        "count4": count4, "count8": count8, "only8": only8, "only4": only4,
        "comp4": round(comp4, 4), "comp8": round(comp8, 4),
        "t4_ms": round(t4_ms, 4), "t8_ms": round(t8_ms, 4),
    }


# ---------------------------------------------------------------------------
# Figure: split-merge
# ---------------------------------------------------------------------------

def splitmerge_figure(label: str, enhanced: np.ndarray, fig_num: int) -> dict:
    result = rg.split_merge(enhanced, min_size=SPLIT_MIN_SIZE, homogeneity_threshold=SPLIT_HOMOGENEITY)

    boundary_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    for (r0, r1, c0, c1) in result["leaves"]:
        cv2.rectangle(boundary_img, (c0, r0), (c1 - 1, r1 - 1), (0, 255, 0), 1)

    merged_rgb = label2rgb(result["labels"], image=enhanced, bg_label=0, alpha=0.55)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(boundary_img)
    axes[0].set_title(f"Quadtree split boundaries\n{result['n_leaves']} leaves", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(merged_rgb)
    axes[1].set_title(f"Merged regions\n{result['n_regions']} regions", fontsize=9)
    axes[1].axis("off")
    fig.suptitle(f"Figure {fig_num}: Region splitting and merging -- {label}", y=1.03)
    savefig(fig, f"task7_splitmerge_{label}.png")

    return {"n_leaves": result["n_leaves"], "n_regions": result["n_regions"]}


# ---------------------------------------------------------------------------
# Figure: summary bar chart (region size vs threshold)
# ---------------------------------------------------------------------------

def summary_bar_figure(threshold_rows: list[dict], fig_num: int):
    labels = [lbl for lbl, _p in IMAGES]
    thresh_names = [name for name, _T in TABLE_THRESHOLDS]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(thresh_names))
    width = 0.25
    for i, label in enumerate(labels):
        vals = [next(r["fraction"] for r in threshold_rows if r["image"] == label and r["threshold_label"] == t) * 100
                for t in thresh_names]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{n}\n(T={T})" for n, T in TABLE_THRESHOLDS], fontsize=8)
    ax.set_ylabel("Region size (% of image)")
    ax.set_title(f"Figure {fig_num}: Region-growing size vs. threshold, all 3 images", fontsize=10)
    ax.legend(fontsize=8)
    savefig(fig, "task7_summary_bars.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fig_n = 1
    threshold_rows = []
    connectivity_rows = []
    splitmerge_rows = []
    step1_rows = []
    seed_rows = []

    for label, path in IMAGES:
        h0, w0, gmin, gmax = step1_figure(label, path, fig_n); fig_n += 1
        step1_rows.append({"image": label, "native_h": h0, "native_w": w0, "min": gmin, "max": gmax})

    for label, path in IMAGES:
        orig = load_gray(path)
        enhanced = enh.equalize_histogram_cv(orig)

        auto = rg.auto_seed(enhanced, blur_ksize=5)
        manual = (enhanced.shape[0] // 2, enhanced.shape[1] // 2)
        seed_rows.append({
            "image": label, "auto_seed_row": auto[0], "auto_seed_col": auto[1],
            "auto_seed_value": int(enhanced[auto]),
            "manual_seed_row": manual[0], "manual_seed_col": manual[1],
            "manual_seed_value": int(enhanced[manual]),
        })

        # Reference pupil mask, reused as-is from src.segment (Otsu-anchored
        # stricter dark-core mask + largest-component isolation) -- used
        # only to quantify missing pixels / leakage / accuracy below, not
        # re-derived.
        seg_result = sg.segment_eye(enhanced)
        ref_mask = seg_result.region_mask > 0
        ref_area = int(ref_mask.sum())

        region_growing_figure(label, enhanced, auto, manual, fig_n); fig_n += 1

        prev_count = None
        for name, T in TABLE_THRESHOLDS:
            mask, cnt = rg.region_grow(enhanced, auto, T, connectivity=8)
            reg = mask > 0
            inter = int(np.logical_and(ref_mask, reg).sum())
            missing = int(np.logical_and(ref_mask, ~reg).sum())
            leakage = int(np.logical_and(reg, ~ref_mask).sum())
            union = int(np.logical_or(ref_mask, reg).sum())
            iou = inter / union if union else 1.0
            growth_ratio = (cnt / prev_count) if prev_count else None
            threshold_rows.append({
                "image": label, "threshold_label": name, "T": T,
                "region_pixels": cnt, "fraction": round(cnt / enhanced.size, 4),
                "reference_pupil_pixels": ref_area,
                "missing_pixels": missing, "leakage_pixels": leakage,
                "iou_vs_reference": round(iou, 4),
                "growth_ratio_vs_prev": round(growth_ratio, 3) if growth_ratio else "",
            })
            prev_count = cnt

        connectivity_rows.append({"image": label, **connectivity_figure(label, enhanced, auto, fig_n)})
        fig_n += 1

        splitmerge_rows.append({
            "image": label, "min_size": SPLIT_MIN_SIZE, "homogeneity_threshold": SPLIT_HOMOGENEITY,
            **splitmerge_figure(label, enhanced, fig_n),
        })
        fig_n += 1

    summary_bar_figure(threshold_rows, fig_n); fig_n += 1

    def write_csv(name, rows):
        path = RESULTS_DIR / name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")

    write_csv("task7_step1.csv", step1_rows)
    write_csv("task7_seeds.csv", seed_rows)
    write_csv("task7_threshold_metrics.csv", threshold_rows)
    write_csv("task7_connectivity_metrics.csv", connectivity_rows)
    write_csv("task7_splitmerge_metrics.csv", splitmerge_rows)

    print("\n=== Seeds ===")
    for r in seed_rows:
        print(r)
    print("\n=== Threshold metrics ===")
    for r in threshold_rows:
        print(r)
    print("\n=== Connectivity metrics ===")
    for r in connectivity_rows:
        print(r)
    print("\n=== Split-merge metrics ===")
    for r in splitmerge_rows:
        print(r)


if __name__ == "__main__":
    main()
