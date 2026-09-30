#!/usr/bin/env python3
"""Generate all Task 8 (Binary Morphological Processing: Erosion and
Dilation) figures and CSVs into results/task8_*.png / results/task8_*.csv.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Source segmentation = Task 7 region-growing output (T=45, 8-connectivity,
auto seed) on the Task 3 histogram-equalized image. Run:
    ./.venv/bin/python scripts/make_task8_figures.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import enhancement as enh              # noqa: E402
from src import region_growing as rg            # noqa: E402
from src import segment as sg                   # noqa: E402
from src import binary_morphology as bm         # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

IMAGES = [
    ("good_lighting", ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png"),
    ("poor_lighting", ROOT / "data/samples/alert/s0014_08297_0_0_1_1_0_02.png"),
    ("glasses_reflection", ROOT / "data/samples/alert/s0035_00133_0_1_1_2_1_02.png"),
]
UPSCALE_MIN = 300
RG_T = 45                  # Task 7 "High" tolerance: richest structure to morph
THRESHOLDS = [40, 60, 100]  # binary-conversion thresholds shown
T_BIN = 60                 # threshold carried into the morphology experiments
SE_LABELS = {"square3": "3x3 square", "cross3": "3x3 cross", "square5": "5x5 square"}
SES = {k: bm.make_se(*v) for k, v in bm.SE_SPECS.items()}
ITERS = [1, 2, 3]
SMALL_AREA = 30
FG = "#1f3a5f"


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = max(1, UPSCALE_MIN // min(h, w))
    if s > 1:
        img = cv2.resize(img, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
    return img


def savefig(fig, name):
    fig.savefig(RESULTS / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", RESULTS / name)


def show(ax, m, title):
    ax.imshow(m, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=7.5)
    ax.axis("off")


def write_csv(name, rows):
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", RESULTS / name)


# ---------------- explanatory diagrams ----------------

def diagram_se():
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, (k, se) in zip(axes, SES.items()):
        n = se.shape[0]
        for i in range(n):
            for j in range(n):
                on = se[i, j] > 0
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=FG if on else "white", ec="#888"))
                ax.text(j + .5, n - 1 - i + .5, "1" if on else "0", ha="center", va="center",
                        color="white" if on else "#555", fontsize=9)
        c = n // 2
        ax.add_patch(Rectangle((c + .08, n - 1 - c + .08), .84, .84, fc="none", ec="#d62728", lw=2.2))
        ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{SE_LABELS[k]}\n{int((se > 0).sum())} active pixels", fontsize=9)
    fig.text(0.5, 0.02, "Red outline = origin (anchor) pixel. Filled = 1 (part of B), white = 0.",
             ha="center", fontsize=8)
    savefig(fig, "task8_diagram_se.png")


def diagram_fit_hit():
    A = np.zeros((9, 12), np.uint8)
    A[2:7, 2:8] = 1          # main block
    A[4, 8:11] = 1           # thin 1-px arm
    A[0, 11] = 1             # isolated noise pixel
    se = SES["cross3"]
    er = bm.erode_ref(A * 255, se) > 0
    di = bm.dilate_ref(A * 255, se) > 0
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    titles = ["Input A (with 1-px arm + noise dot)", "Erosion A (-) B: SE must FIT inside A",
              "Dilation A (+) B: SE must HIT A"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks(range(13)); ax.set_yticks(range(10)); ax.grid(True, color="#ddd", lw=.5)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.set_title(t, fontsize=7.6)
    def fill(ax, M, color, alpha=1.0):
        for i, j in zip(*np.nonzero(M)):
            ax.add_patch(Rectangle((j, i), 1, 1, fc=color, ec="#666", alpha=alpha))
    fill(axes[0], A, FG)
    fill(axes[1], A, "#c9d3e0"); fill(axes[1], er, FG)
    fill(axes[2], di, "#f4b6b6"); fill(axes[2], A, FG)
    # SE placed at a fit position (fits) and at a non-fit (arm)
    def place(ax, r, c, color):
        for i, j in zip(*np.nonzero(se)):
            ax.add_patch(Rectangle((c - 1 + j, r - 1 + i), 1, 1, fc="none", ec=color, lw=2))
    place(axes[1], 4, 4, "#2ca02c")      # fits
    place(axes[1], 4, 9, "#d62728")      # arm: does not fit
    place(axes[2], 1, 10, "#ff7f0e")     # hits noise dot
    fig.text(0.5, -0.02,
             "Erosion: green cross fits -> centre kept; red cross on the 1-px arm does not fit -> centre removed; "
             "noise dot vanishes.  Dilation: orange cross touches the noise dot -> its whole neighbourhood is added.\n"
             "Dark = result, light = removed (erosion) / added (dilation).", ha="center", fontsize=7.4)
    savefig(fig, "task8_diagram_fit_hit.png")


# ---------------- per-image figures ----------------

def source_segmentation(gray):
    enhanced = enh.equalize_histogram_cv(gray)
    seed = rg.auto_seed(enhanced)
    region, cnt = rg.region_grow(enhanced, seed, RG_T, connectivity=8)
    seg = np.where(region > 0, enhanced, 255).astype(np.uint8)
    return enhanced, seed, region, cnt, seg


def step1_figure(label, path, gray, region, seg, fig_n):
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    raw_up = cv2.resize(raw, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.2))
    axes[0, 0].imshow(cv2.cvtColor(raw_up, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Original\n(IMREAD_COLOR)", fontsize=7.5); axes[0, 0].axis("off")
    show(axes[0, 1], gray, "Grayscale\n(identical: 1-channel IR)")
    show(axes[0, 2], region, f"Task 7 region-grown\nmask (T={RG_T})")
    show(axes[0, 3], seg, "Segmented image\n(region kept, rest=255)")
    for ax, t in zip(axes[1], THRESHOLDS + [None]):
        if t is None:
            ax.axis("off"); continue
        b = bm.to_binary(seg, t)
        show(ax, b, f"Binary, t={t} (fg = pixel < t)\n{bm.fg_count(b)} px, {bm.n_components(b)} comp.")
    fig.suptitle(f"Figure {fig_n}: Binary conversion, {label}", fontsize=9, y=0.99)
    savefig(fig, f"task8_step1_{label}.png")


def op_figure(label, binary, op, fig_n):
    fn = bm.erode if op == "erosion" else bm.dilate
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for r, (k, se) in enumerate(SES.items()):
        show(axes[r, 0], binary, f"Input binary (t={T_BIN})\n{bm.fg_count(binary)} px")
        for c, it in enumerate(ITERS, start=1):
            o = fn(binary, se, it)
            m = bm.measure(o, binary)
            show(axes[r, c], o, f"{SE_LABELS[k]}, {it} it.\n{m['fg_pixels']} px ({m['pct_change']:+.1f}%), {m['components']} comp.")
    fig.suptitle(f"Figure {fig_n}: {op.capitalize()} with three structuring elements, 1-3 iterations, {label}", fontsize=9, y=0.995)
    fig.tight_layout()
    savefig(fig, f"task8_{op}_{label}.png")


def compare_figure(label, binary, fig_n):
    se = SES["square3"]
    e, d = bm.erode(binary, se), bm.dilate(binary, se)
    grad = cv2.subtract(d, e)
    ov = np.zeros(binary.shape + (3,), np.uint8)
    ov[binary > 0] = (200, 200, 200)
    ov[(d > 0) & (binary == 0)] = (230, 80, 80)      # added by dilation
    ov[(binary > 0) & (e == 0)] = (60, 110, 230)     # removed by erosion
    fig, axes = plt.subplots(1, 5, figsize=(9.5, 2.4))
    show(axes[0], binary, "Input A")
    show(axes[1], e, "Erosion 3x3")
    show(axes[2], d, "Dilation 3x3")
    show(axes[3], grad, "Dilation - Erosion\n(morph. gradient)")
    axes[4].imshow(ov, interpolation="nearest"); axes[4].axis("off")
    axes[4].set_title("Grey=A, blue=removed by erosion\nred=added by dilation", fontsize=7)
    fig.suptitle(f"Figure {fig_n}: Erosion vs dilation (3x3 square, 1 iteration), {label}", fontsize=9, y=1.06)
    savefig(fig, f"task8_compare_{label}.png")


def duality_figure(binary, label, fig_n):
    se = SES["cross3"]
    comp = 255 - binary
    lhs = bm.erode(binary, se)
    rhs = 255 - bm.dilate(comp, se[::-1, ::-1].copy())
    diff = cv2.absdiff(lhs, rhs)
    fig, axes = plt.subplots(1, 5, figsize=(9.5, 2.4))
    show(axes[0], binary, "A")
    show(axes[1], comp, "Background A^c")
    show(axes[2], lhs, "A (-) B (erode A)")
    show(axes[3], rhs, "(A^c (+) B_reflected)^c")
    show(axes[4], diff, f"|difference|\n{int((diff > 0).sum())} px differ")
    fig.suptitle(f"Figure {fig_n}: Duality check, 3x3 cross, {label}: erosion of A equals complement of dilation of A^c", fontsize=9, y=1.06)
    savefig(fig, "task8_duality.png")


def curves_figure(curve_rows, fig_n):
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
    for c, (label, _) in enumerate(IMAGES):
        for k in SES:
            rows = [r for r in curve_rows if r["image"] == label and r["se"] == k]
            er = [r for r in rows if r["op"] == "erosion"]
            di = [r for r in rows if r["op"] == "dilation"]
            axes[0, c].plot([r["iter"] for r in er], [r["pct_change"] for r in er], label=SE_LABELS[k])
            axes[1, c].plot([r["iter"] for r in di], [r["components"] for r in di], label=SE_LABELS[k])
        axes[0, c].set_title(f"{label}: erosion, area change (%)", fontsize=8)
        axes[1, c].set_title(f"{label}: dilation, components", fontsize=8)
        axes[0, c].set_xlabel("iterations", fontsize=7); axes[1, c].set_xlabel("iterations", fontsize=7)
        for a in axes[:, c]:
            a.tick_params(labelsize=7); a.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Figure {fig_n}: Area (erosion) and component count (dilation) vs iteration count", fontsize=9, y=1.0)
    fig.tight_layout()
    savefig(fig, "task8_curves.png")


def bars_figure(rows, fig_n):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    labs = [l for l, _ in IMAGES]
    w = 0.25
    for ax, op in zip(axes, ["erosion", "dilation"]):
        for i, k in enumerate(SES):
            vals = [next(r["pct_change"] for r in rows if r["image"] == l and r["se"] == k and r["op"] == op and r["iter"] == 1) for l in labs]
            ax.bar(np.arange(3) + i * w, vals, w, label=SE_LABELS[k])
        ax.set_xticks(np.arange(3) + w); ax.set_xticklabels(labs, fontsize=7)
        ax.set_title(f"{op.capitalize()}: foreground area change, 1 iteration (%)", fontsize=8)
        ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.3, axis="y"); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Figure {fig_n}: Area change per structuring element", fontsize=9, y=1.02)
    savefig(fig, "task8_area_bars.png")


def main():
    diagram_se()
    diagram_fit_hit()
    data = {}
    bin_rows, seg_rows = [], []
    for n, (label, path) in enumerate(IMAGES):
        gray = load_gray(path)
        enhanced, seed, region, cnt, seg = source_segmentation(gray)
        otsu_t, _ = sg.otsu_threshold(seg)
        step1_figure(label, path, gray, region, seg, 3 + n)
        seg_rows.append({"image": label, "size": f"{gray.shape[0]}x{gray.shape[1]}", "seed": f"{seed[0]},{seed[1]}",
                         "region_grow_T": RG_T, "region_pixels": cnt, "otsu_on_segmented": otsu_t})
        for t in THRESHOLDS:
            b = bm.to_binary(seg, t)
            bin_rows.append({"image": label, "threshold": t, **bm.measure(b),
                             "small_comps_le30": sum(1 for a in bm.component_areas(b) if a <= SMALL_AREA)})
        data[label] = bm.to_binary(seg, T_BIN)

    op_rows, curve_rows, vm_rows, thin_rows, dual_rows = [], [], [], [], []
    for label, b in data.items():
        for k, se in SES.items():
            thin_rows.append({"image": label, "se": k, "thin_pixels_A_minus_open": bm.thin_pixels(b, se),
                              "gap_pixels_close_minus_A": bm.gap_pixels(b, se)})
            cv = bm.component_vanish_iterations(b, se)
            vm = bm.merge_iterations(b, se)
            vm_rows.append({"image": label, "se": k, "components0": vm["n0"],
                            "erosion_all_vanish_iter": bm.iterations_to_vanish(b, se),
                            "erosion_component_vanish_iters": "/".join(str(v) for v in cv),
                            "dilation_first_merge_iter": vm["first_merge_iter"],
                            "dilation_single_component_iter": vm["single_iter"]})
            for op, fn in (("erosion", bm.erode), ("dilation", bm.dilate)):
                for it in ITERS:
                    op_rows.append({"image": label, "se": k, "op": op, "iter": it, **bm.measure(fn(b, se, it), b)})
                for it in range(0, 41):
                    m = bm.measure(fn(b, se, it) if it else b, b)
                    curve_rows.append({"image": label, "se": k, "op": op, "iter": it, **m})
            for it in ITERS:
                dual_rows.append({"image": label, "se": k, "iter": it, **bm.duality_check(b, se, it)})
        op_rows.insert(0, {"image": label, "se": "input", "op": "-", "iter": 0, **bm.measure(b, b)}) if False else None

    # input rows
    input_rows = [{"image": l, **bm.measure(b, b)} for l, b in data.items()]

    fn_ = 6
    for op in ("erosion", "dilation"):
        for label, b in data.items():
            op_figure(label, b, op, fn_); fn_ += 1
    for label, b in data.items():
        compare_figure(label, b, fn_); fn_ += 1
    bars_figure(curve_rows, fn_); fn_ += 1
    curves_figure(curve_rows, fn_); fn_ += 1
    duality_figure(data["good_lighting"], "good_lighting", fn_)

    write_csv("task8_segmentation_source.csv", seg_rows)
    write_csv("task8_binary_thresholds.csv", bin_rows)
    write_csv("task8_input_binary.csv", input_rows)
    write_csv("task8_operations.csv", op_rows)
    write_csv("task8_curves.csv", curve_rows)
    write_csv("task8_vanish_merge.csv", vm_rows)
    write_csv("task8_thin_gap.csv", thin_rows)
    write_csv("task8_duality.csv", dual_rows)

    bad = [r for r in dual_rows if any(r[c] for c in ("erode_vs_dual_diff", "dilate_vs_dual_diff", "erode_cv_vs_ref_diff", "dilate_cv_vs_ref_diff"))]
    print("DUALITY:", "PASS" if not bad else f"FAIL {bad}", f"({len(dual_rows)} cases)")
    for name, rows in (("seg", seg_rows), ("bin", bin_rows), ("input", input_rows), ("vm", vm_rows), ("thin", thin_rows)):
        print("==", name)
        for r in rows:
            print(r)
    print("== ops")
    for r in op_rows:
        print(r)


if __name__ == "__main__":
    main()
