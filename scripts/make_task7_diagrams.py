#!/usr/bin/env python3
"""Task 7 -- explanatory / conceptual diagrams (region growing, connectivity,
quadtree split-merge, pipeline context).

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Additive companion to scripts/make_task7_figures.py: that script produces the
*measured-result* figures (Figures 1-13, already embedded in
docs/task7_report.html and NOT touched here). This script produces
*explanatory* diagrams -- schematics, a flowchart, a scanline profile, a
progression sequence -- numbered Figure 14-20 so the existing figures never
need renumbering. Written into results/task7_diag_*.png.

Does not modify src/region_growing.py. Where a diagram needs the algorithm's
exact running-mean / homogeneity semantics (progression snapshots, the
quadtree recursion, the region-adjacency merge test) it calls straight into
src.region_growing's public and module-level helpers so the picture matches
the tested code, not a re-description of it.

Run: ./.venv/bin/python scripts/make_task7_diagrams.py
"""
from __future__ import annotations

import sys
import textwrap
from collections import deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import enhancement as enh          # noqa: E402
from src import region_growing as rg        # noqa: E402

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

UPSCALE_MIN = 300  # same convention as make_task7_figures.py

IMAGES = {
    "good_lighting": ROOT / "data/samples/alert/s0031_00366_1_0_1_0_1_02.png",
    "poor_lighting": ROOT / "data/samples/alert/s0014_08297_0_0_1_1_0_02.png",
    "glasses_reflection": ROOT / "data/samples/alert/s0035_00133_0_1_1_2_1_02.png",
}


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


# ===========================================================================
# Figure 14 -- region-growing flowchart
# ===========================================================================

BOX_PROC = "#dbe8fb"   # process (light blue)
BOX_DEC = "#fdf1cf"    # decision (light amber)
BOX_IO = "#e0f3e0"     # I/O (light green)
BOX_TERM = "#eaeaea"   # terminator (light grey)
EDGE = "#333333"


def _wrap(text, width=26):
    return "\n".join(textwrap.wrap(text, width=width))


def _terminator(ax, cx, cy, w, h, text):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={h/2}",
                          linewidth=1.3, edgecolor=EDGE, facecolor=BOX_TERM, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.6, fontweight="bold", zorder=3)


def _process(ax, cx, cy, w, h, text):
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0,rounding_size=0.04",
                          linewidth=1.3, edgecolor=EDGE, facecolor=BOX_PROC, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, _wrap(text), ha="center", va="center", fontsize=8.0, zorder=3)


def _io(ax, cx, cy, w, h, text):
    skew = 0.22
    pts = [(cx - w / 2 + skew, cy + h / 2), (cx + w / 2 + skew, cy + h / 2),
           (cx + w / 2 - skew, cy - h / 2), (cx - w / 2 - skew, cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.3, edgecolor=EDGE, facecolor=BOX_IO, zorder=2))
    ax.text(cx, cy, _wrap(text, 28), ha="center", va="center", fontsize=7.8, zorder=3)


def _decision(ax, cx, cy, w, h, text):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.3, edgecolor=EDGE, facecolor=BOX_DEC, zorder=2))
    ax.text(cx, cy, _wrap(text, 20), ha="center", va="center", fontsize=7.6, zorder=3)


def _arrow(ax, p1, p2, label=None, label_dx=0.18, label_ha="left", **kw):
    style = dict(arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=EDGE, shrinkA=0, shrinkB=0)
    style.update(kw)
    ax.add_patch(FancyArrowPatch(p1, p2, **style, zorder=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + label_dx, my, label, fontsize=7.2, color="#8a4b00", ha=label_ha, va="center", zorder=3,
                style="italic")


def diagram_flowchart(fig_num: int):
    fig, ax = plt.subplots(figsize=(10.6, 13.2))
    ax.set_xlim(-5.9, 5.9)
    ax.set_ylim(-3.0, 21.5)
    ax.axis("off")
    ax.set_aspect("equal")

    W, H = 3.1, 1.05
    WIO = 3.7
    x0 = 0.0
    ys = {}
    y = 20.6
    ys["start"] = y; _terminator(ax, x0, y, 1.9, 0.85, "Start"); y -= 1.55
    ys["seed"] = y; _io(ax, x0, y, WIO, 1.1, "Select seed p0 (auto: darkest\npoint after Gaussian blur, or manual)"); y -= 1.6
    ys["init"] = y; _process(ax, x0, y, W, H, "Init: R = {p0}; region_sum = I(p0);\nregion_count = 1; μ_R = I(p0)"); y -= 1.55
    ys["push"] = y; _process(ax, x0, y, W, H, "Push p0 onto stack S; mark p0 visited"); y -= 1.6

    ys["loopA"] = y; _decision(ax, x0, y, 2.5, 1.15, "Stack S\nempty?"); y -= 1.75
    ys["pop"] = y; _process(ax, x0, y, W, H, "Pop pixel p = (r,c) from S"); y -= 1.55
    ys["examine"] = y; _process(ax, x0, y, W, H, "Take next unvisited\nneighbour q of p\n(4- or 8-connected)"); y -= 1.75
    ys["test"] = y; _decision(ax, x0, y, 2.6, 1.3, "|I(q) − μ_R|\n≤ T ?"); y -= 1.85
    ys["reject"] = y; _process(ax, x0 - 2.7, y, 2.15, 1.05, "Mark q visited,\ndiscard")
    ys["accept"] = y; _process(ax, x0 + 2.95, y, 2.75, 1.65,
                                "Add q to R; mark visited;\nregion_sum += I(q);\nregion_count += 1;\n"
                                "μ_R = region_sum/region_count;\npush q onto S")
    y -= 2.15

    ys["moreq"] = y; _decision(ax, x0, y, 2.7, 1.2, "More\nneighbours\nof p?"); y -= 1.9

    ys["output"] = y; _io(ax, x0, y, WIO, 1.1, "Output mask M (R marked 255),\n|R| = region_count"); y -= 1.6
    ys["end"] = y; _terminator(ax, x0, y, 1.9, 0.85, "End")

    # main spine arrows
    _arrow(ax, (x0, ys["start"] - 0.43), (x0, ys["seed"] + 0.55))
    _arrow(ax, (x0, ys["seed"] - 0.55), (x0, ys["init"] + 0.53))
    _arrow(ax, (x0, ys["init"] - 0.53), (x0, ys["push"] + 0.53))
    _arrow(ax, (x0, ys["push"] - 0.53), (x0, ys["loopA"] + 0.58))
    _arrow(ax, (x0, ys["loopA"] - 0.58), (x0, ys["pop"] + 0.53), label="No", label_dx=0.22)
    _arrow(ax, (x0, ys["pop"] - 0.53), (x0, ys["examine"] + 0.58))
    _arrow(ax, (x0, ys["examine"] - 0.58), (x0, ys["test"] + 0.65))
    _arrow(ax, (x0 - 0.15, ys["test"] - 0.15), (x0 - 2.7, ys["reject"] + 0.53), label="No", label_dx=-0.1, label_ha="right")
    _arrow(ax, (x0 + 0.15, ys["test"] - 0.15), (x0 + 2.95, ys["accept"] + 0.83), label="Yes", label_dx=0.1)
    _arrow(ax, (x0 - 2.7, ys["reject"] - 0.53), (x0 - 0.05, ys["moreq"] + 0.63))
    _arrow(ax, (x0 + 2.95, ys["accept"] - 0.83), (x0 + 0.05, ys["moreq"] + 0.63))
    _arrow(ax, (x0, ys["moreq"] - 0.6), (x0, ys["output"] + 0.55), label="No")
    _arrow(ax, (x0, ys["output"] - 0.55), (x0, ys["end"] + 0.43), label="Yes")

    # loop back: more neighbours -> examine (right side, well clear of the wide "accept" box)
    loopR = 5.1
    _arrow(ax, (x0 + 1.35, ys["moreq"]), (loopR, ys["moreq"]))
    ax.plot([loopR, loopR], [ys["moreq"], ys["examine"]], color=EDGE, linewidth=1.2, zorder=1)
    _arrow(ax, (loopR, ys["examine"]), (x0 + 1.55, ys["examine"]), label="Yes", label_dx=0.15, label_ha="left")

    # loop back: stack not empty -> pop (left side, i.e. outer while loop)
    loopL = -5.1
    _arrow(ax, (x0 - 1.35, ys["pop"]), (loopL, ys["pop"]))
    ax.plot([loopL, loopL], [ys["pop"], ys["loopA"]], color=EDGE, linewidth=1.2, zorder=1)
    _arrow(ax, (loopL, ys["loopA"]), (x0 - 1.25, ys["loopA"]))

    ax.text(loopL - 0.32, (ys["pop"] + ys["loopA"]) / 2, "(loop while\nstack non-empty)",
            fontsize=6.6, color="#555", ha="center", va="center", rotation=90, style="italic")
    ax.text(loopR + 0.32, (ys["moreq"] + ys["examine"]) / 2, "(loop over\nneighbours of p)",
            fontsize=6.6, color="#555", ha="center", va="center", rotation=90, style="italic")

    legend_items = [
        (BOX_TERM, "Terminator (start / end)"),
        (BOX_IO, "Input / Output"),
        (BOX_PROC, "Process"),
        (BOX_DEC, "Decision"),
    ]
    for i, (color, label) in enumerate(legend_items):
        ly = -0.5 - i * 0.55
        ax.add_patch(Rectangle((-5.7, ly - 0.16), 0.32, 0.32, facecolor=color, edgecolor=EDGE, linewidth=1.0))
        ax.text(-5.25, ly, label, fontsize=7.6, va="center")

    ax.set_title(
        f"Figure {fig_num}: Region-growing flowchart -- BFS from seed p0, admitting neighbour q when\n"
        r"$|I(q)-\mu_R|\leq T$, with $\mu_R$ the running region mean updated after every admission",
        fontsize=9.6, pad=10,
    )
    savefig(fig, "task7_diag_flowchart.png")


# ===========================================================================
# Figure 15 -- 4- vs 8-connectivity neighbourhood diagram
# ===========================================================================

def _draw_neighbourhood(ax, included: set, title: str):
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            x, y = dc, -dr
            is_centre = (dr == 0 and dc == 0)
            is_in = (dr, dc) in included
            color = "#c9c9c9" if is_centre else ("#a8d5a8" if is_in else "#f2f2f2")
            ax.add_patch(Rectangle((x - 0.48, y - 0.48), 0.96, 0.96,
                                    facecolor=color, edgecolor="#333", linewidth=1.2))
            if is_centre:
                ax.text(x, y + 0.14, "p", fontsize=11, ha="center", va="center", fontweight="bold")
                ax.text(x, y - 0.24, "(0,0)", fontsize=6.8, ha="center", va="center", color="#444")
            else:
                dirname = {(-1, 0): "N", (1, 0): "S", (0, -1): "W", (0, 1): "E",
                           (-1, -1): "NW", (-1, 1): "NE", (1, -1): "SW", (1, 1): "SE"}[(dr, dc)]
                if is_in:
                    ax.text(x, y + 0.14, dirname, fontsize=9.5, ha="center", va="center", fontweight="bold")
                    ax.text(x, y - 0.24, f"({dr:+d},{dc:+d})", fontsize=6.8, ha="center", va="center", color="#444")
                else:
                    ax.text(x, y, "–", fontsize=10, ha="center", va="center", color="#999")
    ax.set_title(title, fontsize=10)


def diagram_connectivity(fig_num: int):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    conn4 = set(rg._CONN4)
    conn8 = set(rg._CONN8)
    _draw_neighbourhood(axes[0], conn4, "4-connected neighbourhood $N_4(p)$\n(orthogonal only)")
    _draw_neighbourhood(axes[1], conn8, "8-connected neighbourhood $N_8(p)$\n(orthogonal + diagonal)")
    fig.suptitle(
        f"Figure {fig_num}: Pixel-neighbourhood definitions used by region_grow(...,connectivity=4|8).\n"
        "Each cell is labelled with its (Δrow,Δcol) offset from the centre pixel p; green = admitted "
        "as a candidate neighbour, grey/dash = not considered a neighbour.",
        fontsize=9.3, y=1.06,
    )
    savefig(fig, "task7_diag_connectivity.png")


# ===========================================================================
# Figure 16 -- region-growing progression (frontier snapshots)
# ===========================================================================

def _region_grow_with_snapshots(img, seed, T, connectivity, snapshot_counts):
    """Same BFS semantics as src.region_growing.region_grow, but records a
    copy of the mask each time region_count first reaches one of
    `snapshot_counts` (sorted ascending). Read-only w.r.t. src/ -- duplicated
    here only to add instrumentation, not to change any admission logic
    (identical neighbour list, identical running-mean update, identical
    |I-mean|<=T test as src.region_growing.region_grow).
    """
    neighbours = rg._CONN8 if connectivity == 8 else rg._CONN4
    h, w = img.shape[:2]
    img_i = img.astype(np.int32)
    sr, sc = seed
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    visited[sr, sc] = True
    mask[sr, sc] = 255
    region_sum = float(img_i[sr, sc])
    region_count = 1
    mean = region_sum / region_count

    targets = sorted(snapshot_counts)
    snaps = []
    ti = 0
    if ti < len(targets) and region_count >= targets[ti]:
        snaps.append((targets[ti], mask.copy())); ti += 1

    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                visited[nr, nc] = True
                if abs(int(img_i[nr, nc]) - mean) <= T:
                    mask[nr, nc] = 255
                    region_count += 1
                    region_sum += img_i[nr, nc]
                    mean = region_sum / region_count
                    q.append((nr, nc))
                    while ti < len(targets) and region_count >= targets[ti]:
                        snaps.append((targets[ti], mask.copy())); ti += 1
    while ti < len(targets):
        snaps.append((targets[ti], mask.copy())); ti += 1
    return snaps, region_count


def diagram_progression(fig_num: int):
    label = "good_lighting"
    orig = load_gray(IMAGES[label])
    enhanced = enh.equalize_histogram_cv(orig)
    seed = rg.auto_seed(enhanced, blur_ksize=5)
    T = 45  # High -- large enough to show clear outward growth in a handful of snapshots

    _, final_count = rg.region_grow(enhanced, seed, T, connectivity=8)
    fracs = [0.02, 0.1, 0.3, 0.6, 1.0]
    targets = [max(1, round(f * final_count)) for f in fracs]
    snaps, _ = _region_grow_with_snapshots(enhanced, seed, T, connectivity=8, snapshot_counts=targets)

    fig, axes = plt.subplots(1, len(snaps), figsize=(3.1 * len(snaps), 3.4))
    base_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    for ax, (cnt, mask), frac in zip(axes, snaps, fracs):
        overlay = base_rgb.copy()
        overlay[mask > 0] = (255, 60, 60)
        blended = cv2.addWeighted(base_rgb, 0.45, overlay, 0.55, 0)
        ax.imshow(blended)
        ax.plot(seed[1], seed[0], "+", color="yellow", markersize=10, markeredgewidth=2)
        ax.set_title(f"{cnt} px\n({frac*100:.0f}% of final)", fontsize=8.6)
        ax.axis("off")
    fig.suptitle(
        f"Figure {fig_num}: Region-growing progression -- {label}, auto seed, T={T} (High). "
        "The frontier expands outward from the seed (yellow +) one BFS layer at a time; "
        f"final region = {final_count}px.",
        fontsize=9.3, y=1.05,
    )
    savefig(fig, "task7_diag_progression.png")


# ===========================================================================
# Figure 17 -- quadtree splitting diagram (schematic image + tree structure)
# ===========================================================================

def _make_schematic_image(n=32):
    """Small (32x32) schematic "pupil-like dark disc on bright background".
    Sized deliberately so the recursion stays shallow and legible: the disc
    sits inside the left half only (so the two right-hand quadrants are pure
    background and stop at depth 1), and min_size=8 with a 32-pixel frame
    caps the recursion at depth 2 (32 -> 16 -> 8) -- a small, fully labelled
    tree rather than a wall of leaves.
    """
    img = np.full((n, n), 210, dtype=np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    disc = (xx - 8) ** 2 + (yy - 16) ** 2 <= 6 ** 2
    img[disc] = 25
    rng = np.random.default_rng(7)
    img = np.clip(img.astype(int) + rng.integers(-3, 4, size=img.shape), 0, 255).astype(np.uint8)
    return img


def _build_quadtree(img, r0, r1, c0, c1, min_size, threshold, depth, nodes, parent):
    block = img[r0:r1, c0:c1]
    homog = rg._homogeneous(block, threshold)
    size_floor = (r1 - r0) <= min_size or (c1 - c0) <= min_size
    is_leaf = homog or size_floor
    node = {
        "rect": (r0, r1, c0, c1), "depth": depth, "mean": float(block.mean()),
        "range": int(block.max()) - int(block.min()), "is_leaf": is_leaf,
        "reason": "size floor" if (size_floor and not homog) else ("homogeneous" if homog else "split"),
        "parent": parent, "children": [],
    }
    idx = len(nodes)
    nodes.append(node)
    if parent is not None:
        nodes[parent]["children"].append(idx)
    if not is_leaf:
        h, w = r1 - r0, c1 - c0
        rm, cm = r0 + h // 2, c0 + w // 2
        _build_quadtree(img, r0, rm, c0, cm, min_size, threshold, depth + 1, nodes, idx)
        _build_quadtree(img, r0, rm, cm, c1, min_size, threshold, depth + 1, nodes, idx)
        _build_quadtree(img, rm, r1, c0, cm, min_size, threshold, depth + 1, nodes, idx)
        _build_quadtree(img, rm, r1, cm, c1, min_size, threshold, depth + 1, nodes, idx)
    return idx


def _assign_leaf_x(nodes, idx, counter):
    node = nodes[idx]
    if node["is_leaf"]:
        node["x"] = counter[0]
        counter[0] += 1
    else:
        for ch in node["children"]:
            _assign_leaf_x(nodes, ch, counter)
        node["x"] = sum(nodes[ch]["x"] for ch in node["children"]) / len(node["children"])


DEPTH_COLORS = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#2980b9"]


def diagram_quadtree(fig_num: int):
    n_px = 32
    img = _make_schematic_image(n_px)
    min_size, threshold = 8, 25
    nodes = []
    _build_quadtree(img, 0, n_px, 0, n_px, min_size, threshold, 0, nodes, None)
    _assign_leaf_x(nodes, 0, [0])
    max_depth = max(nd["depth"] for nd in nodes)
    x_scale = 2.7
    for nd in nodes:
        nd["x"] *= x_scale

    fig, (ax_img, ax_tree) = plt.subplots(1, 2, figsize=(17.5, 5.8),
                                           gridspec_kw={"width_ratios": [1, 2.1]})

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)
    for nd in nodes:
        r0, r1, c0, c1 = nd["rect"]
        color = DEPTH_COLORS[min(nd["depth"], len(DEPTH_COLORS) - 1)]
        lw = 2.4 if nd["is_leaf"] else 1.3
        ax_img.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                    edgecolor=color, linewidth=lw))
    ax_img.set_title(f"Schematic image, {n_px}x{n_px}\nmin_size={min_size}, homogeneity threshold={threshold}",
                      fontsize=9.3)
    ax_img.axis("off")
    handles = [Line2D([0], [0], color=DEPTH_COLORS[d], lw=2, label=f"depth {d} boundary")
               for d in range(max_depth + 1)]
    ax_img.legend(handles=handles, fontsize=7.2, loc="lower right", framealpha=0.9)

    for nd in nodes:
        if nd["parent"] is not None:
            p = nodes[nd["parent"]]
            ax_tree.plot([p["x"], nd["x"]], [-p["depth"], -nd["depth"]], color="#888", linewidth=1.0, zorder=1)
    for nd in nodes:
        color = DEPTH_COLORS[min(nd["depth"], len(DEPTH_COLORS) - 1)]
        marker = "s" if nd["is_leaf"] else "o"
        ax_tree.scatter([nd["x"]], [-nd["depth"]], s=260 if nd["is_leaf"] else 190,
                         marker=marker, color=color, edgecolor="black", linewidth=0.9, zorder=2)
        label = f"μ={nd['mean']:.0f}, R={nd['range']}"
        if nd["is_leaf"]:
            label += f"\n({nd['reason']})"
        ax_tree.text(nd["x"], -nd["depth"] - 0.28, label, fontsize=6.6, ha="center", va="top")
    ax_tree.set_xlim(-1, max(nd["x"] for nd in nodes) + 1)
    ax_tree.set_ylim(-max_depth - 1.1, 0.6)
    ax_tree.axis("off")
    ax_tree.set_title(
        "Quadtree recursion (● = split further, ■ = leaf)\n"
        r"homogeneous $\Leftrightarrow$ max$-$min $\leq$ threshold, tested at each node",
        fontsize=9.3,
    )

    fig.suptitle(
        f"Figure {fig_num}: Quadtree splitting on a schematic image -- a block is kept as a leaf once it is "
        "homogeneous (max-min <= threshold) or hits the minimum size, else it recurses into 4 quadrants "
        "(TL/TR/BL/BR). The two right-hand quadrants are pure background and stop at depth 1; the two "
        "left-hand quadrants straddle the dark disc and recurse to depth 2, where most sub-blocks are forced "
        "leaves by the 8px size floor despite still being heterogeneous.",
        fontsize=9.0, y=1.02,
    )
    savefig(fig, "task7_diag_quadtree.png")


# ===========================================================================
# Figure 18 -- region adjacency graph / merge criterion
# ===========================================================================

def _make_rag_schematic(cell=16):
    """4x4 grid of `cell`x`cell` blocks in 4 clusters (2x2 blocks each). Every
    block is filled with a *constant* value (cluster base mean + a small
    deterministic per-position offset), so:
      - each 2x2 cluster's own 4 leaves differ by at most 6 (offsets
        -3/+3/-2/+2) -> within the merge tolerance, so they should merge;
      - different clusters' base means are >=100 apart -> never within
        tolerance, so cluster boundaries should never merge.
    A small SPLIT_THRESH (below that within-cluster range of 6) is used only
    for the split step, forcing every 2x2 cluster to split all the way down
    to its 4 individual `cell`x`cell` leaves (16 leaves total) rather than
    stopping early because a whole cluster already looks "homogeneous enough"
    at the quadrant level -- the split and merge criteria are deliberately
    different tolerances here (split_merge's own `merge_tolerance` parameter)
    purely to keep this illustration's leaf count readable; the real
    src.region_growing.split_merge calls in make_task7_figures.py use one
    shared tolerance for both, as documented in the report.
    """
    n = cell * 4
    img = np.zeros((n, n), dtype=np.float64)
    base = {(0, 0): 30, (0, 1): 150, (1, 0): 200, (1, 1): 70}  # (cluster-row, cluster-col) -> base mean
    offset = {(0, 0): -3, (0, 1): 3, (1, 0): -2, (1, 1): 2}    # (block-row%2, block-col%2) -> intra-cluster offset
    for br in range(4):
        for bc in range(4):
            cluster_mean = base[(br // 2, bc // 2)]
            off = offset[(br % 2, bc % 2)]
            img[br * cell:(br + 1) * cell, bc * cell:(bc + 1) * cell] = cluster_mean + off
    return np.clip(img, 0, 255).astype(np.uint8)


def diagram_rag(fig_num: int):
    cell = 16
    img = _make_rag_schematic(cell)
    split_thresh, tolerance = 5, 20
    result = rg.split_merge(img, min_size=cell, homogeneity_threshold=split_thresh, merge_tolerance=tolerance)
    leaves, means, labels = result["leaves"], result["leaf_means"], result["labels"]
    n = len(leaves)
    centroids = [((r0 + r1) / 2, (c0 + c1) / 2) for (r0, r1, c0, c1) in leaves]
    region_ids = [int(labels[int(cy), int(cx)]) for (cy, cx) in centroids]

    fig, (ax_img, ax_graph) = plt.subplots(1, 2, figsize=(12, 5.6))

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)
    for (r0, r1, c0, c1) in leaves:
        ax_img.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="cyan", linewidth=1.0))
    ax_img.set_title(f"Schematic 4x4-leaf image ({n} leaves)\ncell mean shown as grey level", fontsize=9.3)
    ax_img.axis("off")

    cmap = plt.get_cmap("tab10")
    unique_regions = sorted(set(region_ids))
    region_color = {rid: cmap(i % 10) for i, rid in enumerate(unique_regions)}

    n_merged, n_kept = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if rg._leaves_adjacent(leaves[i], leaves[j]):
                diff = abs(means[i] - means[j])
                merged = diff <= tolerance
                color, style, lw = ("#2e7d32", "-", 1.8) if merged else ("#c62828", "--", 1.1)
                n_merged += merged
                n_kept += (not merged)
                (yi, xi), (yj, xj) = centroids[i], centroids[j]
                ax_graph.plot([xi, xj], [yi, yj], color=color, linestyle=style, linewidth=lw, zorder=1, alpha=0.9)
                mx, my = (xi + xj) / 2, (yi + yj) / 2
                ax_graph.text(mx, my, f"{diff:.0f}", fontsize=5.6, color=color, ha="center", va="center",
                               bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.75), zorder=3)

    for (cy, cx), rid in zip(centroids, region_ids):
        ax_graph.scatter([cx], [cy], s=260, color=region_color[rid], edgecolor="black", linewidth=1.0, zorder=2)
    ax_graph.invert_yaxis()
    ax_graph.set_aspect("equal")
    ax_graph.set_title(
        f"Region adjacency graph over the {n} leaves\n"
        f"node colour = final merged region id ({result['n_regions']} regions); "
        f"edge label = |μ_i − μ_j|", fontsize=9.3)
    ax_graph.axis("off")
    legend_handles = [
        Line2D([0], [0], color="#2e7d32", lw=1.8, label=f"merge: |Δμ| ≤ tol ({tolerance})"),
        Line2D([0], [0], color="#c62828", lw=1.1, ls="--", label=f"kept separate: |Δμ| > tol"),
    ]
    ax_graph.legend(handles=legend_handles, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Figure {fig_num}: Region adjacency graph and the merge criterion, on a schematic 4-cluster image. "
        f"{n_merged} adjacent leaf pairs merge (mean difference within tolerance {tolerance}), "
        f"{n_kept} stay separate (cluster boundaries) -- {n} leaves collapse to {result['n_regions']} regions.",
        fontsize=9.1, y=1.03,
    )
    savefig(fig, "task7_diag_rag.png")


# ===========================================================================
# Figure 19 -- intensity profile with threshold acceptance bands
# ===========================================================================

THRESH_BANDS = [(60, "#fde0dc", "T=60 (Very High)"), (45, "#fff0c2", "T=45 (High)"),
                (25, "#dff5d8", "T=25 (Medium)"), (10, "#cfe8fb", "T=10 (Low)")]


def _scanline_panel(ax, label: str):
    orig = load_gray(IMAGES[label])
    enhanced = enh.equalize_histogram_cv(orig)
    seed = rg.auto_seed(enhanced, blur_ksize=5)
    row, col0 = seed
    profile = enhanced[row, :].astype(int)
    seed_val = int(enhanced[seed])
    x = np.arange(len(profile))

    for T, color, band_label in THRESH_BANDS:
        lo, hi = max(0, seed_val - T), min(255, seed_val + T)
        ax.axhspan(lo, hi, color=color, zorder=0, label=f"{band_label}: [{lo},{hi}]")
    ax.plot(x, profile, color="#111", linewidth=1.3, zorder=3, label="scanline intensity I(row,x)")
    ax.axvline(col0, color="#7a1fa2", linestyle=":", linewidth=1.2, zorder=2)
    ax.plot([col0], [seed_val], marker="*", color="#7a1fa2", markersize=13, zorder=4)
    ax.set_xlim(0, len(profile) - 1)
    ax.set_ylim(0, 255)
    ax.set_xlabel("column x (scanline through seed row)", fontsize=8.4)
    ax.set_ylabel("intensity I(row, x)", fontsize=8.4)
    ax.set_title(f"{label} -- seed row {row}, seed value {seed_val}", fontsize=9.2)
    return seed_val


def diagram_intensity_profile(fig_num: int):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)
    _scanline_panel(axes[0], "poor_lighting")
    _scanline_panel(axes[1], "glasses_reflection")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
               framealpha=0.9)
    fig.suptitle(
        f"Figure {fig_num}: Horizontal intensity scanline through the seed, with T=10/25/45/60 acceptance "
        "bands shaded around the seed value (illustrative fixed-seed bands; the implementation itself compares "
        "against the running region mean μ_R, §3.3). poor_lighting's profile stays inside even the "
        "widest (T=60) band across a long, shallow run toward the eyelid crease -- the mechanism behind its "
        "leakage at T=60. glasses_reflection's profile jumps out of every band abruptly at the glare edge, "
        "which is why it does not leak at any tested T.",
        fontsize=9.0, y=1.05,
    )
    savefig(fig, "task7_diag_intensity_profile.png")


# ===========================================================================
# Figure 20 -- pipeline block diagram
# ===========================================================================

def diagram_pipeline(fig_num: int):
    stages = [
        ("Acquisition", "IR eye-crop\ncapture / dataset"),
        ("Enhancement\n(Task 3)", "Histogram\nequalization"),
        ("Filtering\n(Task 4)", "Spatial / noise\nfiltering"),
        ("Segmentation\n(Task 6 / 7)", "Otsu / adaptive /\nedge ops; region\ngrowing; split-merge"),
        ("Feature\nextraction", "Pupil area,\naperture, EAR"),
        ("Drowsiness\ndecision", "PERCLOS-style\neye-closure logic"),
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    w, h, gap = 1.95, 1.5, 0.55
    total_w = n * w + (n - 1) * gap
    x0 = -total_w / 2 + w / 2
    xs = [x0 + i * (w + gap) for i in range(n)]
    highlight_idx = 3

    for i, (title, sub) in enumerate(stages):
        cx = xs[i]
        is_this_task = (i == highlight_idx)
        face = "#ffe4b5" if is_this_task else "#dbe8fb"
        edge_lw = 2.4 if is_this_task else 1.3
        box = FancyBboxPatch((cx - w / 2, -h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                              linewidth=edge_lw, edgecolor="#333", facecolor=face, zorder=2)
        ax.add_patch(box)
        ax.text(cx, 0.28, title, ha="center", va="center", fontsize=8.8, fontweight="bold", zorder=3)
        ax.text(cx, -0.32, sub, ha="center", va="center", fontsize=7.4, zorder=3)
        if i < n - 1:
            _arrow(ax, (cx + w / 2, 0), (xs[i + 1] - w / 2, 0))

    ax.annotate("this report", xy=(xs[highlight_idx], h / 2), xytext=(xs[highlight_idx], h / 2 + 0.65),
                ha="center", fontsize=8.6, fontweight="bold", color="#a0522d",
                arrowprops=dict(arrowstyle="-|>", color="#a0522d", lw=1.3))

    ax.set_xlim(-total_w / 2 - 0.6, total_w / 2 + 0.6)
    ax.set_ylim(-1.3, 1.9)
    ax.axis("off")
    ax.set_title(
        f"Figure {fig_num}: Where region growing sits in the driver-drowsiness pipeline. A correctly "
        "segmented pupil region feeds pupil area / eye-aperture as a drowsiness feature; leakage past the "
        "pupil boundary at this stage directly corrupts that downstream feature.",
        fontsize=9.2, pad=14,
    )
    savefig(fig, "task7_diag_pipeline.png")


# ===========================================================================
# Main
# ===========================================================================

def main():
    diagram_flowchart(14)
    diagram_connectivity(15)
    diagram_progression(16)
    diagram_quadtree(17)
    diagram_rag(18)
    diagram_intensity_profile(19)
    diagram_pipeline(20)


if __name__ == "__main__":
    main()
