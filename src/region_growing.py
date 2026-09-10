"""Task 7 -- Region Growing, Connectivity Comparison, and Region Splitting/Merging.

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Operates on the same grayscale `uint8` `np.ndarray` convention as
`src.enhancement` / `src.segment` / `src.edge_segmentation`. Additive and
self-contained: does not modify any existing module.

Contents:
  A. Region growing   -- region_grow (BFS from a seed, comparing each
                          candidate neighbour against the *running* region
                          mean, updated as pixels are absorbed), auto_seed
                          (darkest point after a light Gaussian blur).
  B. Region splitting/merging -- split_merge: a quadtree split on a
                          homogeneity criterion (max-min range per block),
                          followed by a union-find merge of adjacent leaves
                          whose mean intensities are close.

Why the pupil is the region-of-interest here: in these near-IR eye crops the
pupil/iris is the darkest, most compact, roughly circular structure against
brighter sclera and skin -- exactly the seed-and-grow shape region growing
is suited to, and its size/shape over time is the geometric signal PERCLOS-
style drowsiness detection (see src/perclos.py) ultimately depends on.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# A. Region growing
# ---------------------------------------------------------------------------

_CONN4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_CONN8 = _CONN4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def auto_seed(img: np.ndarray, blur_ksize: int = 5) -> tuple[int, int]:
    """Darkest point after a small Gaussian blur (avoids seeding on a single
    hot/cold noise pixel). Returns (row, col).
    """
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    idx = int(np.argmin(blurred))
    r, c = np.unravel_index(idx, blurred.shape)
    return int(r), int(c)


def region_grow(
    img: np.ndarray,
    seed: tuple[int, int],
    T: float,
    connectivity: int = 8,
    update_mean: bool = True,
) -> tuple[np.ndarray, int]:
    """Grow a region from `seed` by BFS, absorbing a neighbour whenever it is
    within `T` intensity units of the *running* region mean.

    Parameters
    ----------
    img : uint8 grayscale image.
    seed : (row, col) starting pixel.
    T : intensity-difference tolerance against the running region mean.
    connectivity : 4 or 8.
    update_mean : if True (assignment's "update the region mean if
        required"), the comparison mean is recomputed after every pixel is
        absorbed; if False, every candidate is compared only against the
        original seed intensity.

    Returns
    -------
    (mask, count) -- uint8 {0,255} mask of the grown region and its pixel
    count.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    h, w = img.shape[:2]
    sr, sc = seed
    if not (0 <= sr < h and 0 <= sc < w):
        raise ValueError(f"seed {seed} outside image bounds {img.shape[:2]}")

    neighbours = _CONN8 if connectivity == 8 else _CONN4
    img_i = img.astype(np.int32)

    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)

    visited[sr, sc] = True
    mask[sr, sc] = 255
    region_sum = float(img_i[sr, sc])
    region_count = 1
    mean = region_sum / region_count

    q: deque[tuple[int, int]] = deque([(sr, sc)])
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
                    if update_mean:
                        mean = region_sum / region_count
                    q.append((nr, nc))
    return mask, region_count


# ---------------------------------------------------------------------------
# B. Region splitting and merging (quadtree split + union-find merge)
# ---------------------------------------------------------------------------

def _homogeneous(block: np.ndarray, threshold: float) -> bool:
    """Homogeneity test: max-min intensity range within the block."""
    return bool(int(block.max()) - int(block.min()) <= threshold)


def _quadtree_split(
    img: np.ndarray, r0: int, r1: int, c0: int, c1: int,
    min_size: int, threshold: float, leaves: list[tuple[int, int, int, int]],
) -> None:
    h, w = r1 - r0, c1 - c0
    block = img[r0:r1, c0:c1]
    if h <= min_size or w <= min_size or _homogeneous(block, threshold):
        leaves.append((r0, r1, c0, c1))
        return
    rm, cm = r0 + h // 2, c0 + w // 2
    if rm == r0 or rm == r1 or cm == c0 or cm == c1:
        leaves.append((r0, r1, c0, c1))
        return
    _quadtree_split(img, r0, rm, c0, cm, min_size, threshold, leaves)
    _quadtree_split(img, r0, rm, cm, c1, min_size, threshold, leaves)
    _quadtree_split(img, rm, r1, c0, cm, min_size, threshold, leaves)
    _quadtree_split(img, rm, r1, cm, c1, min_size, threshold, leaves)


def _leaves_adjacent(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    r0a, r1a, c0a, c1a = a
    r0b, r1b, c0b, c1b = b
    row_overlap = not (r1a <= r0b or r1b <= r0a)
    col_overlap = not (c1a <= c0b or c1b <= c0a)
    row_touch = r1a == r0b or r1b == r0a
    col_touch = c1a == c0b or c1b == c0a
    return (row_overlap and col_touch) or (col_overlap and row_touch)


def split_merge(
    img: np.ndarray,
    min_size: int = 16,
    homogeneity_threshold: float = 20,
    merge_tolerance: float | None = None,
) -> dict:
    """Quadtree split on a max-min homogeneity criterion, then union-find
    merge of adjacent leaves whose mean intensities are within
    `merge_tolerance` (defaults to `homogeneity_threshold`).

    Returns a dict with:
      leaves          -- list of (r0, r1, c0, c1) leaf bounding boxes
      leaf_means      -- mean intensity per leaf, same order as `leaves`
      labels          -- int32 (h, w) label map after merging (1..n_regions)
      n_leaves        -- number of leaves after splitting
      n_regions       -- number of regions after merging
    """
    h, w = img.shape[:2]
    leaves: list[tuple[int, int, int, int]] = []
    _quadtree_split(img, 0, h, 0, w, min_size, homogeneity_threshold, leaves)
    if merge_tolerance is None:
        merge_tolerance = homogeneity_threshold

    n = len(leaves)
    means = [float(img[r0:r1, c0:c1].mean()) for (r0, r1, c0, c1) in leaves]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if abs(means[i] - means[j]) <= merge_tolerance and _leaves_adjacent(leaves[i], leaves[j]):
                union(i, j)

    labels = np.zeros((h, w), dtype=np.int32)
    root_to_label: dict[int, int] = {}
    next_label = 1
    for i, (r0, r1, c0, c1) in enumerate(leaves):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[r0:r1, c0:c1] = root_to_label[root]

    return {
        "leaves": leaves,
        "leaf_means": means,
        "labels": labels,
        "n_leaves": n,
        "n_regions": len(root_to_label),
    }
