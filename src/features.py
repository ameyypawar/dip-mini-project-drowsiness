"""Week-6 feature assembly: HOG + LBP texture features, plus family-wise joins
of the Week 3/4 morphology and Week 5 transform feature tables, into a single
matrix `src.classify` trains on.

Four feature families, any subset combinable via `build_feature_matrix`:

  - "morph":     the 14 `morph_*` columns of
                 `data/features/segmorph_features.csv` (Week 3/4 shape
                 features on the *cleaned* pupil/iris mask -- see
                 `src.morphology.FEATURE_NAMES`). The 3 `seg_*` columns in
                 that same CSV (pre-morphology-cleanup diagnostics) are
                 deliberately excluded: they are superseded by the cleaned
                 mask's `morph_*` features and are not part of the Week 6
                 brief's "14 morph_* features" family.
  - "transform": all 19 non-key columns of
                 `data/features/transform_features.csv` (Week 5 DFT/DCT/
                 Hough features -- see `src.transforms.FEATURE_COLUMNS`).
  - "hog":       Histogram of Oriented Gradients on the 64x64 preprocessed
                 crop (`skimage.feature.hog`). Default params
                 (orientations=9, pixels_per_cell=(8,8),
                 cells_per_block=(2,2), block_norm='L2-Hys') give 1764 dims
                 on a 64x64 crop.
  - "lbp":       Uniform Local Binary Pattern histogram (P=8, R=1,
                 method='uniform', 10 bins) on the same crop -- 10 dims.

morph/transform are *joined* from their existing CSVs (never recomputed in
bulk -- src/segment.py, src/morphology.py, src/transforms.py are read/import
only per the Week 6 brief). hog/lbp have no precomputed table, so they are
computed here from each raw image
(`source_relpath` -> `src.preprocess.preprocess_pipeline` -> hog/lbp) and
cached to `data/features/hog_lbp_cache.npz` so repeated notebook/script runs
don't recompute all 6000 crops every time.

For single-image inference (Week 7 live entry point, see
`extract_family_features_single` and `src.classify.predict_eye_state`) there
is no CSV to join against, so morph/transform features are instead computed
fresh via the same Week 3/4/5 functions (`src.segment.segment_eye`,
`src.morphology.clean_mask`/`extract_mask_features`,
`src.transforms.compute_row_features`) on the preprocessed crop -- using the
exact same column names/order as the CSVs, so a live feature vector lines up
column-for-column with what the model was trained on.

Non-finite values: checked directly -- neither `segmorph_features.csv` nor
`transform_features.csv` currently contains any NaN/inf over all 6000 rows.
The Hough "not found" case already writes an explicit sentinel 0.0 for
cx/cy/radius/n_circles, gated by `hough_found=0`, not NaN. As a defensive
measure (a different crop, a future recompute, or a degenerate HOG/LBP input
could in principle hit a genuine 0/0), `build_feature_matrix` still runs the
fully-assembled matrix through `np.nan_to_num` (nan -> 0.0, +inf -> 1e6,
-inf -> -1e6) before returning, and reports how many cells were touched so
that number is auditable rather than silently swallowed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
from skimage.feature import local_binary_pattern

from src.preprocess import DEFAULT_CONFIG, preprocess_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKING_SET_CSV = REPO_ROOT / "data" / "working_set.csv"
SEGMORPH_CSV = REPO_ROOT / "data" / "features" / "segmorph_features.csv"
TRANSFORM_CSV = REPO_ROOT / "data" / "features" / "transform_features.csv"
HOG_LBP_CACHE = REPO_ROOT / "data" / "features" / "hog_lbp_cache.npz"

FAMILIES = ("morph", "transform", "hog", "lbp")


# ---------------------------------------------------------------------------
# HOG / LBP configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HOGConfig:
    """`skimage.feature.hog` params. Defaults match the Week-6 baseline
    measurement (1764 dims on the 64x64 canonical crop): 8x8 cells ->
    8x8 grid, 2x2 blocks -> 7x7=49 block positions, 49*2*2*9=1764.
    """

    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"


@dataclass(frozen=True)
class LBPConfig:
    """`skimage.feature.local_binary_pattern` params. Defaults: uniform LBP,
    P=8 neighbours, R=1px radius, 10-bin histogram (P+2 uniform patterns:
    P+1 uniform codes 0..P plus 1 bin for all non-uniform codes).
    """

    P: int = 8
    R: int = 1
    method: str = "uniform"
    n_bins: int = 10


DEFAULT_HOG_CONFIG = HOGConfig()
DEFAULT_LBP_CONFIG = LBPConfig()


def compute_hog(img: np.ndarray, cfg: HOGConfig = DEFAULT_HOG_CONFIG) -> np.ndarray:
    """HOG feature vector for one preprocessed grayscale crop."""
    feat = skimage_hog(
        img,
        orientations=cfg.orientations,
        pixels_per_cell=cfg.pixels_per_cell,
        cells_per_block=cfg.cells_per_block,
        block_norm=cfg.block_norm,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def compute_lbp_hist(img: np.ndarray, cfg: LBPConfig = DEFAULT_LBP_CONFIG) -> np.ndarray:
    """Normalized (sums to 1) uniform-LBP histogram for one preprocessed crop."""
    lbp = local_binary_pattern(img, P=cfg.P, R=cfg.R, method=cfg.method)
    hist, _ = np.histogram(lbp.ravel(), bins=cfg.n_bins, range=(0, cfg.n_bins))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32)


# ---------------------------------------------------------------------------
# Working-set / CSV loading
# ---------------------------------------------------------------------------

def load_working_set(path: str | Path = WORKING_SET_CSV) -> list[dict]:
    """Load all rows of `data/working_set.csv` in file order (6000 rows,
    all splits). Never re-split/re-shuffle -- the `split` column already
    encodes the subject-disjoint train/val/test assignment.
    """
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_csv_by_filename(path: str | Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# HOG/LBP bulk computation with on-disk cache
# ---------------------------------------------------------------------------

def _hog_lbp_config_hash(hog_cfg: HOGConfig, lbp_cfg: LBPConfig) -> str:
    payload = json.dumps(
        {"hog": hog_cfg.__dict__, "lbp": lbp_cfg.__dict__, "preprocess": DEFAULT_CONFIG.__dict__},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_hog_lbp_matrix(
    working_set: list[dict],
    repo_root: str | Path = REPO_ROOT,
    hog_cfg: HOGConfig = DEFAULT_HOG_CONFIG,
    lbp_cfg: LBPConfig = DEFAULT_LBP_CONFIG,
    cache_path: str | Path = HOG_LBP_CACHE,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """HOG matrix (N x 1764) and LBP matrix (N x 10) for every row of
    `working_set`, in that row order. Cached to `cache_path` (npz) keyed by
    the exact filename list + a hash of (hog_cfg, lbp_cfg, preprocess
    config) -- a cache hit requires both to match exactly, else it
    recomputes (e.g. if working_set's row set/order changed).
    """
    root = Path(repo_root)
    filenames = [row["filename"] for row in working_set]
    cfg_hash = _hog_lbp_config_hash(hog_cfg, lbp_cfg)
    cache_path = Path(cache_path)

    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            str(cached["config_hash"]) == cfg_hash
            and list(cached["filenames"]) == filenames
        ):
            return cached["hog"], cached["lbp"], filenames
        print(f"  hog/lbp cache at {cache_path} is stale (config or row set changed) -- recomputing")

    n = len(working_set)
    hog_dim = compute_hog(np.zeros((DEFAULT_CONFIG.target_size, DEFAULT_CONFIG.target_size), dtype=np.uint8), hog_cfg).shape[0]
    lbp_dim = lbp_cfg.n_bins
    hog_mat = np.zeros((n, hog_dim), dtype=np.float32)
    lbp_mat = np.zeros((n, lbp_dim), dtype=np.float32)

    t0 = time.perf_counter()
    for i, row in enumerate(working_set):
        img_path = root / row["source_relpath"]
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"could not read image: {img_path}")
        enhanced = preprocess_pipeline(raw, DEFAULT_CONFIG)
        hog_mat[i] = compute_hog(enhanced, hog_cfg)
        lbp_mat[i] = compute_lbp_hist(enhanced, lbp_cfg)
        if (i + 1) % 1000 == 0:
            print(f"  hog/lbp {i + 1}/{n} ({time.perf_counter() - t0:.1f}s elapsed)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        filenames=np.array(filenames),
        hog=hog_mat,
        lbp=lbp_mat,
        config_hash=np.array(cfg_hash),
    )
    print(f"  wrote {cache_path} ({time.perf_counter() - t0:.1f}s total)")
    return hog_mat, lbp_mat, filenames


# ---------------------------------------------------------------------------
# Bulk feature matrix (train/val/test all at once, from CSVs + cache)
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    X: np.ndarray
    feature_names: list[str]
    filenames: list[str]
    n_nonfinite_imputed: int


def build_feature_matrix(
    working_set: list[dict],
    families: Sequence[str],
    repo_root: str | Path = REPO_ROOT,
    segmorph_csv: str | Path = SEGMORPH_CSV,
    transform_csv: str | Path = TRANSFORM_CSV,
    hog_cfg: HOGConfig = DEFAULT_HOG_CONFIG,
    lbp_cfg: LBPConfig = DEFAULT_LBP_CONFIG,
    cache_path: str | Path = HOG_LBP_CACHE,
    use_cache: bool = True,
) -> FeatureMatrix:
    """Assemble a feature matrix for `working_set` rows (in that row order)
    over the requested subset of `families` (any of "morph", "transform",
    "hog", "lbp"). morph/transform are joined on `filename` from their CSVs
    (imported column order -- `src.morphology.FEATURE_NAMES` /
    `src.transforms.FEATURE_COLUMNS` -- so column order matches what
    `extract_family_features_single` produces for live inference). hog/lbp
    are computed once and cached (see `compute_hog_lbp_matrix`).
    """
    families = list(families)
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {unknown}, expected subset of {FAMILIES}")
    if not families:
        raise ValueError("families must be non-empty")

    filenames = [row["filename"] for row in working_set]
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if "morph" in families:
        from src.morphology import FEATURE_NAMES as MORPH_FEATURE_NAMES

        morph_cols = [f"morph_{name}" for name in MORPH_FEATURE_NAMES]
        by_fn = _load_csv_by_filename(segmorph_csv)
        arr = np.array(
            [[float(by_fn[fn][c]) for c in morph_cols] for fn in filenames],
            dtype=np.float64,
        )
        blocks.append(arr)
        feature_names.extend(morph_cols)

    if "transform" in families:
        from src.transforms import FEATURE_COLUMNS as TRANSFORM_FEATURE_COLUMNS

        transform_cols = [c for c in TRANSFORM_FEATURE_COLUMNS if c not in ("filename", "split", "eye_state")]
        by_fn = _load_csv_by_filename(transform_csv)
        arr = np.array(
            [[float(by_fn[fn][c]) for c in transform_cols] for fn in filenames],
            dtype=np.float64,
        )
        blocks.append(arr)
        feature_names.extend(transform_cols)

    if "hog" in families or "lbp" in families:
        hog_mat, lbp_mat, cache_filenames = compute_hog_lbp_matrix(
            working_set, repo_root=repo_root, hog_cfg=hog_cfg, lbp_cfg=lbp_cfg,
            cache_path=cache_path, use_cache=use_cache,
        )
        assert cache_filenames == filenames, "hog/lbp cache row order mismatch"
        if "hog" in families:
            blocks.append(hog_mat.astype(np.float64))
            feature_names.extend([f"hog_{i}" for i in range(hog_mat.shape[1])])
        if "lbp" in families:
            blocks.append(lbp_mat.astype(np.float64))
            feature_names.extend([f"lbp_{i}" for i in range(lbp_mat.shape[1])])

    X = np.hstack(blocks)
    n_nonfinite = int((~np.isfinite(X)).sum())
    if n_nonfinite:
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return FeatureMatrix(X=X, feature_names=feature_names, filenames=filenames, n_nonfinite_imputed=n_nonfinite)


# ---------------------------------------------------------------------------
# Single-image feature extraction (Week 7 live inference path)
# ---------------------------------------------------------------------------

def extract_family_features_single(
    enhanced: np.ndarray,
    families: Sequence[str],
    hog_cfg: HOGConfig = DEFAULT_HOG_CONFIG,
    lbp_cfg: LBPConfig = DEFAULT_LBP_CONFIG,
) -> dict[str, float]:
    """Compute every requested family's features for one already-preprocessed
    64x64 crop, keyed by the same column names `build_feature_matrix` uses.

    morph/transform have no CSV to join against for a fresh single image, so
    they are computed live via the Week 3/4/5 functions themselves
    (`src.segment.segment_eye`, `src.morphology.clean_mask` /
    `extract_mask_features`, `src.transforms.compute_row_features`) -- the
    same functions `scripts/build_segmorph_features.py` and
    `src.transforms.build_feature_table` used to build the training CSVs, so
    a live vector is computed identically to how training features were
    computed, just row-by-row instead of from a cached table.
    """
    feats: dict[str, float] = {}

    if "morph" in families:
        from src.morphology import DEFAULT_MORPH_CONFIG, FEATURE_NAMES, clean_mask, extract_mask_features
        from src.segment import DEFAULT_SEGMENT_CONFIG, segment_eye

        seg = segment_eye(enhanced, DEFAULT_SEGMENT_CONFIG)
        cleaned = clean_mask(seg.region_mask, DEFAULT_MORPH_CONFIG)
        morph_feats = extract_mask_features(cleaned, min_area_frac=DEFAULT_SEGMENT_CONFIG.min_region_area_frac)
        for name in FEATURE_NAMES:
            feats[f"morph_{name}"] = float(morph_feats[name])

    if "transform" in families:
        from src.transforms import DEFAULT_HOUGH, compute_row_features

        tf = compute_row_features(enhanced, DEFAULT_HOUGH)
        for k, v in tf.items():
            feats[k] = float(v)

    if "hog" in families:
        hv = compute_hog(enhanced, hog_cfg)
        for i, v in enumerate(hv):
            feats[f"hog_{i}"] = float(v)

    if "lbp" in families:
        lv = compute_lbp_hist(enhanced, lbp_cfg)
        for i, v in enumerate(lv):
            feats[f"lbp_{i}"] = float(v)

    return feats


def vectorize_features(feats: dict[str, float], feature_names: list[str]) -> np.ndarray:
    """Order a single-image feature dict into a (1, D) row matching
    `feature_names` (the order the model was trained on), imputing any
    non-finite value to 0.0 defensively (see module docstring).
    """
    row = np.array([[feats[name] for name in feature_names]], dtype=np.float64)
    return np.nan_to_num(row, nan=0.0, posinf=1e6, neginf=-1e6)


def main() -> None:
    """Warm the HOG/LBP cache for the full working set (all 4 families)."""
    working_set = load_working_set()
    print(f"loaded {len(working_set)} rows from {WORKING_SET_CSV}")
    fm = build_feature_matrix(working_set, families=list(FAMILIES))
    print(f"assembled matrix: {fm.X.shape}, {len(fm.feature_names)} feature names, "
          f"{fm.n_nonfinite_imputed} non-finite cells imputed")


if __name__ == "__main__":
    main()
