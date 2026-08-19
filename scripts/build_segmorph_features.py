"""Build the Week-3/4 segmentation + morphology feature table.

Runs the full classical-DIP pipeline on every image in `data/working_set.csv`
(6000 rows, both splits, all eye states):

    raw crop -> src.preprocess.preprocess_pipeline (Week 2, 64x64 enhanced)
             -> src.segment.segment_eye            (Week 3, Otsu-anchored
                pupil isolation, degenerate-aware)
             -> src.morphology.clean_mask           (Week 4, opening ->
                closing -> hole-fill)
             -> src.morphology.extract_mask_features (Week 4, regionprops
                shape/position features on the cleaned mask)

One row per image, keyed on `filename` so Week 6 can join this table against
the parallel Week-5 transform-feature table. Output:
`data/features/segmorph_features.csv`.

Usage:
    ./.venv/bin/python scripts/build_segmorph_features.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import PreprocessConfig, preprocess_pipeline
from src.segment import DEFAULT_SEGMENT_CONFIG, segment_eye
from src.morphology import DEFAULT_MORPH_CONFIG, FEATURE_NAMES, clean_mask, extract_mask_features

WORKING_SET_CSV = REPO_ROOT / "data" / "working_set.csv"
OUT_CSV = REPO_ROOT / "data" / "features" / "segmorph_features.csv"

SEG_COLS = ["seg_otsu_threshold", "seg_dark_area_frac", "seg_is_degenerate"]
MORPH_COLS = [f"morph_{name}" for name in FEATURE_NAMES]
OUT_COLS = ["filename", "split", "eye_state"] + SEG_COLS + MORPH_COLS


def process_one(source_relpath: str) -> dict:
    raw = cv2.imread(str(REPO_ROOT / source_relpath), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(source_relpath)

    img = preprocess_pipeline(raw, PreprocessConfig())
    seg = segment_eye(img, DEFAULT_SEGMENT_CONFIG)
    cleaned = clean_mask(seg.region_mask, DEFAULT_MORPH_CONFIG)
    morph_feats = extract_mask_features(cleaned, min_area_frac=DEFAULT_SEGMENT_CONFIG.min_region_area_frac)

    row = {
        "seg_otsu_threshold": seg.otsu_value,
        "seg_dark_area_frac": seg.dark_area_frac,
        "seg_is_degenerate": int(seg.is_degenerate),
    }
    row.update({f"morph_{k}": v for k, v in morph_feats.items()})
    return row


def main() -> None:
    with WORKING_SET_CSV.open(newline="") as fh:
        working_set = list(csv.DictReader(fh))
    print(f"loaded {len(working_set)} rows from {WORKING_SET_CSV}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n_errors = 0
    with OUT_CSV.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUT_COLS)
        writer.writeheader()
        for i, ws_row in enumerate(working_set):
            try:
                feats = process_one(ws_row["source_relpath"])
            except Exception as exc:  # noqa: BLE001 -- report and continue, do not abort a 6000-row run
                print(f"  ERROR on {ws_row['filename']}: {exc}")
                n_errors += 1
                continue

            out_row = {
                "filename": ws_row["filename"],
                "split": ws_row["split"],
                "eye_state": ws_row["eye_state"],
            }
            out_row.update(feats)
            writer.writerow(out_row)

            if (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i + 1}/{len(working_set)} done ({elapsed:.1f}s elapsed)")

    elapsed = time.perf_counter() - t0
    n_written = len(working_set) - n_errors
    print(f"\nwrote {OUT_CSV}: {n_written} rows ({n_errors} errors)")
    print(f"total time: {elapsed:.2f}s ({elapsed / len(working_set) * 1000:.2f}ms/image)")


if __name__ == "__main__":
    main()
