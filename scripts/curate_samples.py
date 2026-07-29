#!/usr/bin/env python3
"""curate_samples.py

Selects a small, stratified, redistributable subset of the MRL Eye Dataset
for coursework (data/samples/alert, data/samples/drowsy) and writes a
manifest describing every selected file.

MRL filename convention (from data/raw/mrlEyes_2018_01/annotation.txt):

    subjectID_imageID_gender_glasses_eyeState_reflections_lightingConditions_sensorID.png

    gender:      0 - male,   1 - female
    glasses:     0 - no,     1 - yes
    eye state:   0 - close,  1 - open
    reflections: 0 - none, 1 - low, 2 - high
    lighting:    0 - bad,     1 - good
    sensor:      01 - RealSense SR300 640x480
                 02 - IDS Imaging, 1280x1024
                 03 - Aptina Imaging 752x480

Splitting the basename (without extension) on '_' yields exactly 8 fields:
    [subject_id, image_id, gender, glasses, eye_state, reflections, lighting, sensor]

Selection targets (per 20-image class, unless noted otherwise):
    - glasses:      8 with (1) / 12 without (0)
    - lighting:     10 bad (0) / 10 good (1)
    - reflections:  >= 2 of each of {0, 1, 2}
    - sensors 02 and 03: >= 2 each, counted across all 40 selected images
    - max 2 images per subject, counted across all 40 selected images
      (=> at least 20 distinct subjects across the full sample)

These are SOFT targets. Closed-eye (eyeState=0) sensor-02 images are scarce
(~60 images in the whole dataset), so full simultaneous satisfaction of every
target for the drowsy class is not guaranteed. Targets are relaxed, in order,
as follows -- and the achieved-vs-target numbers are always reported, never
hard-failed on:

    1. sensor coverage (02/03 >= 2 each, combined)   -- relaxed first
    2. reflections >= 2 each per class               -- relaxed second
    3. glasses 8/12 split                            -- relaxed third
    4. lighting 10/10 split                          -- relaxed fourth
    5. max-2-images-per-subject cap                  -- relaxed only as an
                                                          absolute last resort

Reproduction: with the dataset extracted at data/raw/mrlEyes_2018_01/, run

    .venv/bin/python scripts/curate_samples.py

The random seed (20231080) is fixed in this file, so re-running regenerates
the identical 40-image selection and manifest.
"""
from __future__ import annotations

import csv
import hashlib
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SEED = 20231080

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "raw" / "mrlEyes_2018_01"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.csv"

N_PER_CLASS = 20
GLASSES_TARGET = {"0": 12, "1": 8}
LIGHTING_TARGET = {"0": 10, "1": 10}
REFLECTIONS_TARGET = {"0": 2, "1": 2, "2": 2}
SENSOR_TARGET = {"02": 2, "03": 2}  # global, across all 40
SUBJECT_CAP = 2  # global, across all 40


@dataclass
class Record:
    path: Path
    subject_id: str
    image_id: str
    gender: str
    glasses: str
    eye_state: str
    reflections: str
    lighting: str
    sensor: str


def parse_filename(path: Path) -> Record | None:
    parts = path.stem.split("_")
    if len(parts) != 8:
        return None
    subject_id, image_id, gender, glasses, eye_state, reflections, lighting, sensor = parts
    return Record(
        path=path,
        subject_id=subject_id,
        image_id=image_id,
        gender=gender,
        glasses=glasses,
        eye_state=eye_state,
        reflections=reflections,
        lighting=lighting,
        sensor=sensor,
    )


def scan_dataset() -> tuple[list[Record], int]:
    records: list[Record] = []
    malformed = 0
    for png_path in DATASET_DIR.glob("s*/*.png"):
        rec = parse_filename(png_path)
        if rec is None:
            malformed += 1
            continue
        records.append(rec)
    return records, malformed


def greedy_select(
    pool: list[Record],
    n: int,
    subject_count: dict[str, int],
    sensor_count: dict[str, int],
    rng: random.Random,
) -> tuple[list[Record], dict[str, int], dict[str, int], dict[str, int]]:
    """Greedily select n records from pool, scoring candidates by how many
    unmet soft quotas they would satisfy. Hard constraint: SUBJECT_CAP per
    subject (relaxed only if pool is exhausted before reaching n).
    """
    remaining = list(pool)
    rng.shuffle(remaining)
    selected: list[Record] = []
    local_glasses = {"0": 0, "1": 0}
    local_lighting = {"0": 0, "1": 0}
    local_reflections = {"0": 0, "1": 0, "2": 0}

    def score(rec: Record) -> float:
        s = 0.0
        if local_reflections[rec.reflections] < REFLECTIONS_TARGET.get(rec.reflections, 0):
            s += 10.0
        if sensor_count.get(rec.sensor, 0) < SENSOR_TARGET.get(rec.sensor, 0):
            s += 8.0
        if local_glasses[rec.glasses] < GLASSES_TARGET[rec.glasses]:
            s += 5.0
        if local_lighting[rec.lighting] < LIGHTING_TARGET[rec.lighting]:
            s += 5.0
        s += rng.random()  # deterministic tie-breaker given fixed seed
        return s

    def commit(rec: Record) -> None:
        selected.append(rec)
        subject_count[rec.subject_id] = subject_count.get(rec.subject_id, 0) + 1
        sensor_count[rec.sensor] = sensor_count.get(rec.sensor, 0) + 1
        local_glasses[rec.glasses] += 1
        local_lighting[rec.lighting] += 1
        local_reflections[rec.reflections] += 1

    while len(selected) < n and remaining:
        best_idx = None
        best_score = -1.0
        for idx, rec in enumerate(remaining):
            if subject_count.get(rec.subject_id, 0) >= SUBJECT_CAP:
                continue
            sc = score(rec)
            if sc > best_score:
                best_score = sc
                best_idx = idx
        if best_idx is None:
            break  # subject cap exhausted the eligible pool
        commit(remaining.pop(best_idx))

    if len(selected) < n:
        # Last-resort relaxation: subject cap.
        for rec in remaining:
            if len(selected) >= n:
                break
            commit(rec)

    return selected, local_glasses, local_lighting, local_reflections


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not DATASET_DIR.is_dir():
        print(f"ERROR: dataset dir not found: {DATASET_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {DATASET_DIR} ...")
    records, malformed = scan_dataset()
    print(f"Parsed {len(records)} images, skipped {malformed} malformed filenames.")

    closed_pool = [r for r in records if r.eye_state == "0"]
    open_pool = [r for r in records if r.eye_state == "1"]
    print(f"Pool sizes: closed(eye_state=0)={len(closed_pool)}  open(eye_state=1)={len(open_pool)}")

    rng = random.Random(SEED)
    subject_count: dict[str, int] = {}
    sensor_count: dict[str, int] = {}

    # Closed (drowsy) first: sensor-02 closed-eye frames are scarce, so give
    # them first claim on the global sensor-coverage quota.
    drowsy_selected, drowsy_glasses, drowsy_lighting, drowsy_reflections = greedy_select(
        closed_pool, N_PER_CLASS, subject_count, sensor_count, rng
    )
    alert_selected, alert_glasses, alert_lighting, alert_reflections = greedy_select(
        open_pool, N_PER_CLASS, subject_count, sensor_count, rng
    )

    if len(drowsy_selected) < N_PER_CLASS or len(alert_selected) < N_PER_CLASS:
        print(
            f"WARNING: could only select {len(drowsy_selected)} drowsy / "
            f"{len(alert_selected)} alert images (subject-cap pool exhaustion).",
            file=sys.stderr,
        )

    (SAMPLES_DIR / "alert").mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / "drowsy").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for class_name, selected in (("alert", alert_selected), ("drowsy", drowsy_selected)):
        dest_dir = SAMPLES_DIR / class_name
        for rec in selected:
            dest_path = dest_dir / rec.path.name
            shutil.copy2(rec.path, dest_path)
            with Image.open(dest_path) as im:
                width, height = im.size
            manifest_rows.append(
                {
                    "filename": rec.path.name,
                    "class": class_name,
                    "subject_id": rec.subject_id,
                    "image_id": rec.image_id,
                    "gender": rec.gender,
                    "glasses": rec.glasses,
                    "eye_state": rec.eye_state,
                    "reflections": rec.reflections,
                    "lighting": rec.lighting,
                    "sensor": rec.sensor,
                    "width": width,
                    "height": height,
                    "bytes": dest_path.stat().st_size,
                    "sha256": sha256_of(dest_path),
                    "source_relpath": str(rec.path.relative_to(REPO_ROOT)),
                }
            )

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "class", "subject_id", "image_id", "gender", "glasses",
                "eye_state", "reflections", "lighting", "sensor", "width", "height",
                "bytes", "sha256", "source_relpath",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} rows to {MANIFEST_PATH}")

    # --- Coverage report ---
    distinct_subjects = len(subject_count)
    print("\n=== Achieved coverage ===")
    print(f"Distinct subjects across all {len(alert_selected) + len(drowsy_selected)} images: {distinct_subjects} (target >= 20)")
    print(f"Max images for any single subject: {max(subject_count.values()) if subject_count else 0} (cap target = {SUBJECT_CAP})")

    print("\nSensor coverage (global, target >=2 for 02 and 03):")
    for sensor in ("01", "02", "03"):
        print(f"  sensor {sensor}: {sensor_count.get(sensor, 0)}")

    for class_name, glasses_c, lighting_c, reflections_c in (
        ("alert", alert_glasses, alert_lighting, alert_reflections),
        ("drowsy", drowsy_glasses, drowsy_lighting, drowsy_reflections),
    ):
        print(f"\n{class_name} class (n={glasses_c['0'] + glasses_c['1']}):")
        print(f"  glasses    no(0)={glasses_c['0']:2d} (target {GLASSES_TARGET['0']})   yes(1)={glasses_c['1']:2d} (target {GLASSES_TARGET['1']})")
        print(f"  lighting   bad(0)={lighting_c['0']:2d} (target {LIGHTING_TARGET['0']})   good(1)={lighting_c['1']:2d} (target {LIGHTING_TARGET['1']})")
        print(
            f"  reflections none(0)={reflections_c['0']:2d}  low(1)={reflections_c['1']:2d}  "
            f"high(2)={reflections_c['2']:2d}  (target >=2 each)"
        )

    unmet = []
    for sensor, target in SENSOR_TARGET.items():
        if sensor_count.get(sensor, 0) < target:
            unmet.append(f"sensor {sensor}: got {sensor_count.get(sensor, 0)}, wanted >={target}")
    for class_name, reflections_c in (("alert", alert_reflections), ("drowsy", drowsy_reflections)):
        for refl, target in REFLECTIONS_TARGET.items():
            if reflections_c[refl] < target:
                unmet.append(f"{class_name} reflections={refl}: got {reflections_c[refl]}, wanted >={target}")
    for class_name, glasses_c in (("alert", alert_glasses), ("drowsy", drowsy_glasses)):
        for g, target in GLASSES_TARGET.items():
            if glasses_c[g] != target:
                unmet.append(f"{class_name} glasses={g}: got {glasses_c[g]}, wanted =={target}")
    for class_name, lighting_c in (("alert", alert_lighting), ("drowsy", drowsy_lighting)):
        for lt, target in LIGHTING_TARGET.items():
            if lighting_c[lt] != target:
                unmet.append(f"{class_name} lighting={lt}: got {lighting_c[lt]}, wanted =={target}")

    print("\n=== Targets not fully met (soft targets, relaxed per declared priority order) ===")
    if unmet:
        for line in unmet:
            print(f"  - {line}")
    else:
        print("  (none - all soft targets met)")


if __name__ == "__main__":
    main()
