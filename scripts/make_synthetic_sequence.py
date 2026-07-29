#!/usr/bin/env python3
"""make_synthetic_sequence.py -- Week 7: build a labelled SYNTHETIC frame
sequence to validate PERCLOS before real driving footage exists.

*** This is a CONSTRUCTED sequence, not real driving footage. ***
Every frame is a genuine MRL Eye Dataset still (real camera capture of a
real eye), but the *order* and *timing* of frames -- which real stills get
shown for how long, in what sequence -- is authored by this script to
simulate a plausible drowsiness episode: alert periods, normal blinks
(short closures), and sustained microsleeps (long closures). The subject
was not actually blinking or dozing off in this order; that continuity is
synthetic. See docs/week7_live_inference.md for why ("synthetic now, real
footage later" -- validates the PERCLOS *mechanism* today; EAR/MAR/PERCLOS
on an actual driver's face is unvalidated until real footage is recorded).

Because MRL images are isolated eye crops with no face, this sequence
only carries a per-frame open/closed *eye-state* ground truth -- it feeds
the Week-6 classifier signal path through PERCLOS, not the EAR path
(EAR/MAR need face landmarks; see src/ear.py's module docstring).

Output (data/synthetic/, unless the video would exceed the ~10MB commit
budget -- see main()'s size check and printed report):
  - data/synthetic/sequence_manifest.csv: one row per frame -- frame_idx,
    timestamp_s, subject_id, segment_state ("alert"/"blink"/"microsleep"),
    ground_truth_label ("open"/"closed"), source_relpath (the real MRL
    still shown at that frame).
  - data/synthetic/sequence.mp4: the rendered demo video (grayscale eye
    crops, resized for visibility, no text overlay baked in -- overlays
    are `scripts/run_demo.py`'s job so that script gets genuinely
    exercised on this file).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "raw" / "mrlEyes_2018_01"
OUT_DIR = REPO_ROOT / "data" / "synthetic"
MANIFEST_PATH = OUT_DIR / "sequence_manifest.csv"
VIDEO_PATH = OUT_DIR / "sequence.mp4"
COMMIT_SIZE_BUDGET_BYTES = 10 * 1024 * 1024  # ~10MB, per project brief

DEFAULT_SUBJECTS = ["s0001", "s0030"]  # both have >=1000 frames in each
                                        # eye state (verified by directory
                                        # scan) -- enough for visual
                                        # variety without heavy repetition
SEED = 20231080  # same seed as scripts/curate_samples.py, for consistency
FRAME_SIZE = 200  # display resolution per frame (upscaled from ~60-140px
                   # native MRL crops for visibility in the demo video)

# One driver "session" plan per subject: (segment_state, duration_seconds).
# segment_state in {"alert", "blink", "microsleep"}; "alert" -> ground
# truth "open", "blink" and "microsleep" -> ground truth "closed". This is
# the authored timeline described in the module docstring: alert period,
# a few normal (short) blinks, a sustained microsleep, recovery, a second
# (longer) microsleep episode near the end, and a final alert period.
SESSION_PLAN: list[tuple[str, float]] = [
    ("alert", 5.0),
    ("blink", 0.2),
    ("alert", 3.0),
    ("blink", 0.2),
    ("alert", 2.0),
    ("blink", 0.3),
    ("alert", 4.0),
    ("microsleep", 2.5),
    ("alert", 3.0),
    ("blink", 0.2),
    ("alert", 2.0),
    ("microsleep", 3.0),
    ("alert", 4.0),
]

STATE_TO_LABEL = {"alert": "open", "blink": "closed", "microsleep": "closed"}


@dataclass
class SubjectPool:
    subject_id: str
    open_paths: list[Path]
    closed_paths: list[Path]


def scan_subject(subject_id: str) -> SubjectPool:
    subj_dir = DATASET_DIR / subject_id
    if not subj_dir.is_dir():
        raise FileNotFoundError(f"subject dir not found: {subj_dir}")
    open_paths, closed_paths = [], []
    for p in subj_dir.glob("*.png"):
        parts = p.stem.split("_")
        if len(parts) != 8:
            continue
        eye_state = parts[4]
        (open_paths if eye_state == "1" else closed_paths).append(p)
    if not open_paths or not closed_paths:
        raise ValueError(f"{subject_id}: need both open and closed frames, got open={len(open_paths)} closed={len(closed_paths)}")
    return SubjectPool(subject_id, sorted(open_paths), sorted(closed_paths))


@dataclass
class SyntheticFrame:
    frame_idx: int
    timestamp_s: float
    subject_id: str
    segment_state: str
    ground_truth_label: str
    source_path: Path


def build_sequence(
    pool: SubjectPool, fps: float, plan: list[tuple[str, float]], rng: random.Random,
) -> list[SyntheticFrame]:
    """Expand `plan`'s (state, duration) segments into one SyntheticFrame
    per output frame at `fps`, sampling *with* replacement (but shuffled,
    so consecutive frames rarely repeat the exact same still) from the
    subject's real open/closed pool for each frame's ground-truth state.
    """
    frames: list[SyntheticFrame] = []
    t = 0.0
    frame_idx = 0
    for state, duration_s in plan:
        label = STATE_TO_LABEL[state]
        source_pool = pool.open_paths if label == "open" else pool.closed_paths
        n_frames = max(1, round(duration_s * fps))
        # sample without-replacement chunks, reshuffling and refilling
        # when exhausted, so a short pool still varies across a long
        # segment without ever needing eye_state != label.
        chosen: list[Path] = []
        while len(chosen) < n_frames:
            batch = list(source_pool)
            rng.shuffle(batch)
            chosen.extend(batch)
        chosen = chosen[:n_frames]

        for src in chosen:
            frames.append(SyntheticFrame(
                frame_idx=frame_idx,
                timestamp_s=round(t, 4),
                subject_id=pool.subject_id,
                segment_state=state,
                ground_truth_label=label,
                source_path=src,
            ))
            frame_idx += 1
            t += 1.0 / fps

    return frames


def render_video(frames: list[SyntheticFrame], fps: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (FRAME_SIZE, FRAME_SIZE), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
    try:
        for f in frames:
            img = cv2.imread(str(f.source_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"could not read {f.source_path}")
            resized = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC)
            writer.write(resized)
    finally:
        writer.release()


def write_manifest(frames: list[SyntheticFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_idx", "timestamp_s", "subject_id", "segment_state",
            "ground_truth_label", "source_relpath",
        ])
        writer.writeheader()
        for fr in frames:
            writer.writerow({
                "frame_idx": fr.frame_idx,
                "timestamp_s": fr.timestamp_s,
                "subject_id": fr.subject_id,
                "segment_state": fr.segment_state,
                "ground_truth_label": fr.ground_truth_label,
                "source_relpath": str(fr.source_path.relative_to(REPO_ROOT)),
            })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=str, default=",".join(DEFAULT_SUBJECTS),
                    help="comma-separated MRL subject IDs to concatenate into one session")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not DATASET_DIR.is_dir():
        print(f"ERROR: MRL dataset not found at {DATASET_DIR} (see scripts/extract_dataset.sh)", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    subject_ids = [s.strip() for s in args.subjects.split(",") if s.strip()]

    all_frames: list[SyntheticFrame] = []
    t_offset = 0.0
    for subject_id in subject_ids:
        pool = scan_subject(subject_id)
        seq = build_sequence(pool, args.fps, SESSION_PLAN, rng)
        base_idx = len(all_frames)
        for i, fr in enumerate(seq):
            fr.timestamp_s = round(fr.timestamp_s + t_offset, 4)
            fr.frame_idx = base_idx + i
        all_frames.extend(seq)
        t_offset = all_frames[-1].timestamp_s + 1.0 / args.fps
        print(f"{subject_id}: {len(seq)} frames ({len(pool.open_paths)} open / {len(pool.closed_paths)} closed source stills)")

    manifest_path = args.out_dir / "sequence_manifest.csv"
    video_path = args.out_dir / "sequence.mp4"
    write_manifest(all_frames, manifest_path)
    print(f"wrote manifest: {manifest_path} ({len(all_frames)} frames)")

    render_video(all_frames, args.fps, video_path)
    size_bytes = video_path.stat().st_size
    print(f"wrote video: {video_path} ({size_bytes / 1024:.1f} KB)")

    n_open = sum(1 for f in all_frames if f.ground_truth_label == "open")
    n_closed = len(all_frames) - n_open
    duration_s = all_frames[-1].timestamp_s + 1.0 / args.fps
    print(f"\nsession duration: {duration_s:.1f}s at {args.fps} fps, {len(all_frames)} frames")
    print(f"ground truth: open={n_open} ({n_open/len(all_frames):.1%})  closed={n_closed} ({n_closed/len(all_frames):.1%})")

    if size_bytes > COMMIT_SIZE_BUDGET_BYTES:
        print(
            f"\nWARNING: {video_path} is {size_bytes/1e6:.1f} MB, over the "
            f"~10MB commit budget. Move it out of the repo (e.g. a "
            f"gitignored path) before committing -- do NOT commit as-is.",
            file=sys.stderr,
        )
    else:
        print(f"\nOK to commit: {size_bytes/1e6:.2f} MB < 10MB budget.")


if __name__ == "__main__":
    main()
