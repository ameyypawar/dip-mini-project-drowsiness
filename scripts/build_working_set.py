"""Build the shared working set used by Weeks 3-6.

Every downstream week (segmentation, morphology, transforms, classification)
extracts features from *this* image list, so that feature CSVs can be joined on
`filename` and the week-6 classifier trains and evaluates on a consistent,
honestly-split population.

Two properties matter:

1. **Class balance.** Equal numbers of open- and closed-eye images, so accuracy
   is meaningful without reweighting.

2. **Subject-disjoint splits.** Train/val/test are split *by subject*, never by
   image. The MRL dataset contains thousands of frames per person; a random
   per-image split would put near-identical frames of the same eye in both
   train and test, and the classifier would score highly by recognising the
   person rather than the eye state. Splitting by subject is what makes the
   week-6 numbers trustworthy.

Usage:
    ./.venv/bin/python scripts/build_working_set.py
"""

from __future__ import annotations

import csv
import glob
import os
import random
from collections import defaultdict

SEED = 20231080
N_PER_CLASS = 3000
RAW_GLOB = "data/raw/mrlEyes_2018_01/s*/*.png"
OUT_CSV = "data/working_set.csv"

# Subject-level split proportions.
TRAIN_FRAC, VAL_FRAC = 0.62, 0.19  # remainder -> test


def parse(path: str) -> dict | None:
    """Decode MRL metadata from a filename, or None if it does not conform."""
    parts = os.path.basename(path)[:-4].split("_")
    if len(parts) != 8:
        return None
    return {
        "filename": os.path.basename(path),
        "subject_id": parts[0],
        "gender": parts[2],
        "glasses": parts[3],
        "eye_state": parts[4],
        "reflections": parts[5],
        "lighting": parts[6],
        "sensor": parts[7],
        "source_relpath": path,
    }


def main() -> None:
    rng = random.Random(SEED)

    rows = [r for p in glob.glob(RAW_GLOB) if (r := parse(p))]
    print(f"scanned {len(rows)} images")

    subjects = sorted({r["subject_id"] for r in rows})
    rng.shuffle(subjects)
    n_tr = int(len(subjects) * TRAIN_FRAC)
    n_va = int(len(subjects) * VAL_FRAC)
    split_of = {}
    for s in subjects[:n_tr]:
        split_of[s] = "train"
    for s in subjects[n_tr : n_tr + n_va]:
        split_of[s] = "val"
    for s in subjects[n_tr + n_va :]:
        split_of[s] = "test"

    # Sample per class, spreading the quota evenly across that split's subjects
    # so no single prolific subject dominates.
    by_split_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_split_class[(split_of[r["subject_id"]], r["eye_state"])].append(r)

    split_share = {"train": TRAIN_FRAC, "val": VAL_FRAC}
    split_share["test"] = 1.0 - TRAIN_FRAC - VAL_FRAC

    selected: list[dict] = []
    for split in ("train", "val", "test"):
        for state in ("0", "1"):
            pool = by_split_class[(split, state)]
            quota = int(N_PER_CLASS * split_share[split])

            per_subject: dict[str, list[dict]] = defaultdict(list)
            for r in pool:
                per_subject[r["subject_id"]].append(r)
            for lst in per_subject.values():
                rng.shuffle(lst)

            # Round-robin across subjects until the quota is met.
            picked, subs = [], sorted(per_subject)
            i = 0
            while len(picked) < quota and any(per_subject[s] for s in subs):
                s = subs[i % len(subs)]
                if per_subject[s]:
                    picked.append(per_subject[s].pop())
                i += 1

            for r in picked:
                r["split"] = split
            selected.extend(picked)

    rng.shuffle(selected)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cols = [
        "filename", "split", "eye_state", "subject_id", "gender", "glasses",
        "reflections", "lighting", "sensor", "source_relpath",
    ]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in selected:
            w.writerow({c: r[c] for c in cols})

    # Report, and assert the property that makes the split trustworthy.
    print(f"\nwrote {OUT_CSV}: {len(selected)} images")
    print(f"{'split':<7}{'images':>8}{'open':>8}{'closed':>8}{'subjects':>10}")
    seen: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        sel = [r for r in selected if r["split"] == split]
        subs = {r["subject_id"] for r in sel}
        seen[split] = subs
        opens = sum(r["eye_state"] == "1" for r in sel)
        print(f"{split:<7}{len(sel):>8}{opens:>8}{len(sel) - opens:>8}{len(subs):>10}")

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = seen[a] & seen[b]
        assert not overlap, f"subject leakage between {a} and {b}: {overlap}"
    print("\nOK: no subject appears in more than one split")


if __name__ == "__main__":
    main()
