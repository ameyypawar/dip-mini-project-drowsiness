# data/samples/

A small (40-image) curated subset of the **MRL Eye Dataset**, redistributed here for coursework.

## Source

- Dataset: MRL Eye Dataset
- Publisher: VSB -- Technical University of Ostrava
- URL: https://mrl.cs.vsb.cz/data/eyedataset/
- Full dataset size: 84,898 images across 37 subjects

## Citation

> Fusek, R. (2018). *MRL Eye Dataset*. VSB -- Technical University of Ostrava.
> https://mrl.cs.vsb.cz/data/eyedataset/

## Terms / license note

The MRL Eye Dataset is made available by its authors free of charge for research
and academic use. This repository redistributes a 40-image subset (20 open-eye,
20 closed-eye frames) exclusively for a Digital Image Processing coursework
mini-project, with attribution to the original authors as above. If the dataset
authors or VSB -- TU Ostrava object to this redistribution, the subset will be
removed from this repository on request (contact via the repository's
issue tracker).

## Filename convention

Reproduced from `data/raw/mrlEyes_2018_01/annotation.txt`:

```
subjectID_imageID_gender_glasses_eyeState_reflections_lightingConditions_sensorID.png

gender:      0 - male,   1 - female
glasses:     0 - no,     1 - yes
eye state:   0 - close,  1 - open
reflections: 0 - none, 1 - low, 2 - high
lighting conditions/image quality: 0 - bad, 1 - good
sensor type: 01 - RealSense SR300 640x480
             02 - IDS Imaging, 1280x1024
             03 - Aptina Imaging 752x480

example: s001_00123_0_0_0_0_0_01.png
```

Original filenames (and thus their embedded metadata) are preserved unchanged
for every copied file in `alert/` and `drowsy/`.

## Folder layout

- `alert/` -- 20 images with `eyeState = 1` (open)
- `drowsy/` -- 20 images with `eyeState = 0` (closed)
- `manifest.csv` -- one row per image: filename, class, subject_id, image_id,
  gender, glasses, eye_state, reflections, lighting, sensor, width, height,
  bytes, sha256, source_relpath

## Reproducing this selection

Selection is deterministic given a fixed random seed. With the full dataset
extracted at `data/raw/mrlEyes_2018_01/` (see `scripts/extract_dataset.sh`),
regenerate the identical 40-image subset and manifest with:

```
.venv/bin/python scripts/curate_samples.py
```

Seed used: `20231080` (set via `random.seed(20231080)` in
`scripts/curate_samples.py`).

## Important caveat: these are per-frame proxy labels, not a drowsiness judgement

**A single closed-eye frame is not, by itself, evidence of drowsiness.**

- `drowsy/` = images with `eyeState = 0` (eyes closed in that single frame)
- `alert/` = images with `eyeState = 1` (eyes open in that single frame)

Real drowsiness detection is a **temporal judgement** -- most commonly PERCLOS
(PERcentage of eye CLOSure over time), which looks at the fraction of time
the eyes are closed across a *sequence* of frames, not any single frame in
isolation. A person can blink (a normal, brief eye closure) and register as
"closed" in one frame without being remotely drowsy; conversely a driver can
be drowsy while a single sampled frame happens to show open eyes.

These two folders label the **per-frame eye-state input** that feeds into
that later temporal judgement -- they are not themselves a drowsiness label.
Any downstream classifier trained on `alert/` vs `drowsy/` per-frame images
is learning eye-state (open vs closed), and a drowsiness verdict requires
aggregating that signal over time (e.g. PERCLOS thresholding) in later
project weeks.
