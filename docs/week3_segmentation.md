# Week 3 — Segmentation

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

Full evidence (side-by-side masks, Otsu histogram with threshold marked, glasses/reflection
failure grid) is in `notebooks/03_segmentation.ipynb`, executed top-to-bottom with stored
outputs. This document is the condensed report: what was measured, what was chosen, and why.
All numbers below are computed over the **full 6000-image working set**
(`data/working_set.csv`), not a hand-picked subset, unless stated otherwise.

## Intensity assumption

After `src.preprocess.preprocess_pipeline` (Week 2: resize -> glare suppression -> gamma ->
CLAHE), the **pupil/iris is the darkest structure** in these near-IR MRL crops and the
sclera/skin is comparatively bright. Every mask in `src/segment.py` therefore uses the
`*_INV` threshold variant: pixels **below** the threshold are foreground (255).

## Method comparison — Otsu vs. adaptive mean vs. adaptive Gaussian

Run over all 6000 working-set images, measuring connected-component count (fragmentation/
speckle proxy) and per-image timing:

| method | mean n_components | median n_components | mean dark-area fraction | time/image |
|---|---|---|---|---|
| **Otsu (global)** | **5.82** | 4.0 | 0.499 | 118.9 us |
| Adaptive mean (block=15, C=3) | 28.07 | 23.0 | 0.357 | 155.7 us |
| Adaptive Gaussian (block=15, C=3) | 61.87 | 55.0 | 0.292 | 212.3 us |

A block-size/C sweep (7/11/15/21 x 2/3/5, 600-image sample) confirmed this is not
parameter-sensitive: at every combination tried, Otsu's raw-mask fragmentation stayed flat
around 6 components while both adaptive variants ranged from ~18 to ~127 components,
consistently 3-20x more fragmented than Otsu regardless of block size or C.

**Otsu (global) is the chosen default.** It produces a mask with roughly 5-11x fewer connected
components at comparable-or-better speed. This is explainable: Week 2's CLAHE already
normalizes *local* contrast across 4x4 tiles, so by the time thresholding runs, a single
*global* dark/bright split is meaningful across the whole 64x64 crop — while adaptive
thresholding's local neighbourhood statistic instead reacts to CLAHE-amplified local texture
and noise, fragmenting what should be one smooth region. The two techniques actively work
against each other on this pipeline; global Otsu is the one that matches how the image was
already conditioned.

## Region isolation and the closed-eye degenerate case

**Honest first finding:** a plain Otsu split of the *whole* 64x64 frame is close to a 50/50
partition **regardless of eye state** — mean dark-area fraction 0.501 (closed) vs. 0.497
(open), essentially indistinguishable. MRL crops are already tightly bound around the eye, so
Otsu's global sclera/iris split does not, by itself, isolate a pupil-sized region; both open
and closed eyes have roughly half-dark, half-bright 64x64 frames (eyelid crease/eyelash texture
is dark enough to fill the same fraction as an open iris+shadow).

To recover a meaningful pupil candidate, region isolation (`isolate_dark_region`) is anchored
on a **stricter, Otsu-derived threshold**: `otsu_value * pupil_threshold_scale`. A 2D sweep of
`pupil_threshold_scale` in `{0.6, 0.65, 0.7, 0.75, 0.8}` and `min_region_area_frac` in
`{0.008, 0.01, 0.015, 0.02, 0.03}` (full 6000 images per cell) measured the degenerate-mask
rate per eye state and a naive "predict closed iff degenerate" accuracy:

| scale | min_area | deg_rate(open) | deg_rate(closed) | gap | naive accuracy |
|---|---|---|---|---|---|
| 0.60 | 0.010 | 0.739 | 0.811 | 0.072 | 0.536 |
| 0.65 | 0.010 | 0.587 | 0.710 | 0.123 | 0.562 |
| **0.70** | **0.010** | **0.338** | **0.518** | **0.180** | **0.590** |
| 0.70 | 0.008 | 0.274 | 0.461 | 0.188 | 0.594 |
| 0.75 | 0.015 | 0.204 | 0.379 | 0.174 | 0.587 |
| 0.80 | 0.020 | 0.087 | 0.188 | 0.101 | 0.550 |

**Chosen defaults: `pupil_threshold_scale=0.7`, `min_region_area_frac=0.01`** — the best
naive-accuracy operating point found in the sweep (0.590 over all 6000 images), with a real
18-point gap in degenerate rate between classes.

**Honest limitation:** this is a *modest* signal, not a solved classifier — naive accuracy sits
only a little above chance-level 0.50. Closed eyes are more likely to produce an empty/
degenerate mask than open eyes, and that emptiness is a real, usable signal, but a large
fraction of closed eyes still pass the area gate (eyelash strokes can be large enough) and a
meaningful fraction of open eyes still fail it (small/dim pupils under bad lighting or glasses
glare). Week 4's shape features on the cleaned mask (solidity, extent — not just area) recover
more of the real separating power; see `docs/week4_morphology.md`.

## Failure cases: eyeglass reflections

The curated 40-image sample (`data/samples/`) has 8 images that are simultaneously
`glasses=1` and `reflections=2` (high reflection) — the hardest case for this pipeline, since
Week 2's glare suppression only approximately repairs the specular hotspot before segmentation
runs. Visualized in notebook Section 5: results are mixed and reported honestly — several
crops segment reasonably (inpainting did its job), but others produce a degenerate mask or a
visibly noisier/smaller region than a glasses-free image of the same eye state would. No claim
of uniformly clean segmentation on this subset is made.

## What this locks in for Week 4

`src/segment.py`'s `SegmentConfig` defaults, used by `scripts/build_segmorph_features.py`:

```python
SegmentConfig(
    method="otsu",
    adaptive_block_size=15,
    adaptive_C=3,
    pupil_threshold_scale=0.7,
    min_region_area_frac=0.01,
)
```

`segment_eye(img, cfg)` returns `raw_mask` (general Otsu/adaptive split, for method comparison),
`pupil_mask_raw` (stricter Otsu-anchored dark-core mask), and `region_mask` (largest-component
isolation of `pupil_mask_raw`, empty if degenerate). Week 4 (`src/morphology.py`) takes
`region_mask` and cleans it further (opening -> closing -> hole-fill) before any shape feature
is trusted.
