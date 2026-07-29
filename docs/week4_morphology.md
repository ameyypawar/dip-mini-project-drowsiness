# Week 4 — Morphological Operations

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

Full evidence (before/after cleanup grids, structuring-element sweeps, feature-separability box
plots) is in `notebooks/04_morphology.ipynb`, executed top-to-bottom with stored outputs. This
document is the condensed report. Numbers are measured either on an 800-image random sample of
the working set (structuring-element sweeps — cheap enough to sweep many configurations) or the
**full 6000-image** feature table `data/features/segmorph_features.csv` (separability
analysis).

## Why morphology, on top of Week 3's isolation

Week 3's `segment_eye(...).pupil_mask_raw` (the stricter, Otsu-anchored dark-core mask, *before*
largest-component isolation) is fragmented: mean **5.25** connected components over an
800-image sample, and **84%** of images have more than one component. Morphology cleans this up
before any shape is trusted.

## Opening — speckle removal, structuring-element sweep

Measured mean component count after `opening(pupil_mask_raw, kernel)` and mean fraction of
original foreground area retained, across shape x size:

| shape | size | mean n_components after open | mean area retained |
|---|---|---|---|
| ellipse | 2 | 3.38 | 0.812 |
| **ellipse** | **3** | **2.58** | **0.664** |
| ellipse | 5 | 1.09 | 0.336 |
| ellipse | 7 | 0.56 | 0.173 |
| rect | 3 | 2.09 | 0.551 |
| cross | 3 | 2.58 | 0.664 |

(raw, pre-opening baseline: mean 5.25 components)

Rect and cross kernels behave almost identically to ellipse at this tiny (2-7px) scale on a
64x64 image — the shape choice barely matters here, only the size does. Larger kernels remove
more components but also destroy more real signal: a 5x5 ellipse only retains 33.6% of original
area (over-erosion — almost certainly eating real pupil pixels along with noise), while a 3x3
retains 66.4% while still roughly halving the component count (5.25 -> 2.58).

**Chosen: `open_shape="ellipse", open_size=3`** — best measured speckle-removal-vs-signal
tradeoff. Ellipse is kept over rect/cross as the more standard, isotropic choice for a target
that should be roughly circular (pupil).

## Closing — gap bridging, size sweep (applied after the chosen opening)

| close size (ellipse) | mean n_components after close | mean area ratio (vs. post-open) |
|---|---|---|
| 2 | 2.54 | 0.958 |
| **5** | **2.23** | **1.050** |
| 7 | 2.16 | 1.069 |
| 9 | 2.03 | 1.107 |

(post-opening baseline: mean 2.58 components)

**Chosen: `close_shape="ellipse", close_size=5`.** Component count keeps dropping through 5px
while area only inflates modestly (1.05x) — evidence closing is bridging genuinely nearby
fragments of the same blob, not swallowing unrelated dark regions elsewhere in the frame.
7-9px closes merge slightly more but at a less conservative area-inflation cost.

## Hole filling

Measured on the 800-image sample (after opening + 5x5 closing): `fill_holes` changed only
**5/800** images, adding a mean of **0.12 pixels** per image. Honest reading: Week 2's glare
suppression (threshold -> dilate -> inpaint) already repairs most specular-highlight gaps
*before* segmentation ever runs, so there is little left for hole filling to do on this
particular dataset. It is kept in `clean_mask` for completeness — the rare image that still has
an enclosed gap, and any future input (e.g. a live webcam ROI) where glare suppression might be
disabled or less effective.

**Chosen `MorphConfig` (locked default):**

```python
MorphConfig(
    open_shape="ellipse", open_size=3,
    close_shape="ellipse", close_size=5,
    do_fill_holes=True,
)
```

`clean_mask(mask, cfg)` composes these in order: opening -> closing -> hole filling. Opening
runs first so speckle is gone *before* closing has a chance to bridge that same noise into the
main blob.

## Feature extraction and separability (full 6000 images)

`extract_mask_features` runs `skimage.measure.regionprops` on `clean_mask` output and reports
14 features per image (`src/morphology.py::FEATURE_NAMES`): `n_components`,
`dark_area_ratio`, `largest_area`, `largest_area_ratio`, `filled_area`, `filled_area_ratio`,
`centroid_row_norm`, `centroid_col_norm`, `eccentricity`, `solidity`, `extent`,
`bbox_aspect_ratio`, `vertical_extent_norm`, `is_degenerate`. All 6000 rows are in
`data/features/segmorph_features.csv` (built by `scripts/build_segmorph_features.py`, 6.57s
total, ~1.10ms/image, 0 errors), prefixed `morph_*`, alongside 3 Week-3 `seg_*` columns
(`seg_otsu_threshold`, `seg_dark_area_frac`, `seg_is_degenerate`).

Open vs. closed comparison (Cohen's d = standardized mean difference; full 6000 images, 3000
per class):

| feature | closed mean | open mean | Cohen's d |
|---|---|---|---|
| **morph_extent** | 0.2365 | 0.3592 | **0.421** |
| **morph_solidity** | 0.3360 | 0.4986 | **0.411** |
| morph_n_components | 0.468 | 0.630 | 0.308 |
| morph_centroid_row_norm | 0.1939 | 0.2675 | 0.293 |
| morph_is_degenerate | 0.551 | 0.382 | -0.344 |
| morph_bbox_aspect_ratio | 0.7192 | 0.9791 | 0.222 |
| morph_eccentricity | 0.4215 | 0.5137 | 0.206 |
| morph_centroid_col_norm | 0.2455 | 0.3143 | 0.211 |
| morph_largest_area / largest_area_ratio / filled_area / filled_area_ratio | ~78px / 1.9% | ~97px / 2.4% | 0.138 |
| morph_dark_area_ratio | 0.0195 | 0.0239 | 0.130 |
| **morph_vertical_extent_norm** | 0.1492 | 0.1501 | **0.004** |

Best single-feature threshold accuracy over all 6000 images: `morph_solidity` and
`morph_extent` both reach **~0.60**; `morph_is_degenerate` alone reaches **0.585**
(matching Week 3's naive-accuracy finding, since it is derived from the same degenerate flag,
recomputed post-cleanup).

**Honest negative result:** `morph_vertical_extent_norm` — intended as the "eyelid aperture"
proxy — shows essentially **zero** separation here (Cohen's d = 0.004), contrary to the
initial hypothesis that it would be the strongest discriminator. The likely reason: Week 3's
region isolation already restricts the mask to the pupil's stricter, darker core rather than
the full sclera-opening band, so its bounding-box height no longer tracks how open the eyelid
is — it tracks how large the isolated dark blob happens to be, which is noisy in both eye
states. This is reported as measured, not adjusted after the fact to match the expected story.

**What actually separates the classes here is shape/topology, not raw area**: `extent`
(bounding-box fill fraction) and `solidity` (area / convex-hull area) are the strongest single
features (d ≈ 0.41-0.42) — an open pupil tends to isolate as a more solid, box-filling blob,
while a closed eye's stray dark fragments (eyelash strands) are thinner and less convex.
Fragmentation-related features (`n_components`, `is_degenerate`) are the next tier (d ≈ 0.3-0.37).

## Limitations and what this means for Week 6

- No single classical feature here comes close to solving eye-state classification alone —
  every effect size is in the "small-to-medium" range (|d| ≈ 0.13-0.42), and the best single
  -threshold accuracy measured is ~0.60 against a 0.50 baseline.
- This is an honest, expected outcome for a *segmentation/morphology*-only feature set on a
  dataset whose crops are already tightly bound to the eye (limited room for area/extent
  differences to manifest). Week 6's classifier should combine several of these `morph_*`
  features together (not rely on any one), and ideally combine them with Week 5's
  transform-domain features (DFT/DCT energy, Hough-fitted iris radius) for a meaningfully
  better-than-baseline result.
- `data/features/segmorph_features.csv` is keyed on `filename` and carries `split`/`eye_state`
  through unchanged from `data/working_set.csv`, ready for a `filename`-join against the
  parallel Week-5 transform-feature table.
