# Week 2 — Image Enhancement

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

Full evidence (dimension histograms, resize-vs-letterbox grids, CLAHE/gamma
sweeps with pixel-intensity histograms, glare-suppression demo) is in
`notebooks/02_image_enhancement.ipynb`, executed top-to-bottom with stored
outputs. This document is the condensed report: what was measured, what was
chosen, and why.

**Correction note:** an earlier draft of this document picked 128x128 and
stated "128 covers 90.4% of images as a downsample" — that had the
direction backwards. 90.4% of images are *smaller* than 128 and would need
to be **enlarged** (upsampled) to reach it, not shrunk. The corrected
analysis below picks **64x64** instead, and the direction is verified
explicitly (not eyeballed) in the notebook.

## Canonical size — **64x64** (chosen, locked in for weeks 3-8)

Measured with `dimension_stats()` (`src/preprocess.py`) over the **full
extracted dataset**: 84,898 images in `data/raw/mrlEyes_2018_01/`, all 37
subject directories.

| Stat | Width | Height | Aspect ratio (w/h) |
|---|---|---|---|
| min | 52 px | 52 px | 1.0 |
| max | 299 px | 299 px | 1.0 |
| mean | 93.4 px | 93.4 px | 1.0 |
| median | 87 px | 87 px | 1.0 |
| distinct sizes | 236 (w,h) pairs across the full dataset | | |

(An independently drawn 8,000-image random sample gives ~161 distinct
sizes — consistent with the full-dataset figure of 236, since a smaller
sample sees fewer distinct sizes. The 236 figure describes the full
84,898-image population; use it, not the sample figure, as the
authoritative count.)

Every MRL eye crop is a perfect square (width == height, aspect ratio
exactly 1.0, no exceptions across all 84,898 images) — so there is zero
aspect-ratio-distortion risk from any resize strategy on this dataset.

**Direction, verified explicitly** (full dataset, width array):

| target | shrink (w > target) | exact (w == target) | enlarge (w < target) |
|---|---|---|---|
| **64** | **94.3%** | 1.4% | **4.4%** |
| 96 | 34.8% | 1.3% | 63.8% |
| 128 | 9.6% | 0.5% | 89.9% |

The median crop (87px) and mean (93.4px) are both larger than 64, so a
small canonical size means most images are **shrunk** (information-reducing,
`INTER_AREA`), not enlarged. Only 4.4% need `INTER_CUBIC` upsampling at
target=64. A target of 128 inverts this: 89.9% of images would need to be
*enlarged*, inventing pixel detail that was never captured by the sensor —
and weeks 3-5's Otsu thresholding, morphological opening/closing, and Hough
circle fitting would then be run largely on interpolation artifacts rather
than real measurements. That is the defensible-direction argument for
picking the **smaller** canonical size. 64x64 is also a fast, standard
working size for the planned Week-6 classical-feature (HOG/LBP) +
scikit-learn SVM/RF classifier — keeps feature extraction and inference
cheap on CPU.

## Resize mode — **plain `resize`** (not letterbox)

Both `normalize_size(img, 64, mode="resize")` and
`normalize_size(img, 64, mode="letterbox")` are implemented in
`src/preprocess.py`. Compared directly on smallest/median/largest curated
samples (notebook Section 2): since aspect ratio is exactly 1.0 for every
MRL image, the two modes produce **identical output** (verified with
`np.array_equal`) — letterbox degenerates to plain resize with zero padding
when the input is already square. `resize` is chosen for simplicity.
`letterbox` is kept in `normalize_size` for reuse on non-square inputs
(e.g. a Week-7 webcam eye ROI before its own square crop step).

## CLAHE clip limit — **2.0**, tile grid — **(4,4)** (tile grid corrected for 64x64)

Swept clip_limit in {1.0, 2.0, 3.0, 4.0, 6.0, 8.0} on the curated
bad-lighting subset (`lighting=0`, n=20), all resized to 64x64, measuring
mean/std of pixel intensity (std as a contrast proxy) after gamma=1.5 +
CLAHE.

**Tile-grid pitfall found and fixed:** the original (128x128) analysis used
`tile_grid=(8,8)`, i.e. 16x16px tiles. Carrying that same `(8,8)` grid over
to the new 64x64 size gives only **8x8px tiles** — too few pixels per tile
for CLAHE's clip-and-redistribute mechanism to respond to `clip_limit` at
all:

| clip_limit | tile_grid=(8,8) std (8x8px tiles — broken) | tile_grid=(4,4) std (16x16px tiles — fixed) |
|---|---|---|
| 1.0 | 23.0 | 18.5 |
| 2.0 | 23.0 | 27.2 |
| 3.0 | 23.0 | 32.0 |
| 4.0 | 23.0 | 32.6 |
| 6.0 | 23.0 | 42.2 |

With `(8,8)`, std is flat regardless of clip_limit — a degenerate,
non-functional control, confirmed visually in the notebook (identical
images at every clip value). With `(4,4)` (matching the original 16px/tile
physical tile size), std responds smoothly and monotonically, as expected.
**`clahe_tile_grid` must scale with `target_size`** — fixed at `(4,4)` for
the chosen 64x64 canonical size, not left at `(8,8)`.

Full clip sweep at the corrected `tile_grid=(4,4)` (baseline, gamma-only,
no CLAHE: std=16.5):

| clip_limit | mean | std |
|---|---|---|
| 1.0 | 133.8 | 23.3 |
| **2.0** | **147.3** | **29.6** |
| 3.0 | 154.7 | 35.0 |
| 4.0 | 151.5 | 39.1 |
| 6.0 | 147.4 | 46.4 |
| 8.0 | 142.7 | 50.7 |

(Single-image demo in the notebook figure shows the same pattern at
slightly different absolute numbers: std 18.5 -> 27.2 -> 32.0 -> 32.6 ->
42.2 for clip 1/2/3/4/6 -- the aggregate table above is the full
bad-lighting subset average.)

Contrast rises steadily to clip=3-4, then the mean itself starts
*reversing* past clip=4 while std keeps climbing — that's the clip limit
amplifying noise rather than adding real contrast (visibly grainy in the
notebook's image grid). clip_limit=2.0 (OpenCV's own conventional default)
already delivers a strong, clean contrast gain over the unenhanced
baseline (std 16.5 -> ~27-30, roughly +65-80%) without the graininess
visible past clip=4. **Chosen: clip_limit = 2.0 (confirmed unchanged from
the original decision), tile_grid = (4,4) (changed from (8,8) to match the
new target size).**

## Gamma — **1.5** (confirmed unchanged)

Swept gamma in {0.6, 0.8, 1.0, 1.3, 1.5, 1.6, 2.0} on the same bad-lighting
subset at 64x64, tracking mean intensity and saturated-pixel fraction.
Gamma is a pointwise LUT operation (`output = 255*(input/255)**(1/gamma)`),
independent of image resolution, so the numbers reproduce exactly from the
original 128x128 sweep:

| gamma | mean | saturated px |
|---|---|---|
| 1.0 (baseline) | 77.8 | 0.13% |
| 1.3 | 101.3 | 0.13% |
| **1.5** | **~114** | **~0.13%** |
| 1.6 | 120.0 | 0.13% |
| 2.0 | 139.2 | 0.14% |

The bad-lighting subset sits dark at baseline (mean 77.8/255). Gamma=1.5
lifts mean intensity to ~114, close to the mid-range, while the
saturated-pixel fraction stays flat across the whole sweep — gamma is
redistributing the existing dynamic range, not clipping it. Gamma=2.0
overshoots (mean 139) with no added benefit over 1.5-1.6, so 1.5 is chosen
as the balanced setting, unchanged from the original analysis.

## Glare suppression (confirmed unchanged at 64x64)

Method (see `suppress_glare()` docstring in `src/preprocess.py` for the
full explanation): **global intensity threshold (>=235) -> morphological
dilation (elliptical kernel) -> PDE-based inpainting** (`cv2.INPAINT_TELEA`).
Demonstrated on the curated manifest's `glasses=1 & reflections=2` subset
(8 images), resized to 64x64 — the notebook's Section 5 figure shows
original / threshold mask / dilated mask / inpainted result side by side,
plus before/after pixel-intensity histograms.

Result: max intensity drops from 255 (saturated hotspot) to a plausible
in-range value on every sample, while mean intensity is essentially
unchanged (the hotspot is a small fraction of total pixels) — confirming
the glare artifact is removed without altering overall image brightness.
Default parameters (threshold=235, dilate_px=3, inpaint_radius=5) transfer
from 128x128 to 64x64 without needing adjustment.

**Honest limitation:** the inpainted patch is still visible as a faint
lighter-gray disc in several samples rather than being invisible, because
several hotspots straddle the iris/pupil boundary and the fill partially
borrows brighter sclera/eyelid texture from just outside the mask. The
peak-saturation artifact is removed (the actual goal — an un-suppressed
255-valued hotspot would otherwise get mis-thresholded in Week 3's
segmentation or mis-fit by Week 5's Hough circle transform), but perfect
iris-texture recovery is not claimed.

## Hardest-case stress test

The 3 curated images that are simultaneously `lighting=0` (bad light) AND
`reflections=2` (high reflection) — the worst case this pipeline must
handle — were run through the full `preprocess_pipeline()` at 64x64.
Notebook Section 6 shows raw vs. enhanced side by side with per-image mean
intensity; the enhanced output is visibly brighter and higher-contrast
than the raw dark, glare-affected input in all 3 cases.

## What this locks in for weeks 3-8

`src/preprocess.py`'s `PreprocessConfig` defaults are now the canonical
Week-2 decision:

```python
PreprocessConfig(
    target_size=64,
    resize_mode="resize",
    glare_suppression=True,
    glare_threshold=235,
    glare_dilate_px=3,
    glare_inpaint_radius=5,
    gamma=1.5,
    clahe_clip_limit=2.0,
    clahe_tile_grid=(4, 4),
)
```

`preprocess_pipeline(img, cfg)` applies, in fixed order:
`normalize_size -> suppress_glare -> apply_gamma -> apply_clahe`. Every
downstream week (3: segmentation thresholds, 4: morphology on the 64x64
canonical mask, 5: DFT/DCT + Hough on a fixed-size image, 6: classifier
input shape for HOG/LBP feature extraction, 7: live webcam eye ROI resized
to the same 64x64 before inference) should call this single function
rather than re-implementing normalization or enhancement. **If a future
week changes `target_size` again, `clahe_tile_grid` must be re-derived
alongside it** — see the CLAHE section above for why a fixed tile-grid
count silently breaks at a different canonical size.
