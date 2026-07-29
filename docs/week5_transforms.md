# Week 5 — Transforms (DFT/DCT + Hough)

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

Full evidence (magnitude spectra, radial profiles, DCT heatmaps,
energy-compaction curves, Hough overlays, detection-rate table, feature
distributions) is in `notebooks/05_transforms.ipynb`, executed top-to-bottom
with stored outputs. This document is the condensed report: what was tried,
what was measured, what was chosen, and why. All code lives in
`src/transforms.py` (reused, not reimplemented per notebook cell); feature
extraction is in `src/transforms.build_feature_table`, output at
`data/features/transform_features.csv`.

## Fourier (DFT)

`np.fft.fft2` + `np.fft.fftshift` on the enhanced 64x64 crop
(`src.transforms.dft_shifted`). Display uses a log-scaled magnitude
spectrum (`log(1 + |F|)`, min-max normalized) since the DC/low-frequency
component otherwise dwarfs everything else on a linear scale.

### High-frequency energy ratio — radius = 8 px (chosen)

Fraction of spectral energy (`|F|^2`) lying outside a central disk of
`radius` px. Radius chosen by a sweep on a controlled sharp/blurred pair
sample: MRL has no blur ground truth, so pairs were built by applying a
known `cv2.GaussianBlur(ksize=7, sigma=2)` to each enhanced crop, then
scoring how well `high_freq_energy_ratio` at each radius separates sharp
from blurred (AUC via Mann-Whitney U — no sklearn dependency, computed
directly from ranks).

| radius (px) | AUC (n=300 pairs) |
|---|---|
| 4 | 0.7698 |
| 6 | 0.8164 |
| **8** | **0.8246** |
| 10 | 0.8208 |
| 12 | 0.8139 |
| 14 | 0.8025 |
| 16 | 0.7930 |
| 20 | 0.7789 |
| 24 | 0.7736 |

AUC peaks in a shallow plateau over radius 6-10; **8 is chosen** as the
middle of that plateau. Below 8, the "outside" region shrinks and the
ratio saturates toward 0 for both classes (less discriminative range).
Above ~14, AUC steadily degrades — on a 64x64 crop there is little energy
that far from center even in a sharp image, so the outside-energy sum
becomes noise-dominated rather than signal-dominated.

### Radial energy profile

8 concentric annuli (bin `i` = mean power of pixels in
`[i * step, (i+1) * step)` px from center, `step = max_radius / 8`), mean
not sum per bin so inner and outer annuli (very different pixel counts)
are comparable. Overlaid open-vs-closed curves are in the notebook
(Section 5) — both classes show the expected steep low-to-high frequency
falloff of a natural image; the curves separate most visibly in the
low/mid bins, consistent with closed eyes' relatively flatter (eyelid)
texture versus open eyes' iris/sclera/eyelash detail.

## Cosine (DCT)

`cv2.dct` on the float32 enhanced crop (`src.transforms.dct2`).
Energy-compaction feature: fraction of total DCT energy (`sum(coef^2)`)
held in the top-left k x k block (lowest-frequency coefficients),
`src.transforms.dct_energy_compaction`. `DCT_KS = (2, 4, 8, 16)` are
emitted as features — **k=8 is deliberately included because it is the
JPEG block size**, directly reusable for the Week 8 compression sweep
without re-deriving the metric.

The energy-compaction curve (notebook Section 6, all k from 1 to 64) shows
the expected classical-DCT property: a small top-left block already
captures the overwhelming majority of total image energy, confirming why
JPEG-style block quantization can discard higher-frequency coefficients
with limited perceptual loss.

## Classical baseline: Laplacian variance vs DFT-based blur metric

Head-to-head on the same synthetic sharp/blurred construction as the
radius sweep, larger sample (n=600 pairs, fresh random working-set draw):

| Metric | sharp mean ± std | blurred mean ± std | AUC (sharp vs blurred) |
|---|---|---|---|
| DFT high-freq energy ratio (radius=8) | 0.00214 ± 0.00134 | 0.00104 ± 0.00074 | **0.8141** |
| Laplacian variance (`cv2.Laplacian`, `CV_64F`, `.var()`) | 112.09 ± 79.53 | 6.03 ± 3.51 | **1.0000** |

**Result: Laplacian variance separates blurred from sharp better** —
essentially perfectly on this synthetic-blur test, versus a clearly
above-chance but visibly weaker 0.81 for the DFT ratio. Interpretation:
Laplacian variance is a genuinely *local* second-derivative measurement per
pixel, directly sensitive to edge sharpness everywhere in the image. The
DFT ratio pools energy over a coarse radial partition of the *whole*
spectrum and discards phase information entirely — real high-frequency
signal gets diluted against low-frequency energy that leaks past a hard
circular cutoff, especially at the small 64x64 resolution where the
frequency grid itself is coarse. Both metrics are kept in the feature
table: Laplacian variance as the primary blur feature, the DFT ratio as a
complementary global-spectrum summary for Week 6's classifier to weigh
however it turns out to help.

Stated plainly: on this test, **the simpler spatial-domain operator won**.
DFT/DCT remain the mandated transform-domain deliverable for this week and
the DCT energy-compaction work feeds directly into Week 8's JPEG study, but
for the specific job of "is this crop blurred," Laplacian variance is the
better metric and the report should say so rather than talk up the fancier
transform.

## Hough circle transform — iris fitting

**Central hypothesis, exploited by design:** an open eye exposes a roughly
circular iris against sclera; a closed eye shows eyelid/eyelash texture
with no circular boundary. So `HoughCircles` returning nothing on a closed
eye is the *expected, informative* outcome, not a detector failure — see
`src.transforms.detect_iris_circle` and `IrisCircle.found`.

### Parameter tuning

`cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist, param1, param2,
minRadius, maxRadius)` — `dp`/`minDist` fixed early (`dp=1.0`, `minDist=64`,
i.e. the crop height, since at most one iris is expected per 64x64 crop).
`median_blur_ksize` (pre-Hough denoise), `param1` (Canny high threshold),
`param2` (accumulator threshold), `minRadius`, `maxRadius` were grid-searched.

**Pitfall found:** an initial grid search on the curated 40-image sample
(20 open / 20 closed) picked configurations with an apparently large
open-vs-closed detection gap (e.g. open=0.80, closed=0.20 for one
candidate) — but n=20 per class is small enough that this was substantially
sampling noise. Re-evaluating the top candidates on a stratified 800-image
draw from the working set (400 open + 400 closed, disjoint from the
curated 40) showed the true gap is much smaller (roughly 0.03-0.13 across
those same candidates) — a direct illustration of why parameter selection
must be validated on more than 40 images before being trusted.

A second, finer grid search (blur ∈ {3,5,7}, param1 ∈ {40..100}, param2 ∈
{12..20}, minRadius ∈ {6..10}, maxRadius ∈ {14..22}; ~3,150 configurations,
674s wall time) was run directly on the 800-image stratified sample, scoring
each configuration by `open_rate - closed_rate`. The top results were
tightly clustered and near-tied on gap:

| gap | open rate | closed rate | blur | param1 | param2 | minRadius | maxRadius |
|---|---|---|---|---|---|---|---|
| 0.2575 | 0.5500 | 0.2925 | 5 | 100 | 12 | 7 | 14 |
| **0.2575** | **0.5925** | **0.3350** | **5** | **100** | **12** | **6** | **14** |
| 0.2425 | 0.6025 | 0.3600 | 7 | 90 | 12 | 6 | 14 |
| 0.2350 | 0.5350 | 0.3000 | 7 | 100 | 12 | 6 | 14 |
| 0.2325 | 0.6575 | 0.4250 | 7 | 80 | 12 | 6 | 14 |

**Chosen: the second row** — tied for the highest gap, but with
substantially higher absolute detection rates (0.5925/0.3350 vs
0.5500/0.2925 for the tied-gap alternative), meaning more images get a real
fitted radius instead of the missing-circle sentinel. A feature that fires
more often at the same separation margin is more useful downstream.

**Chosen `HoughConfig`** (`src/transforms.py`):

```python
HoughConfig(
    dp=1.0,
    min_dist=64,
    param1=100,
    param2=12,
    min_radius=6,
    max_radius=14,
    median_blur_ksize=5,
)
```

### Detection rate — open vs closed (headline result)

Full working set (all 6,000 images, from `data/features/transform_features.csv`):

| | n | found | rate |
|---|---|---|---|
| open (eye_state=1) | 3,000 | 1,747 | **0.5823** |
| closed (eye_state=0) | 3,000 | 1,132 | **0.3773** |
| **gap (open − closed)** | | | **0.2050** |

(The notebook's own 800-image stratified draw, Section 8, gets a
consistent 0.593 / 0.335 / gap 0.258 — same order of magnitude as both the
tuning-set number and the full-6,000 number above.)

**Read this honestly:** a 0.205 gap on the full working set is real,
usable signal — open eyes trigger a circle fit roughly 1.5x as often as
closed eyes — but it is far short of the clean, near-binary separation the
"open eye shows an iris, closed eye does not" hypothesis would predict in
an idealized image. Hough circle detection on this dataset is a
**contributing feature, not a solver**: neither rate is anywhere near 0 or
1, so a huge fraction of both classes land on the "wrong" side of a
naive found/not-found threshold. The gap is not larger mainly because
closed eyes are not textureless — eyeglass-frame arcs, glare rings from
imperfectly-suppressed IR reflections, and curved eyelid/eyelash creases
all present enough local circular curvature to trip a permissive Hough
accumulator, while genuinely open eyes are lost to partial eyelid
occlusion, low iris/sclera contrast, or off-center pupils.

This modest result is consistent with, not an outlier next to, the
parallel Week 3/4 morphology work on this same working set: their best
single segmentation/morphology feature reaches d≈0.42, a Random Forest
over all 14 morphology features alone scores 0.618 val / 0.682 test, and
their a-priori strongest-looking candidate ("eyelid aperture") was a
near-total washout at d≈0.004. No single classical feature here cleanly
solves open/closed on its own — Week 6's plan is to combine feature
families (transforms + morphology + HOG/LBP texture), not to lean on one.

### Honest limitations

- **Small, noisy detection rates.** Even after tuning, Hough circle
  detection on a 64x64 IR crop is far from a reliable per-image iris
  detector — both open and closed rates are well under 1.0. The feature is
  useful as a *statistical* signal across many images (the population-level
  open/closed gap), not as a trustworthy single-image circle fit.
- **False positives on closed eyes.** 1,132 of 3,000 closed-eye crops
  (37.7%) still return a "circle" — eyeglass frame arcs, eyelash clusters,
  and IR glare rings (particularly on `reflections=2` frames that survived
  Week 2's glare suppression only partially) all present enough local
  curvature to satisfy a permissive `param2`. These are visible in the
  notebook's overlay figure (Section 7) as "found (fp)" annotations.
- **False negatives on open eyes.** Partially occluded iris (eyelid
  overlap on a not-fully-open eye), low contrast between iris and sclera
  in some lighting conditions, and off-center pupils all suppress
  detection on genuinely open eyes — 41.8% of open-eye crops in the full
  working set get no circle at all.
- **`param2` is a precision/recall knob, not a free lunch.** Lowering it
  raises open-eye recall but raises the closed-eye false-positive rate at
  roughly the same time — the tuning above picked a point on that
  trade-off curve rather than eliminating it.
- **Found radius does not cleanly separate classes either** — the
  notebook's 800-image sample gives mean found radius 10.29px (open,
  n=237) vs 10.80px (closed, n=134), i.e. when a circle *is* found on a
  closed eye, it is not systematically smaller/different in size, further
  evidence that closed-eye detections are often genuine circular artifacts
  (frames/glare) rather than small spurious noise fits.

## What Week 6 gets

`data/features/transform_features.csv` — one row per `filename` in
`data/working_set.csv` (6,000 rows), columns `filename, split, eye_state`
followed by:

- `dft_high_freq_ratio`
- `dft_radial_bin0` .. `dft_radial_bin7` (8 bins)
- `dct_compaction_k2`, `dct_compaction_k4`, `dct_compaction_k8`, `dct_compaction_k16`
- `laplacian_variance`
- `hough_found`, `hough_cx`, `hough_cy`, `hough_radius`, `hough_n_circles`

Built by `python -m src.transforms` (`src.transforms.build_feature_table`):
**6,000 rows written in 10.5s (1.74 ms/image)** — fast enough that no
subsampling was needed for the full working set.

Join key is `filename`, matching the parallel segmentation/morphology
feature table so Week 6 can join both on that column with zero risk of
row misalignment (both tables are keyed off the same `data/working_set.csv`
row order and content).
