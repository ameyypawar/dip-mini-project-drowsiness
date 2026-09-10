# Task 3 -- Image Enhancement in the Spatial Domain

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057).

Full graded write-up: `docs/task3_report.pdf` (institutional lab-report template,
built from `docs/task3_report.html` via `scripts/make_task3_pdf.sh`). This
document is a plain-markdown companion covering the same material. All
enhancement code lives in `src/enhancement.py` (new module, does not modify
`src/preprocess.py` / `src/segment.py` / `src/spatial_filtering.py`). Figures
are generated deterministically (seed=0 throughout) by
`scripts/make_task3_figures.py` into `results/task3_*.png` and
`results/task3_metrics.csv`.

## Aim

To implement and evaluate classical spatial-domain image-enhancement
techniques -- intensity transformations (negative, log, gamma, contrast
stretching), histogram analysis and equalization, and image arithmetic
(addition, subtraction, averaging) -- on grayscale near-infrared eye-crop
images from the driver-drowsiness dataset, to quantitatively compare their
effect on brightness, contrast and noise, and to identify the most suitable
technique for images captured under different lighting/glare conditions.

## Images used

Three images from `data/samples/` chosen to span genuinely different
acquisition conditions (per `data/samples/manifest.csv` fields
`glasses`/`lighting`/`reflections`), preferring the physically larger crop
within each condition bucket:

| Label | File | Condition | Native size |
|---|---|---|---|
| `good_lighting` | `alert/s0031_00366_1_0_1_0_1_02.png` | open eye, no glasses, good lighting, no reflection | 128x128 |
| `poor_lighting` | `alert/s0014_08297_0_0_1_1_0_02.png` | open eye, no glasses, poor lighting | 130x130 |
| `glasses_reflection` | `alert/s0035_00133_0_1_1_2_1_02.png` | open eye, eyeglasses + strong IR reflection | 144x144 |

Figures upscale each crop with `cv2.INTER_NEAREST` to a minimum 300px side so
pixel detail survives.

## A. Intensity transformations

`negative`, `log_transform`, `gamma_correction`, `contrast_stretch` -- all
parameterised (gamma, log constant `c`, contrast-stretch percentiles/range).
See `results/task3_transforms_*.png` (original vs. each transform) and
`results/task3_histograms_*.png` (histogram before/after each transform).

## B. Histogram analysis

`compute_histogram` (256-bin) and `histogram_stats` (mean/std/min/max), shown
alongside every image in the figures above.

## C. Histogram equalization

`equalize_histogram_cv` (OpenCV) and `equalize_histogram_manual` (own
CDF-mapping implementation). The own implementation matches
`cv2.equalizeHist` **pixel-for-pixel** (max absolute difference 0) on all 3
test images. Contrast (std-dev) roughly **3-4x's** on all three images:

| Image | Orig. mean | Eq. mean | Orig. std | Eq. std |
|---|---|---|---|---|
| good_lighting | 101.17 | 130.12 | 18.44 | 73.90 |
| poor_lighting | 83.74 | 128.39 | 41.75 | 73.44 |
| glasses_reflection | 103.14 | 132.67 | 22.11 | 74.28 |

See `results/task3_histeq_*.png`.

## D. Image arithmetic

- **Averaging** (`average_images` + `make_noisy_copies`, seeded): noise
  reduction across repeated acquisitions. Noise proxy: single noisy frame
  19.40 -> avg of 5 8.71 -> avg of 20 4.43, tracking the theoretical
  `1/sqrt(N)` prediction (8.68, 4.34). `results/task3_arithmetic_averaging.png`.
- **Subtraction** (`subtract_images`): change/motion detection, demonstrated
  on a genuine same-subject open-vs-closed pair (subject `s0031`) -- mean
  diff 21.13, peak 100, concentrated on the eyelid region that moved.
  `results/task3_arithmetic_subtraction.png`.
- **Addition** (`add_images`, `add_constant`): brightness boost / overlay.
  Blend (0.5\*orig + 0.5\*bright) lifts mean 101.17->129.94; constant +40
  lifts it to 141.17. `results/task3_arithmetic_addition.png`.

## E. Comparison across the 3 images

Score = contrast-gain / noise-gain (both relative to that image's own
original), rejecting any technique that pushes brightness outside [30,225].
Full numbers: `results/task3_metrics.csv`; chart: `results/task3_comparison_bars.png`.

| Image | Winning technique | Score | Why |
|---|---|---|---|
| good_lighting | **hist_eq** | 1.236 | Lowest native contrast (std 18.4) of the 3 -- most headroom for equalization's gain to outpace its noise cost. |
| poor_lighting | **contrast_stretch** | 1.130 | Native histogram already wide (std 41.8); percentile clipping captures most of the achievable gain for less added noise than full equalization needs. |
| glasses_reflection | **hist_eq** | 1.206 | Glare saturates part of the histogram, leaving equalization a large region to redistribute; extracts the most contrast per unit noise of any technique. |

`negative` always scores exactly 1.000 on every image -- negating a uint8
image leaves both std-dev (contrast) and the median-filter residual (noise
proxy) numerically unchanged.

**No single technique is universally best** -- the right choice depends on
how much native contrast headroom the specific image already has.

## Observations & Applications

- **Negative**: polarity flip only; contrast/noise provably unchanged.
- **Log transform**: strong dark-region brightening, but *lowers* contrast
  and noise here (compresses the bright end where pupil/iris edge contrast
  lives) -- least useful of the 5 techniques on this dataset.
- **Gamma correction**: the only continuously-tunable single-knob option for
  a specific brightness target; doesn't win outright but is the most
  controllable.
- **Contrast stretching**: robust, tunable, wins on `poor_lighting`.
- **Histogram equalization**: strongest, most consistent contrast booster
  (3-4x), largest noise cost; wins on 2 of 3 images here because these IR
  crops start with unusually low native contrast.
- **Addition**: brightness boost / overlay compositing.
- **Subtraction**: change / motion (blink) detection between frames.
- **Averaging**: noise reduction across repeated acquisitions, confirmed to
  track `1/sqrt(N)`.

## Challenges Faced

1. **Log-transform scaling on already-dark IR crops** -- resolved by
   auto-deriving `c = 255/log(1+max(r))` per image instead of a fixed constant.
2. **Histogram equalization amplifying sensor noise and IR glare** -- noise
   proxy roughly triples after equalization (e.g. glasses_reflection
   1.70->4.74).
3. **Small image size (67-172px) limits contrast-stretch headroom** -- fewer
   samples per percentile bin than a full-resolution photo.
4. **No true multi-frame burst of the same eye in this dataset** -- the
   averaging demo needed seeded synthetic noisy copies; the subtraction demo
   used a genuine same-subject open/closed pair instead (subject `s0031`).
5. **Repeating letterhead/border/page-number in a Chrome-headless-only PDF
   pipeline** (no pandoc/LaTeX available) -- `position:fixed` in Chrome's
   print-to-pdf is offset by the `@page` margin box, and there is no native
   per-page auto-incrementing counter. Resolved with a `<table>`
   `<thead>`/`<tfoot>` (natively repeats every printed page) plus a script
   that inserts sequentially-numbered markers at safe content seams
   (paragraph/list/figure/table boundaries), calibrated against the actual
   rendered PDF.

## Conclusion

All eight sub-tasks were implemented in `src/enhancement.py` and exercised on
three images spanning good lighting, poor lighting, and eyeglasses+reflection
conditions. Histogram equalization won the measured contrast-gain-per-noise-
gain comparison on 2 of 3 images; contrast stretching won on the third, where
the native histogram already had a wide spread. The own histogram-
equalization implementation matches `cv2.equalizeHist` exactly. Image
averaging's noise reduction matched the theoretical `1/sqrt(N)` prediction,
and image subtraction on a genuine same-subject open/closed pair cleanly
isolated the eyelid region that changed.

## Integration

`scripts/enhance_app.py` -- menu-driven CLI (load image -> pick technique ->
view stats -> save) and non-interactive flags (`--image PATH --technique
gamma --gamma 1.5 --out PATH`), verified to run fully non-interactively for
every technique family and to refuse to block on `input()` when no
`--technique` flag is given and stdin is not a TTY.
