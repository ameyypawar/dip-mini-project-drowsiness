# Week 8 — Image Compression

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

Full evidence (quality ladder, PSNR/SSIM curves, the rate-accuracy curve,
manual-vs-library DCT comparison, 8x8 coefficient heatmap, energy-compaction
figures) is in `notebooks/08_compression.ipynb`, executed top-to-bottom with
stored outputs. This document is the condensed report. All code lives in
`src/compression.py` (imported per notebook cell, never reimplemented
inline); the block-wise DCT quantization core reuses `src.transforms.dct2`,
and the coefficient-truncation study reuses `src.transforms.dct2` /
`dct_energy_compaction` directly (Week 5), not reimplemented.

## The question

An in-vehicle camera transmitting eye crops to a fleet server has limited
bandwidth. How far can a frame be JPEG-compressed before the Week-6
eye-state classifier (`src.classify.predict_eye_state`, RBF-SVM on
HOG+LBP, **test accuracy 0.8965, AUC 0.9621**) degrades? The sweep below
runs on the same `test` split (n=1,140, subject-disjoint) Week 6 reports
0.8965 on, so the numbers are directly comparable.

Compression operates on the **raw, native-resolution MRL eye crop**
(`source_relpath`, e.g. 73x73, 86x86, 100x100px depending on subject),
not the 64x64 `preprocess_pipeline` output — `predict_eye_state` applies
`preprocess_pipeline` internally, so the measured pipeline is exactly
capture → JPEG-compress → transmit → decode → resize/enhance → classify,
matching what a real camera would do. See "Honest limitations" below for
why this still isn't identical to full-frame compression.

## Methods

Three independent codecs, in `src/compression.py`:

1. **Library JPEG** (`cv2.imencode`/`imdecode`) — the real codec, used for
   the headline rate-accuracy sweep.
2. **Manual block-wise 8x8 DCT + quantization** (`manual_dct_reconstruct`)
   — level-shift → per-8x8-block `src.transforms.dct2` → divide-and-round
   by a quality-scaled standard luminance quantization table → dequantize
   → per-block inverse DCT → level-shift back. This is JPEG's actual lossy
   core, built from scratch rather than treated as a black box. It has no
   entropy-coding stage (no zig-zag reorder, no run-length/Huffman coding),
   so its "rate" (`manual_bit_estimate`) is a **zeroth-order Shannon-entropy
   estimate** over the quantized coefficient stream — an approximation,
   never conflated with library JPEG's measured byte count.
3. **Full-image top-k DCT coefficient truncation** (`reconstruct_from_topk`)
   — reuses `src.transforms.dct2` / `dct_energy_compaction` (Week 5)
   unmodified: keep only the top-left k x k block of one whole-image 2D
   DCT, zero everything else, inverse DCT.

Quality levels swept: `QUALITY_LEVELS = (5, 10, 15, 20, 25, 30, 40, 50, 60,
70, 80, 90, 95)` — 13 levels, dense at the low end where accuracy actually
moves. Image-quality metrics are `skimage.metrics.peak_signal_noise_ratio`
and `structural_similarity` (`data_range=255`) against the uncompressed
original.

### Quantization table used

Standard JPEG (ITU-T T.81 Annex K) luminance table, scaled per quality by
the standard libjpeg/IJG formula (`scale_quant_table`): `scale =
5000/quality` if `quality < 50` else `200 − 2·quality`; table entries
`clip(floor((base·scale + 50)/100), 1, 255)`. Quality 50 reproduces the
base table unchanged.

```
16  11  10  16  24  40  51  61
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62
18  22  37  56  68 109 103  77
24  35  55  64  81 104 113  92
49  64  78  87 103 121 120 101
72  92  95  98 112 100 103  99
```

## Headline result: rate-accuracy sweep (test split, n=1,140)

| quality | accuracy | mean bpp | mean KB/frame | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| raw (uncompressed) | **0.8965** | — | — | ∞ | 1.0000 |
| 5  | 0.6149 | 0.419 | 0.471 | 31.24 | 0.7974 |
| 10 | 0.7667 | 0.443 | 0.502 | 35.52 | 0.8727 |
| 15 | 0.8123 | 0.468 | 0.533 | 37.77 | 0.9102 |
| 20 | 0.8456 | 0.491 | 0.563 | 39.25 | 0.9301 |
| 25 | 0.8632 | 0.514 | 0.592 | 40.29 | 0.9416 |
| 30 | 0.8632 | 0.535 | 0.621 | 41.06 | 0.9492 |
| 40 | 0.8798 | 0.577 | 0.676 | 42.17 | 0.9581 |
| 50 | 0.8842 | 0.623 | 0.739 | 43.03 | 0.9642 |
| **60** | **0.8930** | **0.678** | **0.811** | **43.76** | **0.9689** |
| 70 | 0.8956 | 0.771 | 0.934 | 44.75 | 0.9744 |
| 80 | 0.9009 | 0.930 | 1.142 | 46.02 | 0.9801 |
| 90 | 0.9088 | 1.325 | 1.651 | 48.17 | 0.9874 |
| 95 | 0.9035 | 1.869 | 2.350 | 50.21 | 0.9920 |

### Knee point

Defined as the **lowest-bitrate** quality level whose accuracy stays
within 1% absolute of the 0.8965 uncompressed baseline (`find_knee_point`,
scanning ascending quality — quality 50 already misses by 1.23 points,
quality 60 is the first to clear the tolerance at 0.35 points below
baseline):

| quality | bpp | KB/frame | accuracy | PSNR | SSIM |
|---|---|---|---|---|---|
| **60** | **0.678** | **0.811** | **0.8930** | **43.76 dB** | **0.9689** |

**Bandwidth at 30 fps: 0.811 KB/frame × 30 fps = 24.3 KB/s (≈ 194.6
kbit/s).** That is the concrete engineering payoff of this week: a
fleet uplink provisioned for ~24 KB/s per driver-facing eye-state
stream keeps classification within 1 point of the uncompressed baseline.

### PSNR/SSIM vs accuracy — where they agree, and where they don't

At the knee point itself, the three views broadly agree: PSNR 43.76 dB
and SSIM 0.9689 both read as "high quality," and accuracy is within
tolerance. **They diverge sharply earlier in the sweep, though.** At
quality 15, SSIM is already 0.9102 — by the common rule-of-thumb
(SSIM > 0.9 ≈ "good") that reads as acceptable image quality — yet
accuracy has already fallen to 0.8123, **8.4 points below baseline**,
well outside any reasonable tolerance. A system tuned purely by SSIM/PSNR
targets, without ever checking the downstream task, could ship a bitrate
the classifier cannot actually tolerate. The energy-compaction study
(below) makes the same point even more starkly.

**Surprising finding:** accuracy at quality 70–90 (0.8956, 0.9009, 0.9088)
is **slightly higher than the uncompressed baseline** (0.8965), falling
back to 0.9035 at quality 95. This is not a single-point fluke — it is a
small, monotonic-looking bump across four adjacent quality settings. A
plausible explanation: mild JPEG quantization acts as a weak low-pass
filter on sensor noise that HOG/LBP features are not fully robust to, at
a bitrate still high enough to preserve all eye-state-discriminative
structure. The effect is modest (~1–1.5 points) and close to the
sampling-noise floor of a 1,140-image test split (binomial standard error
≈0.9 points at this accuracy), so it should be read as "compression does
not hurt, and may marginally help, in a narrow high-quality band" rather
than "compression reliably improves accuracy."

## Manual DCT + quantization vs library JPEG (matched quality, n=150 sample)

| Q | library PSNR | manual PSNR | library SSIM | manual SSIM | library bpp | manual bpp (entropy estimate) |
|---|---|---|---|---|---|---|
| 10 | 35.56 | 35.55 | 0.8744 | 0.8745 | 0.459 | 0.208 |
| 30 | 41.18 | 40.96 | 0.9507 | 0.9507 | 0.551 | 0.387 |
| 50 | 43.18 | 42.85 | 0.9655 | 0.9653 | 0.637 | 0.522 |
| 70 | 44.88 | 44.35 | 0.9751 | 0.9751 | 0.781 | 0.720 |
| 90 | 48.29 | 47.18 | 0.9877 | 0.9876 | 1.332 | 1.387 |

The manual from-scratch codec tracks library JPEG's PSNR within ~1.1 dB
and SSIM to within 0.0002–0.001 at every matched quality — confirming the
quantization-table scaling and block DCT/IDCT round-trip are implemented
correctly; this **is** what JPEG's lossy core does. The two only diverge
on rate: the manual bpp is a zeroth-order entropy estimate with no
real entropy coder behind it, and it under-shoots real JPEG at low/mid
quality (no exploitation of the long zero-runs a real Huffman/RLE stage
would compress well) and slightly over-shoots at quality 90 (few zeros
left to exploit, so the estimate and the real coder converge and then
cross). Treat the manual bpp column as illustrative, not measured.

## Energy compaction / top-k reconstruction (64x64 preprocessed crop, n=150 mean)

Full-image DCT, top-left k x k coefficients kept (`src.transforms.dct2`
/ `dct_energy_compaction`, Week 5, reused unmodified):

| k | n coefficients kept (of 4096) | energy fraction | PSNR (dB) | SSIM |
|---|---|---|---|---|
| 1  | 1    (0.0%) | 0.9758 | 20.69 | 0.4453 |
| 2  | 4    (0.1%) | 0.9791 | 21.40 | 0.4519 |
| 4  | 16   (0.4%) | 0.9865 | 23.38 | 0.5276 |
| 8  | 64   (1.6%) | 0.9968 | 29.58 | 0.7802 |
| 16 | 256  (6.2%) | 0.9993 | 35.68 | 0.9227 |
| 32 | 1024 (25.0%) | 0.9998 | 41.00 | 0.9740 |
| 64 | 4096 (100%) | 1.0000 | 51.17 | 0.9991 |

**k=1 (DC coefficient only) already retains 97.6% of total DCT energy,
yet SSIM is only 0.45** — a near-unusable, flatly-shaded reconstruction.
This is the sharpest illustration in the whole study of why an
energy/PSNR-style metric can be misleading on its own: almost all image
*energy* sits in the DC term, but almost all the *structure* a perceptual
metric (or a classifier) needs lives in the discarded high-frequency
coefficients. Energy fraction crosses 0.99 only at k=16 (256/4096
coefficients, 6.2%), by which point SSIM has recovered to 0.92 — a much
more useful threshold for "is this reconstruction usable" than the raw
energy number.

## Recommended operating point

**JPEG quality ≈ 60** — 0.678 bpp, 0.811 KB/frame, 24.3 KB/s at 30 fps,
accuracy 0.8930 (0.35 points below the 0.8965 baseline, PSNR 43.76 dB,
SSIM 0.9689). This is the lowest-bitrate point in the sweep that stays
within a 1%-absolute accuracy tolerance of the uncompressed baseline.
Quality 90 gives the single highest measured accuracy (0.9088) at 1.65
KB/frame (49.5 KB/s @ 30fps) if bandwidth is not the binding constraint;
quality 60 is the recommendation when bandwidth is the binding constraint,
which is the premise of this week's question.

## Honest limitations

- **Eye-crop compression ≠ full-frame compression.** This study compresses
  the already-tightly-cropped eye ROI, not the full driver-facing camera
  frame a real system would actually capture and transmit. A real
  pipeline is capture full frame → compress full frame → transmit →
  decode → detect/crop eye region → classify. Full-frame JPEG spends bits
  on background content the eye-state classifier never sees, and the 8x8
  block grid falls differently relative to eye-region boundaries when
  cropping happens *after* decode rather than before encode. The
  qualitative shape of the rate-accuracy curve (graceful degradation, then
  a knee, then collapse) should transfer to the full-frame case, but the
  specific knee-point quality/bitrate numbers reported here are an
  eye-crop-specific estimate, not a full-frame bandwidth budget.
- **Manual-DCT bitrate is an estimate**, not a measured size — see the
  manual-vs-library table above.
- **Single classifier.** The rate-accuracy sweep is measured against one
  model (Week 6's RBF-SVM/HOG+LBP bundle, `models/eye_state_classifier.joblib`).
  A different downstream model (e.g. a CNN operating on raw pixels rather
  than HOG/LBP) could have a different, likely different-shaped,
  compression tolerance curve.
- **Small test split for the "accuracy > baseline" anomaly.** n=1,140 puts
  a binomial standard error of roughly 0.9 points around a 0.90 accuracy
  reading, so the quality-70–90 "improvement" over baseline, while
  consistent across four adjacent quality levels (which argues against
  pure noise), should not be treated as a large or fully certain effect.

## What this module produces

`src/compression.py` — `jpeg_encode_decode`, `jpeg_quality_sweep`,
`scale_quant_table`, `manual_dct_reconstruct`, `manual_bit_estimate`,
`reconstruct_from_topk`, `topk_reconstruction_sweep`, `load_test_split`,
`rate_accuracy_sweep`, `find_knee_point`, `bandwidth_kb_per_s`. Run
`python -m src.compression` for the full rate-accuracy sweep (test split
JPEG-quality sweep) and the knee-point/bandwidth summary printed to
stdout — takes ~140s.

`notebooks/08_compression.ipynb` — full visual/quantitative evidence:
quality ladder, PSNR/SSIM-vs-quality curves, the rate-accuracy curve
(headline figure), manual-vs-library DCT comparison, 8x8 DCT coefficient
heatmap + quantization table, energy-compaction ladder and curves, written
conclusion cell.

`results/week8_quality_ladder.png`,
`results/week8_psnr_ssim_vs_quality.png`,
`results/week8_rate_accuracy_curve.png`,
`results/week8_manual_vs_library_dct.png`,
`results/week8_dct_heatmap_qtable.png`,
`results/week8_energy_compaction_ladder.png`,
`results/week8_energy_compaction.png`.
