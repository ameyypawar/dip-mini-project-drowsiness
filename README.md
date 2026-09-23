# Driver Drowsiness Detection using Digital Image Processing

Semester mini project — Digital Image Processing.

| Student | Roll Number |
|---|---|
| Amey Pawar | 23108B0057 |
| Sejal Andhale | 23108B0049 |

![sample](results/week1_sample_grid.png)

## Problem

Driver fatigue is a leading contributor to road accidents, yet most vehicles have no
onboard way to detect drowsiness before it causes a crash. This project classifies eye
state (open/closed) from image data using a classical DIP pipeline, then tracks eyelid
closure over time (PERCLOS) on live video to raise an alert.

## Headline results

| Metric | Value |
|---|---|
| Eye-state test accuracy | **0.8965** |
| ROC AUC | 0.9621 |
| Full pipeline throughput | 46.8 fps |
| Bandwidth at compression knee | 24.3 KB/s (~195 kbit/s) |

Splits are **subject-disjoint** (train 3720 / val 1140 / test 1140; 22/7/8 subjects), so
these numbers are not inflated by the same person appearing in train and test. The test
split was evaluated once, at the end.

## DIP topic coverage

| Topic | Module | Measured result |
|---|---|---|
| Image enhancement | `src/preprocess.py`, `src/enhancement.py` | CLAHE 2.0 / tile (4,4), gamma 1.5, glare inpainting; 64×64 canonical size |
| Spatial filtering | `src/spatial_filtering.py` | median 3×3 best on salt-and-pepper (50.15 dB); mean 5×5 best on Gaussian |
| Segmentation | `src/segment.py` | Otsu chosen over adaptive (5.82 vs 28.07 components) |
| Morphological ops | `src/morphology.py` | opening 3×3 → closing 5×5 → hole-fill; 14 shape features |
| Transforms | `src/transforms.py` | DFT/DCT; Hough iris detection 0.582 open vs 0.377 closed |
| Compression | `src/compression.py` | JPEG Q=60 knee: 0.8930 accuracy at 0.678 bpp |

## Repository layout

```
submissions/  every graded PDF — all task reports and the poster
src/          pipeline modules (preprocess, enhancement, spatial_filtering,
              segment, morphology, transforms, features, classify,
              detect, ear, perclos, compression)
scripts/      dataset build, feature build, demo app, PDF renderers
notebooks/    01-09, executed with outputs
docs/         per-week write-ups and the HTML sources for each report
data/samples/ 40 curated MRL images + provenance manifest
models/       trained eye-state classifier (26.8 MB)
assets/       Haar cascades + MediaPipe face landmarker
results/      all figures
```

## Setup

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
bash scripts/extract_dataset.sh                 # optional: full MRL dataset
```

Run the live demo on a video file or webcam:

```bash
./.venv/bin/python scripts/run_demo.py --source video --video data/synthetic/sequence.mp4
./.venv/bin/python scripts/run_demo.py --source webcam
```

Interactive enhancement menu:

```bash
./.venv/bin/python scripts/enhance_app.py
```

## Weekly reports

| Week | Topic | Report |
|---|---|---|
| 1 | Proposal, dataset | [proposal](submissions/Task1_Project_Proposal.pdf) |
| 2 | Enhancement | [week2](docs/week2_enhancement.md) |
| 3 | Segmentation | [week3](docs/week3_segmentation.md) |
| 4 | Morphology | [week4](docs/week4_morphology.md) |
| 5 | Transforms | [week5](docs/week5_transforms.md) |
| 6 | Classification | [week6](docs/week6_classification.md) |
| 7 | Live inference, PERCLOS | [week7](docs/week7_live_inference.md) |
| 8 | Compression | [week8](docs/week8_compression.md) |

## Submissions

Every graded PDF lives in **[`submissions/`](submissions/)**:

| # | Deliverable | File |
|---|---|---|
| Task 1 | Project proposal | [Task1_Project_Proposal.pdf](submissions/Task1_Project_Proposal.pdf) |
| Task 2 | Image fundamentals | [Task2_Image_Fundamentals.pdf](submissions/Task2_Image_Fundamentals.pdf) |
| Task 3 | Image enhancement | [Task3_Image_Enhancement.pdf](submissions/Task3_Image_Enhancement.pdf) |
| Task 4 | Spatial filtering | [Task4_Spatial_Filtering.pdf](submissions/Task4_Spatial_Filtering.pdf) |
| Task 5 | Sharpening & integration | [Task5_Sharpening_Filters.pdf](submissions/Task5_Sharpening_Filters.pdf) |
| Task 6 | Segmentation | [Task6_Segmentation.pdf](submissions/Task6_Segmentation.pdf) |
| Task 7 | Region growing | [Task7_Region_Growing.pdf](submissions/Task7_Region_Growing.pdf) |
| — | Project poster | [Project_Poster.pdf](submissions/Project_Poster.pdf) |

## Findings worth noting

- **Texture beat shape.** Feature ablation showed HOG+LBP reaching 0.884 validation
  accuracy; adding all segmentation, morphology and transform features changed it by
  0.0000. They contributed real above-chance signal alone (0.630 and 0.660) but were
  redundant with texture. HOG and LBP are themselves classical DIP descriptors, so the
  final model uses no learned feature extractor.
- **Image-quality metrics mislead.** At JPEG Q15, SSIM reads 0.91 ("good") while
  classification accuracy has already dropped 8.4 points. Optimising a vision pipeline
  for PSNR/SSIM optimises the wrong thing.
- **Denoising rescues segmentation.** Salt-and-pepper noise drove Otsu mask IoU to 0.000
  on a poorly-lit image; a median 3×3 filter beforehand restored it to 0.90–0.98.

## Limitations

- PERCLOS and EAR/MAR are validated only on a **synthetic** sequence built from MRL
  stills. No real driving or webcam footage has been tested; the `--source webcam` path
  is written but unexercised.
- Alert timing (0.7109 agreement) is materially weaker than per-frame accuracy (0.9031):
  ~10% per-frame noise accumulated over the PERCLOS window causes false alarms.
- The adaptive EAR threshold's benefit over a fixed one is unproven without multi-subject
  footage.
- Dataset is MRL infrared eye crops only — no full-face driving data.

## Dataset & licence

MRL Eye Dataset (VSB — Technical University of Ostrava), 84,898 images / 37 subjects.
A 40-image subset is redistributed here for coursework with attribution; see
[data/samples/README.md](data/samples/README.md). Code is MIT licensed; the licence
covers code only.
