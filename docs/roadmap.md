# Project Roadmap — Driver Drowsiness Detection

All eight weeks complete. Maps the five mandated DIP topics onto weekly deliverables.

| Week | Topic | Status | Headline result |
|---|---|---|---|
| 1 | Survey, dataset, proposal | ✅ | 40 curated MRL images, subject-stratified, 2-page proposal |
| 2 | Image enhancement | ✅ | 64×64 canonical; CLAHE 2.0 tile (4,4); gamma 1.5; glare inpainting |
| 3 | Segmentation | ✅ | Otsu over adaptive — 5.82 vs 28.07 mean components |
| 4 | Morphological operations | ✅ | opening 3×3 → closing 5×5 → hole-fill; 14 features, best d=0.42 |
| 5 | Transforms | ✅ | DFT/DCT; Hough detection gap 0.205 (0.582 open vs 0.377 closed) |
| 6 | Eye-state classification | ✅ | RBF-SVM on HOG+LBP — test accuracy 0.8965, AUC 0.9621 |
| 7 | Live inference, EAR/MAR, PERCLOS | ✅ | 46.8 fps; per-frame 0.9031, alert-timing 0.7109 |
| 8 | Compression + integration | ✅ | JPEG Q=60 knee: 0.8930 accuracy at 24.3 KB/s @30fps |

## Lab task submissions

| Task | Topic | Report |
|---|---|---|
| 1 | Proposal | `docs/proposal/DIP_Proposal_Drowsiness_Detection.pdf` |
| 3 | Image enhancement (spatial domain) | `docs/task3_report.pdf` |
| 4 | Spatial filtering — smoothing & sharpening | `docs/task4_report.pdf` |

## Topic-to-week cross-check

| DIP topic | Week(s) |
|---|---|
| Image enhancement | 2, Task 3 |
| Spatial filtering | Task 4 |
| Segmentation | 3 |
| Morphological operations | 4 |
| Transforms | 5 |
| Image compression | 8 |

## Remaining work

- Record real webcam footage to validate EAR/MAR and PERCLOS on actual faces; the
  synthetic sequence exercises the pipeline but is not real driving data.
- Reduce PERCLOS false alarms — per-frame classifier noise accumulating over the
  10 s window is the system's main weakness.
