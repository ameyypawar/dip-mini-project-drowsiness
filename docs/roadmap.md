# Project Roadmap — Driver Drowsiness Detection

Maps the five mandated DIP topics onto weekly deliverables. Module names below
(`src/preprocess.py`, `src/ear.py`, `src/detect.py`, `src/perclos.py`) are **stubs planned for
future weeks — not yet created.**

| Week | DIP topic(s) in focus | Deliverable | Notes |
|---|---|---|---|
| **1** (this week) | Survey / setup | Repo scaffold, literature survey, DIP topic mapping, curated 40-image MRL sample + grid, dataset proposal, PDF | No modeling yet |
| **2** | Image enhancement | Image-size normalization + CLAHE/gamma enhancement experiments in a notebook | Input crops are variable-size — a 200-image sample of MRL shows **73 distinct (width, height) dimensions** — must resize/pad to a canonical shape before any batch processing or CNN input. This is a prerequisite task, done before enhancement tuning. |
| **3** | Segmentation | Adaptive + Otsu thresholding to isolate sclera/iris/pupil on enhanced crops | Feeds eye-state features |
| **4** | Morphological operations | Opening/closing + hole-filling to clean segmentation masks; area/shape features extracted | Prepares features for classifier |
| **5** | Transforms | DFT/DCT blur-rejection filter; Hough circle iris fitting | Also record short **self-recorded video clip** (webcam, few minutes, varied blink/closure patterns) needed for the Week-7 PERCLOS temporal demo — PERCLOS requires a real time-series, which MRL (static crops) cannot provide |
| **6** | Eye-state classification | Train/evaluate classifier on MRL (train/val split); stub `src/preprocess.py`, planned | Classifier trained on MRL only — inference-time face localization is separate |
| **7** | Live inference: EAR/MAR + PERCLOS | Face/eye localization (Haar cascade / MediaPipe) on webcam or the self-recorded clip; stub `src/ear.py`, `src/detect.py`, `src/perclos.py` planned | EAR/MAR computed on live frames only, never on MRL; PERCLOS computed over the recorded clip's frame sequence |
| **8** | Compression + integration | JPEG/DCT quality-vs-accuracy sweep; final integration + demo + report | Wraps all 5 DIP topics into final report |

## Topic-to-week cross-check

| DIP topic | Week(s) |
|---|---|
| Image enhancement | 2 |
| Segmentation | 3 |
| Morphological operations | 4 |
| Transforms | 5 |
| Image compression | 8 |

## Dependencies / risks

- Week 2 normalization choice (resize target size, aspect-ratio handling) affects every
  downstream week — must be decided and documented before Week 3 begins.
- PERCLOS demo depends on the Week-5 self-recorded clip existing and being of sufficient
  length/frame-rate; schedule the recording early in Week 5, not at the end.
- Live-inference weeks (7) depend on a working face/eye localizer (Haar cascade fallback if
  MediaPipe integration slips).
