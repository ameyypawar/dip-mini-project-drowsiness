# Week 6 — Eye-State Classification

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

Full evidence (model-type search table, ablation bar chart, confusion
matrix, ROC/PR curves, per-subject accuracy, error cross-tabulation, worst-
failure contact sheet) is in `notebooks/06_classification.ipynb`, executed
top-to-bottom with stored outputs. This document is the condensed report.
Code lives in `src/features.py` (HOG/LBP + family-wise feature assembly)
and `src/classify.py` (model search, ablation, final training, persistence,
and the `predict_eye_state` entry point Week 7 loads).

By explicit team choice, this is a classical-features + scikit-learn
classifier (LinearSVC / RBF-SVM / RandomForest) rather than a CNN, so that
the Week 2-5 DIP pipeline (preprocessing, segmentation, morphology,
transforms) feeds the classifier instead of being bypassed. No neural
network was trained or benchmarked.

**On the final model using only HOG + LBP:** the ablation below shows the
final classifier's input is HOG + LBP, not the full four-family set. This
is still entirely classical image processing, not a shortcut around it —
HOG (gradient-orientation histograms) and LBP (local binary patterns) are
themselves classical DIP texture descriptors, computed here with no
learned feature extractor anywhere in the pipeline. Weeks 3-5 were
*measured*, not skipped: `src/features.py` computes morphology and
transform features for every image and `src/classify.py`'s ablation
(Section "Feature-family ablation" below) quantifies exactly what each
contributed on `val`, exactly as this course's Week 6 deliverable asks
for. Given the modest standalone numbers those weeks already reported —
morphology-only RandomForest 0.618 val / 0.682 test, morph+transform
0.654 val / 0.764 test, and a Hough open-vs-closed detection gap of only
0.205 (`docs/week5_transforms.md`) — HOG+LBP ending up as the strongest
single family and morph+transform adding nothing on top of it is
consistent with those earlier weeks' own honest findings, not a
surprise.

## Feature families

| family | dims | source |
|---|---|---|
| `morph` | 14 | Week 3/4 `regionprops` shape features on the cleaned pupil/iris mask (`data/features/segmorph_features.csv`, `morph_*` columns) |
| `transform` | 19 | Week 5 DFT high-freq ratio + 8 radial bins, DCT compaction (k=2/4/8/16), Laplacian variance, Hough found/cx/cy/radius/n_circles (`data/features/transform_features.csv`) |
| `hog` | 1764 | Histogram of Oriented Gradients on the 64x64 preprocessed crop (9 orientations, 8x8 cells, 2x2 blocks, L2-Hys norm) — computed here for the first time |
| `lbp` | 10 | Uniform LBP histogram (P=8, R=1) on the same crop — computed here for the first time |

`morph`/`transform` are joined on `filename` from their existing CSVs
(never recomputed in bulk). `hog`/`lbp` are computed once over all 6000
images and cached to `data/features/hog_lbp_cache.npz` (~11s to build,
~33MB) so repeated notebook/script runs don't recompute every time.

**Non-finite values:** checked directly — neither feature CSV contains any
NaN/inf over all 6000 rows (0 non-finite cells found and imputed on this
run). The Hough "not found" case already writes an explicit sentinel 0.0
for `hough_cx/cy/radius/n_circles`, gated by `hough_found=0`, not NaN.
`build_feature_matrix` still defensively runs the assembled matrix through
`np.nan_to_num` (nan→0.0, ±inf→±1e6) and reports the count touched, so a
future recompute or a different crop hitting a genuine 0/0 would be caught
and auditable rather than silently propagating.

## Model selection protocol

1. **Model-type search** (`src.classify.search_model_types`) — LinearSVC (5
   values of C), RBF-SVM (3 C x 2 gamma), RandomForest (2 n_estimators x 2
   max_depth) — 15 fits total, all on the full 4-family (1807-dim) feature
   matrix, fit on `train` (3720 rows), scored on `val` (1140 rows).
2. **Feature-family ablation** (`src.classify.run_ablation`) — the winning
   model type/hyperparameters retrained from scratch on `train`, once per
   family combination, scored on `val`. This also picks the final family
   combination.
3. **Final model** — the winning (model type, hyperparameters, family
   combo) refit on `train` only, wrapped in `CalibratedClassifierCV`
   (5-fold, sigmoid/Platt scaling, fit entirely within `train`) so every
   model type exposes a proper `predict_proba` for ROC/PR curves and for
   `predict_eye_state`'s confidence score.
4. **Test split touched exactly once** — `evaluate_on_test` is the single
   call in the whole pipeline that reads `test`, run only for this one
   final bundle. Confirmed: no other code path in `src/classify.py` or the
   notebook references the `test` split.

Standardization: `StandardScaler` lives inside the same `sklearn.Pipeline`
as the classifier for LinearSVC/RBF-SVM, so `.fit()` only ever sees
`train` rows; RandomForest pipelines skip scaling (tree splits are
scale-invariant).

### Model-type search results (val, all 4 families, 1807 dims)

| model | params | val accuracy | fit time |
|---|---|---|---|
| **rbf_svm** | **C=10, gamma='scale'** | **0.8842** | 13.5s |
| rbf_svm | C=1, gamma='scale' | 0.8798 | 10.5s |
| random_forest | n_estimators=400, max_depth=20 | 0.8684 | 6.0s |
| random_forest | n_estimators=200, max_depth=None | 0.8675 | 3.1s |
| random_forest | n_estimators=400, max_depth=None | 0.8675 | 6.0s |
| random_forest | n_estimators=200, max_depth=20 | 0.8667 | 3.0s |
| linear_svc | C=0.001 | 0.8605 | 1.2s |
| rbf_svm | C=0.1, gamma='scale' | 0.8430 | 14.6s |
| linear_svc | C=0.01 | 0.8360 | 1.7s |
| linear_svc | C=0.1 | 0.8254 | 3.1s |
| linear_svc | C=1.0 | 0.8228 | 5.1s |
| linear_svc | C=10.0 | 0.8219 | 3.1s |
| rbf_svm | C=0.1, gamma=0.01 | 0.5000 | 20.2s |
| rbf_svm | C=1.0, gamma=0.01 | 0.5000 | 20.4s |
| rbf_svm | C=10.0, gamma=0.01 | 0.5000 | 20.2s |

`gamma=0.01` collapses to predicting a single class (0.5 = majority-class
accuracy on the balanced set) — too small a kernel bandwidth for these
feature scales; `gamma='scale'` (the data-adaptive default) is what
actually works. **RBF-SVM (C=10, gamma='scale') is the chosen model type**
— it beats every LinearSVC and RandomForest configuration tried, and beats
the LinearSVC(C=0.01)-on-HOG+LBP baseline measured earlier in the project
(0.840 val) by 4.4 points.

## Feature-family ablation (chosen model: RBF-SVM, C=10, gamma='scale')

| family combo | dims | val accuracy |
|---|---|---|
| morph | 14 | 0.6298 |
| transform | 19 | 0.6596 |
| hog | 1764 | 0.8833 |
| lbp | 10 | 0.5263 |
| morph + transform | 33 | 0.6640 |
| **hog + lbp** | **1774** | **0.8842** |
| morph + transform + hog + lbp (all) | 1807 | 0.8842 |

**Read honestly:** morphology alone (Week 3/4) reaches 0.630, and DFT/DCT/
Hough transforms alone (Week 5) reach 0.660 — both real, above-chance
signal, consistent with the modest gaps already reported in
`docs/week4_morphology.md`/`docs/week5_transforms.md`. Combined,
morph+transform only creeps to 0.664 — the two families are largely
redundant with each other (both are proxies for "is there a dark, round
pupil-shaped region here"). HOG alone already reaches 0.883; LBP alone is
weak in isolation (0.526, barely above chance) but adds nothing measurable
once combined with HOG (0.8833 → 0.8842, +0.0009) beyond a small texture
top-up.

**Precisely what the morph+transform-on-top-of-hog+lbp comparison showed:**
`hog+lbp` = 0.8842105263157894 val accuracy; `morph+transform+hog+lbp`
(all 4 families) = 0.8842105263157894 val accuracy — identical to full
double-precision, i.e. **exactly the same 1008/1140 validation predictions
correct in both configurations, not merely equal after rounding.** This is
finding **(b) — no measurable gain — not (a) — actively harmful.**
Accuracy did not drop when morph+transform were added; it did not move at
all. `hog+lbp` is preferred as the final family combination purely for
parsimony (1774 dims instead of 1807, marginally faster to compute and
persist) given a true tie, not because morph+transform hurt anything.

## Final model & test metrics (test touched exactly once)

**Chosen model:** RBF-SVM, `C=10, gamma='scale'`, on **HOG + LBP** features
(1774 dims), calibrated with `CalibratedClassifierCV` (sigmoid, 5-fold,
**`ensemble=False`**, fit on `train` only — see "Persisted model size" below
for why `ensemble=False` specifically).

| metric | value |
|---|---|
| Accuracy | 0.8965 |
| Precision (positive = open) | 0.9363 |
| Recall (positive = open) | 0.8509 |
| F1 | 0.8915 |
| AUC | 0.9621 |

Confusion matrix (rows = true, cols = predicted, order [closed, open]):

| | pred closed | pred open |
|---|---|---|
| **true closed** | 537 | 33 |
| **true open** | 85 | 485 |

Recall on `open` (0.851) is lower than precision (0.936): the model is more
likely to miss an open eye (call it closed) than to falsely call a closed
eye open. For a drowsiness-detection use case this is the safer direction
to err in (a missed "eyes open" reads as a false drowsy alert, not a missed
real closed-eye/drowsy event) — though this was not a design target, just
an observed asymmetry worth noting for Week 7/8's decision threshold.

*(These are the `ensemble=False` numbers, used in the shipped
`models/eye_state_classifier.joblib`. An earlier `ensemble=True` run —
same model type, hyperparameters, and family combo, before the size fix
below — scored 0.8982/0.9332/0.8579/0.8940/0.9625: every metric moved by
≤0.007. The ablation table above is unaffected either way, since ablation
uses uncalibrated pipelines.)*

## Per-subject test accuracy (8 test subjects)

| subject | n | accuracy |
|---|---|---|
| s0017 | 160 | 0.9938 |
| s0007 | 83 | 0.9880 |
| s0003 | 162 | 0.9444 |
| s0032 | 160 | 0.9250 |
| s0033 | 160 | 0.9187 |
| s0009 | 95 | 0.8947 |
| s0015 | 160 | 0.8875 |
| s0016 | 160 | 0.6625 |

**Range: 0.331** (66.3% to 99.4%) across only 8 subjects — a single mean
test accuracy hides this entirely. `s0016` is the clear outlier and is
*not* explained by the annotated acquisition conditions: all 160 of their
test images have `glasses=0`, and 153/160 have `reflections=0` (i.e.
nothing unusual on paper). Their confusion breakdown: 49 of 89 true-`open`
images were predicted `closed` (a 55.1% false-negative rate on this one
subject alone, vs. 14.9% overall), while true-`closed` images were mostly
handled fine (66/71 correct). This looks like a genuine subject-appearance
generalization gap — this subject's open-eye HOG/LBP texture signature
apparently resembles other subjects' closed-eye signature closely enough
to fool the model — rather than an artifact of lighting, glasses, or
reflections.

## Error analysis: is it glasses / lighting / reflections?

Cross-tabulated test-split error rate by condition:

| condition | value | n | errors | error rate |
|---|---|---|---|---|
| glasses | 0 (no) | 1092 | 114 | 0.1044 |
| glasses | 1 (yes) | 48 | 4 | 0.0833 |
| lighting | 0 (bad) | 508 | 50 | 0.0984 |
| lighting | 1 (good) | 632 | 68 | 0.1076 |
| reflections | 0 (none) | 1036 | 110 | 0.1062 |
| reflections | 1 (low) | 24 | 1 | 0.0417 |
| reflections | 2 (high) | 80 | 7 | 0.0875 |

**None of these show the a-priori expected pattern.** `glasses=1` images
actually have a *lower* error rate (8.3%) than `glasses=0` (10.4%);
`lighting=0` (bad, 9.8%) and `lighting=1` (good, 10.8%) are essentially
indistinguishable; `reflections=2` (high, 8.8%) is not worse than
`reflections=0` (none, 10.6%). Reported honestly rather than forced into
the expected narrative: **who the subject is matters far more to this
model's errors than the three annotated acquisition conditions do** — see
the per-subject breakdown above. That said, several of these cells are
small (`glasses=1` n=48, `reflections=1` n=24), and the test split has
only 8 subjects contributing unevenly to each condition (e.g. `glasses=1`
rows likely come from very few of the 8 subjects), so these per-condition
rates are suggestive, not a clean population-level statement — a
condition x subject interaction confound cannot be ruled out with this
little data. The worst-failures contact sheet (`notebooks/06_...ipynb`,
Section 5; `results/week6_worst_failures.png`) shows the highest-confidence
wrong predictions individually rather than only as aggregate rates.

## Model persisted for Week 7

`models/eye_state_classifier.joblib` — a `src.classify.ClassifierBundle`
(the fitted calibrated pipeline + families used + HOG/LBP configs +
feature-name order + the Week-2 preprocess config), saved with `joblib`.
Entry point: `src.classify.predict_eye_state(img) -> (label, confidence)`,
which takes a raw grayscale crop of any size, applies
`src.preprocess.preprocess_pipeline` internally, computes HOG+LBP on the
enhanced crop, and returns `"open"`/`"closed"` with a calibrated-probability
confidence in [0, 1].

### Persisted model size: 119 MB → 27 MB

The first save of this bundle was **119.5 MB** — over GitHub's 100 MB/file
limit, and Week 7 needs to load this file, so it had to be fixed, not
worked around. Root cause: `CalibratedClassifierCV`'s default
`ensemble=True` keeps one independently-fitted copy of the base RBF-SVM
per CV fold (5 here), and each fitted `SVC` carries its own
`support_vectors_` array (1664 x 1774 float64 ≈ 23.6 MB) — 5 copies ≈ 118
MB, essentially all of the bundle's size, for zero measurable accuracy
benefit (see metric deltas above).

**Fix applied:** `CalibratedClassifierCV(..., ensemble=False)` — the base
estimator is refit once on the full `train` split, and the `cv` folds are
used only internally to get out-of-sample scores for fitting a single
sigmoid calibrator, so exactly one `SVC` (one `support_vectors_` array) is
persisted. Result: **26.77 MB (25.5 MiB, 26,768,275 bytes) on disk** (`joblib.dump(..., compress=3)`),
comfortably under the 100 MB limit — verified directly with `ls -la`, not
assumed.

**Float32 support vectors were tried and rejected, not silently skipped.**
Casting a fitted `SVC`'s `support_vectors_`/`dual_coef_` to `float32` post-hoc
was tested in isolation first: `clf.decision_function(X)` then raises
`ValueError: Buffer dtype mismatch, expected 'const float64_t' but got
'float'` — scikit-learn's libsvm/Cython predict path hard-requires
`float64` support-vector arrays; it is not a dtype the Python-level object
can just be downcast to before calling `.predict()`/`.decision_function()`.
Since `ensemble=False` alone already gets to 26.77 MB (a healthy margin
under 100 MB), this was not pursued further — pushing prediction-breaking
changes for a size win that wasn't needed would have been the wrong
trade.

**Verification after the fix:** `predict_eye_state` was re-run on 30
random test-split images after retraining with `ensemble=False`; 29/30
matched their true label (consistent with the ~0.90 accuracy reported
above — this was a correctness smoke test, not a full re-evaluation). The
full test-set metrics/confusion-matrix/per-subject/error-crosstab numbers
throughout this document are already the post-fix (`ensemble=False`)
numbers, freshly computed from the 26.77 MB bundle.

## Honest limitations

- **Texture, not shape/frequency, is carrying this result.** Morphology
  and transform features (the Week 3/4/5 DIP topics most directly tied to
  "classical" pupil/iris geometry) contribute real but modest standalone
  signal (0.63/0.66 val) and add literally nothing once HOG+LBP texture is
  present. This is an honest, not a flattering, finding about which
  mandated topics ended up mattering for this specific classification task
  on this dataset.
- **Ablation hyperparameters were not re-tuned per family** — the same
  RBF-SVM (C=10, gamma='scale') was reused across all 7 combinations to
  keep the search time-boxed. A weak family might do marginally better
  with its own tuned hyperparameters, though the ~22-point gap to HOG+LBP
  makes it unlikely this would change which family combo wins.
- **8 test subjects.** Per-subject accuracy swings 33.1 points, and some
  error-crosstab cells have as few as 24 rows — both the per-subject and
  per-condition numbers above should be read as indicative of real
  variance at this sample size, not as tight population estimates.
- **No neural-network ceiling comparison.** By explicit project scope, a
  CNN was never trained or benchmarked, so it is unknown how much headroom
  above 0.897 test accuracy a learned-feature model would find on this
  same split.
- **MRL near-IR, 64x64 only.** This result says nothing about
  generalization to visible-light webcam footage; Week 7's live inference
  pipeline is the first real test of that gap.
- **`data/features/hog_lbp_cache.npz` (~33MB)** is a rebuildable cache
  (`python -m src.features` regenerates it deterministically from
  `data/working_set.csv`), not a hand-authored artifact — worth knowing
  before deciding whether to commit it as-is.
