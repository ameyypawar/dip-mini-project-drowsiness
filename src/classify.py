"""Week-6 classification: train/compare/select an eye-state (open=1/closed=0)
classifier over the four Week 2-5 classical-feature families assembled by
`src.features.build_feature_matrix`.

Protocol (see docs/week6_classification.md for the full writeup):

  1. `search_model_types` -- compare LinearSVC / RBF-SVM / RandomForest (a
     small, time-boxed hyperparameter grid each) on the full "all families"
     feature set, fit on `train`, scored on `val`. This picks the model
     *type* + hyperparameters.
  2. `run_ablation` -- retrain that one model type/hyperparams on each
     feature-family subset (`ABLATION_COMBOS`), fit on `train`, scored on
     `val`. This is the ablation table (deliverable 3) *and* picks the final
     feature-family combination (whichever scores highest on val).
  3. `train_final_model` -- refit the chosen (model type, hyperparams,
     family combo) on `train` only, wrapped in `CalibratedClassifierCV` so
     every model type exposes a proper `predict_proba` (needed for ROC/PR
     curves and for `predict_eye_state`'s confidence).
  4. `evaluate_on_test` -- the *only* place the test split is touched, and
     only for this one final bundle.

Standardization is fit on `train` only in every case: `StandardScaler` lives
inside the same `sklearn.Pipeline` as the classifier, so `.fit()` on a train
subset never sees val/test rows, and RandomForest pipelines skip scaling
entirely (tree splits are scale-invariant, and skipping avoids doing
pointless work).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.features import (
    DEFAULT_HOG_CONFIG,
    DEFAULT_LBP_CONFIG,
    FAMILIES,
    HOGConfig,
    LBPConfig,
    WORKING_SET_CSV,
    build_feature_matrix,
    extract_family_features_single,
    load_working_set,
    vectorize_features,
)
from src.preprocess import DEFAULT_CONFIG, preprocess_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "eye_state_classifier.joblib"

POS_LABEL = 1  # eye_state == 1 -> open, matches src.dataset.EyeSample.is_open
LABEL_NAMES = {0: "closed", 1: "open"}

# ---------------------------------------------------------------------------
# Model-type search grid (time-boxed: 5 + 6 + 4 = 15 fits on ~3720x1807)
# ---------------------------------------------------------------------------

SEARCH_GRID: dict[str, list[dict]] = {
    "linear_svc": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0)],
    "rbf_svm": [{"C": c, "gamma": g} for c in (0.1, 1.0, 10.0) for g in ("scale", 0.01)],
    "random_forest": [
        {"n_estimators": n, "max_depth": d}
        for n in (200, 400)
        for d in (None, 20)
    ],
}

ABLATION_COMBOS: list[tuple[str, ...]] = [
    ("morph",),
    ("transform",),
    ("hog",),
    ("lbp",),
    ("morph", "transform"),
    ("hog", "lbp"),
    ("morph", "transform", "hog", "lbp"),
]


def _needs_scaling(model_name: str) -> bool:
    return model_name in ("linear_svc", "rbf_svm")


def make_estimator(model_name: str, params: dict):
    if model_name == "linear_svc":
        return LinearSVC(max_iter=10_000, **params)
    if model_name == "rbf_svm":
        return SVC(kernel="rbf", **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    raise ValueError(f"unknown model_name {model_name!r}")


def make_pipeline(model_name: str, params: dict, calibrate: bool = False, cv: int = 5) -> Pipeline:
    """Build the sklearn Pipeline for `model_name`. Scaling (fit on whatever
    data `.fit()` is later called with -- always `train` in this module) is
    included only for the SVM variants. `calibrate=True` wraps the
    classifier in `CalibratedClassifierCV` so `.predict_proba` is available
    regardless of model type (used only for the final chosen model).

    `ensemble=False`: fits a *single* calibrated classifier (base estimator
    refit on the full training set; the `cv` splits are used only to get
    out-of-sample predictions for fitting the sigmoid calibrator) instead
    of the default `ensemble=True`, which keeps one fitted base estimator
    per CV fold (5x here) and averages their predictions. For an RBF-SVM
    final model this matters concretely for persistence size: each fold's
    fitted SVC carries its own `support_vectors_` (23.6 MB at this
    dataset's support-vector count), so `ensemble=True` bloated the
    persisted joblib bundle to ~119 MB -- over GitHub's 100 MB/file limit
    -- purely from storing 5 near-duplicate copies of the support vectors,
    not from any accuracy benefit. `ensemble=False` stores one, ~24 MB.
    Trade-off: a single calibration mapping instead of an average of 5 is
    very slightly noisier in principle; in practice here it left test
    accuracy/precision/recall/F1/AUC unchanged to 3+ decimal places (see
    docs/week6_classification.md) -- the accuracy cost is not measurable at
    this dataset size, and it is what makes the model persistable at all.
    """
    clf = make_estimator(model_name, params)
    if calibrate:
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=cv, ensemble=False)

    steps = []
    if _needs_scaling(model_name):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _labels(working_set: list[dict]) -> np.ndarray:
    return np.array([int(row["eye_state"]) for row in working_set])


def _split_mask(working_set: list[dict], split: str) -> np.ndarray:
    return np.array([row["split"] == split for row in working_set])


# ---------------------------------------------------------------------------
# Step 1: model-type + hyperparameter search (train/val only)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    model_name: str
    params: dict
    val_acc: float
    fit_seconds: float


def search_model_types(
    working_set: list[dict],
    families: Sequence[str] = FAMILIES,
    grid: dict[str, list[dict]] = SEARCH_GRID,
) -> list[SearchResult]:
    """Fit every (model_name, params) in `grid` on `train`, score on `val`.
    Returns all results sorted best-val_acc-first. Test is never touched.
    """
    fm = build_feature_matrix(working_set, families=list(families))
    y = _labels(working_set)
    train_mask = _split_mask(working_set, "train")
    val_mask = _split_mask(working_set, "val")
    Xtr, ytr = fm.X[train_mask], y[train_mask]
    Xval, yval = fm.X[val_mask], y[val_mask]

    results: list[SearchResult] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for model_name, param_list in grid.items():
            for params in param_list:
                t0 = time.perf_counter()
                pipe = make_pipeline(model_name, params, calibrate=False)
                pipe.fit(Xtr, ytr)
                val_acc = float(pipe.score(Xval, yval))
                elapsed = time.perf_counter() - t0
                results.append(SearchResult(model_name, params, val_acc, elapsed))
                print(f"  [{model_name:14s} {params}] val_acc={val_acc:.4f} ({elapsed:.1f}s)")

    results.sort(key=lambda r: r.val_acc, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Step 2: feature-family ablation (train/val only, chosen model type fixed)
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    families: tuple[str, ...]
    n_features: int
    val_acc: float


def run_ablation(
    working_set: list[dict],
    model_name: str,
    params: dict,
    combos: list[tuple[str, ...]] = ABLATION_COMBOS,
) -> list[AblationResult]:
    """Train `model_name`/`params` on `train`, score on `val`, once per
    family combo in `combos`. Test is never touched.
    """
    y = _labels(working_set)
    train_mask = _split_mask(working_set, "train")
    val_mask = _split_mask(working_set, "val")

    results: list[AblationResult] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for combo in combos:
            fm = build_feature_matrix(working_set, families=list(combo))
            Xtr, ytr = fm.X[train_mask], y[train_mask]
            Xval, yval = fm.X[val_mask], y[val_mask]
            pipe = make_pipeline(model_name, params, calibrate=False)
            pipe.fit(Xtr, ytr)
            val_acc = float(pipe.score(Xval, yval))
            results.append(AblationResult(combo, len(fm.feature_names), val_acc))
            print(f"  {'+'.join(combo):24s} ({len(fm.feature_names):4d} dims) val_acc={val_acc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Step 3: final model -- train on `train` only, calibrated for predict_proba
# ---------------------------------------------------------------------------

@dataclass
class ClassifierBundle:
    pipeline: Pipeline
    families: tuple[str, ...]
    feature_names: list[str]
    model_name: str
    params: dict
    hog_cfg: HOGConfig
    lbp_cfg: LBPConfig
    preprocess_cfg: object


def train_final_model(
    working_set: list[dict],
    families: Sequence[str],
    model_name: str,
    params: dict,
    hog_cfg: HOGConfig = DEFAULT_HOG_CONFIG,
    lbp_cfg: LBPConfig = DEFAULT_LBP_CONFIG,
) -> ClassifierBundle:
    """Fit the final chosen (families, model_name, params) on `train` only.
    Wrapped in `CalibratedClassifierCV` (5-fold, sigmoid/Platt scaling,
    fit entirely within `train`) so the bundle exposes `predict_proba`
    regardless of `model_name` -- LinearSVC has no native `predict_proba`,
    and a uniform probability interface is what ROC/PR curves and
    `predict_eye_state`'s confidence both need.
    """
    fm = build_feature_matrix(working_set, families=list(families))
    y = _labels(working_set)
    train_mask = _split_mask(working_set, "train")
    Xtr, ytr = fm.X[train_mask], y[train_mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        pipe = make_pipeline(model_name, params, calibrate=True, cv=5)
        pipe.fit(Xtr, ytr)

    return ClassifierBundle(
        pipeline=pipe,
        families=tuple(families),
        feature_names=fm.feature_names,
        model_name=model_name,
        params=params,
        hog_cfg=hog_cfg,
        lbp_cfg=lbp_cfg,
        preprocess_cfg=DEFAULT_CONFIG,
    )


def save_bundle(bundle: ClassifierBundle, path: str | Path = MODEL_PATH) -> Path:
    """Persist `bundle` via joblib, compressed (level 3 -- a middle ground
    between size and dump/load speed). Compression alone cannot fix a
    5x-duplicated support-vector array; see `make_pipeline`'s
    `ensemble=False` docstring for the actual fix to bundle size.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str | Path = MODEL_PATH) -> ClassifierBundle:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Step 4: test-split evaluation -- call exactly once, on the final bundle
# ---------------------------------------------------------------------------

def _predict(bundle: ClassifierBundle, working_set: list[dict], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, proba_open) for `working_set` rows where `mask` is True."""
    fm = build_feature_matrix(working_set, families=list(bundle.families))
    assert fm.feature_names == bundle.feature_names, "feature column order mismatch vs. trained bundle"
    y = _labels(working_set)
    X_sub, y_sub = fm.X[mask], y[mask]
    y_pred = bundle.pipeline.predict(X_sub)
    proba_open = bundle.pipeline.predict_proba(X_sub)[:, list(bundle.pipeline.classes_).index(POS_LABEL)]
    return y_sub, y_pred, proba_open


def evaluate_on_test(bundle: ClassifierBundle, working_set: list[dict]) -> dict:
    """The single, final touch of the test split. Returns accuracy,
    precision/recall/F1 (positive class = open/1), AUC, confusion matrix,
    and the raw (y_true, y_pred, proba) arrays for downstream plots
    (ROC/PR curves, error analysis) so nothing needs to re-predict.
    """
    test_mask = _split_mask(working_set, "test")
    y_true, y_pred, proba = _predict(bundle, working_set, test_mask)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=POS_LABEL)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POS_LABEL)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POS_LABEL)),
        "auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "roc_curve": roc_curve(y_true, proba),
        "pr_curve": precision_recall_curve(y_true, proba),
        "y_true": y_true,
        "y_pred": y_pred,
        "proba": proba,
        "test_mask": test_mask,
    }


def per_subject_test_accuracy(bundle: ClassifierBundle, working_set: list[dict]) -> dict[str, dict]:
    """Accuracy per `subject_id`, restricted to the test split (8 subjects)."""
    test_mask = _split_mask(working_set, "test")
    y_true, y_pred, _ = _predict(bundle, working_set, test_mask)
    subject_ids = np.array([row["subject_id"] for row in working_set])[test_mask]

    out: dict[str, dict] = {}
    for sid in sorted(set(subject_ids)):
        sel = subject_ids == sid
        out[sid] = {
            "n": int(sel.sum()),
            "accuracy": float(accuracy_score(y_true[sel], y_pred[sel])),
        }
    return out


def error_crosstab(
    bundle: ClassifierBundle,
    working_set: list[dict],
    attrs: tuple[str, ...] = ("glasses", "lighting", "reflections"),
) -> dict[str, dict]:
    """For each attribute in `attrs`, per-value counts of {n, n_correct,
    n_incorrect, error_rate} on the test split -- the error-analysis
    cross-tabulation (deliverable: are errors concentrated in glasses=1 /
    lighting=0 / reflections=2?).
    """
    test_mask = _split_mask(working_set, "test")
    y_true, y_pred, _ = _predict(bundle, working_set, test_mask)
    correct = y_true == y_pred
    test_rows = [row for row, keep in zip(working_set, test_mask) if keep]

    out: dict[str, dict] = {}
    for attr in attrs:
        values = np.array([row[attr] for row in test_rows])
        per_value: dict[str, dict] = {}
        for v in sorted(set(values)):
            sel = values == v
            n = int(sel.sum())
            n_correct = int(correct[sel].sum())
            per_value[v] = {
                "n": n,
                "n_correct": n_correct,
                "n_incorrect": n - n_correct,
                "error_rate": float((n - n_correct) / n) if n else 0.0,
            }
        out[attr] = per_value
    return out


def worst_failures(bundle: ClassifierBundle, working_set: list[dict], n: int = 16) -> list[dict]:
    """The `n` test-split misclassifications with highest predicted
    confidence in the *wrong* class -- the most surprising / worst failures,
    for the notebook's contact-sheet figure.
    """
    test_mask = _split_mask(working_set, "test")
    y_true, y_pred, proba = _predict(bundle, working_set, test_mask)
    test_rows = [row for row, keep in zip(working_set, test_mask) if keep]

    wrong_idx = np.where(y_true != y_pred)[0]
    # confidence in the (wrong) predicted class
    conf_wrong = np.where(y_pred[wrong_idx] == 1, proba[wrong_idx], 1.0 - proba[wrong_idx])
    order = wrong_idx[np.argsort(-conf_wrong)][:n]

    out = []
    for i in order:
        row = test_rows[i]
        out.append({
            "filename": row["filename"],
            "source_relpath": row["source_relpath"],
            "subject_id": row["subject_id"],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "proba_open": float(proba[i]),
            "glasses": row["glasses"],
            "lighting": row["lighting"],
            "reflections": row["reflections"],
        })
    return out


# ---------------------------------------------------------------------------
# Week-7 live entry point
# ---------------------------------------------------------------------------

_LOADED_BUNDLE: ClassifierBundle | None = None


def predict_eye_state(img: np.ndarray, model_path: str | Path = MODEL_PATH) -> tuple[str, float]:
    """Predict eye state for one raw grayscale eye crop.

    `img` is a raw (variable-size, uint8) grayscale crop -- e.g. a live
    webcam eye ROI (Week 7). Internally applies
    `src.preprocess.preprocess_pipeline` (the canonical Week-2 config,
    matching how every training feature was produced), computes the same
    feature families the persisted model was trained on
    (`src.features.extract_family_features_single`), and returns
    `(label, confidence)` where `label` is `"open"` or `"closed"` and
    `confidence` is the calibrated probability of the predicted class
    (`CalibratedClassifierCV`'s `predict_proba`, so it is a meaningful
    [0, 1] score for any of the three underlying model types).
    """
    global _LOADED_BUNDLE
    if _LOADED_BUNDLE is None:
        _LOADED_BUNDLE = load_bundle(model_path)
    bundle = _LOADED_BUNDLE

    enhanced = preprocess_pipeline(img, bundle.preprocess_cfg)
    feats = extract_family_features_single(enhanced, bundle.families, bundle.hog_cfg, bundle.lbp_cfg)
    X = vectorize_features(feats, bundle.feature_names)

    proba = bundle.pipeline.predict_proba(X)[0]
    classes = list(bundle.pipeline.classes_)
    pred = classes[int(np.argmax(proba))]
    confidence = float(proba[int(np.argmax(proba))])
    return LABEL_NAMES[int(pred)], confidence


# ---------------------------------------------------------------------------
# End-to-end CLI: search -> ablation -> final train -> test eval -> persist
# ---------------------------------------------------------------------------

def main() -> None:
    working_set = load_working_set(WORKING_SET_CSV)
    print(f"loaded {len(working_set)} rows")

    print("\n=== Step 1: model-type search (all families, train/val) ===")
    search_results = search_model_types(working_set)
    best = search_results[0]
    print(f"\nbest model type: {best.model_name} {best.params} (val_acc={best.val_acc:.4f})")

    print("\n=== Step 2: feature-family ablation (chosen model type, train/val) ===")
    ablation_results = run_ablation(working_set, best.model_name, best.params)
    best_combo = max(ablation_results, key=lambda r: r.val_acc)
    print(f"\nbest family combo: {best_combo.families} (val_acc={best_combo.val_acc:.4f})")

    print("\n=== Step 3: train final model (train only) ===")
    bundle = train_final_model(working_set, best_combo.families, best.model_name, best.params)
    path = save_bundle(bundle)
    print(f"saved {path}")

    print("\n=== Step 4: test evaluation (touched exactly once) ===")
    metrics = evaluate_on_test(bundle, working_set)
    print(f"test accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")
    print(f"confusion matrix [[TN,FP],[FN,TP]]:\n{metrics['confusion_matrix']}")

    per_subject = per_subject_test_accuracy(bundle, working_set)
    accs = [v["accuracy"] for v in per_subject.values()]
    print(f"per-subject test accuracy: min={min(accs):.4f} max={max(accs):.4f} "
          f"range={max(accs) - min(accs):.4f}")
    for sid, v in per_subject.items():
        print(f"  {sid}: n={v['n']:3d} acc={v['accuracy']:.4f}")


if __name__ == "__main__":
    main()
