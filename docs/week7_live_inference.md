# Week 7 — Live Inference: Face/Eye Localization, EAR/MAR, and PERCLOS

Digital Image Processing mini project: Driver Drowsiness Detection.
Amey Pawar (23108B0057), Sejal Andhale (23108B0049).

Full evidence (landmark-index verification figure, PERCLOS validation
timeline, fps measurements) is in `notebooks/07_live_inference.ipynb`,
executed top-to-bottom with stored outputs. This document is the
condensed report. Code lives in `src/detect.py` (face/eye localization),
`src/ear.py` (EAR/MAR + adaptive thresholding), `src/perclos.py`
(rolling-window PERCLOS + hysteresis alert), `scripts/make_synthetic_sequence.py`
(labelled synthetic validation sequence), and `scripts/run_demo.py` (the
runnable live/offline demo).

## What was built

| Module | Role |
|---|---|
| `src/detect.py` | `FaceEyeDetector`, two backends: **mediapipe** (primary, `FaceLandmarker` Tasks API, gives face box + eye boxes + 478 landmarks) and **haar** (fallback, vendored cascades, gives face + eye boxes only, no landmarks). Shared `detect()` / `get_eye_crops()` interface regardless of backend. Asserts `.empty()` on both cascades at construction with an error naming the vendored path. |
| `src/ear.py` | `eye_aspect_ratio` / `mean_eye_aspect_ratio` / `mouth_aspect_ratio` (Soukupová & Čech EAR; analogous MAR), landmark index sets, and `EARCalibrator` — the adaptive per-user threshold. |
| `src/perclos.py` | `PERCLOSTracker` — rolling time-window PERCLOS, source-agnostic (classifier label or EAR-below-threshold), two-threshold hysteresis + minimum-dwell debouncing for the alert flag. |
| `scripts/make_synthetic_sequence.py` | Builds `data/synthetic/sequence.mp4` + `sequence_manifest.csv`: a **constructed** (not real) timed sequence from real MRL stills, with known per-frame ground truth, simulating alert/blink/microsleep periods. |
| `scripts/run_demo.py` | Runs the full pipeline over `--source webcam` or `--source video --video PATH`, overlaying boxes, EAR/MAR, eye state + confidence, rolling PERCLOS, and an ALERT banner. |
| `notebooks/07_live_inference.ipynb` | Landmark verification figure, PERCLOS-vs-ground-truth validation, classifier-vs-EAR-PERCLOS comparison (and why it's limited), fps measurement. |

## Why EAR/MAR are never computed on the MRL dataset

MRL Eye Dataset images (`data/samples/`, `data/working_set.csv`) are
already-cropped eye patches with no surrounding face. EAR needs 6
per-eye landmarks and MAR needs 4 mouth landmarks from a face-mesh model
run on a full face — there is nothing for a face-landmark model to
detect in an eye crop. This is enforced structurally, not just by
convention: `src/detect.FaceEyeDetector` run on the synthetic sequence
(itself built from MRL crops) correctly reports **no face on 0/588
frames** (see notebook Section 4 / `scripts/run_demo.py`'s printed
`face detected on 0/588 frames`), so the EAR code path is never reached
on that data — `scripts/run_demo.py` falls back to feeding the whole
frame straight to the Week-6 classifier instead.

## Landmark indices — verified, not assumed

MediaPipe 1.0.0 removed `mp.solutions.face_mesh`; only the Tasks API
(`mediapipe.tasks.python.vision.FaceLandmarker`) remains, returning 478
landmarks (468 face-mesh + 10 iris points) in the same canonical
topology. The commonly-quoted index sets (left eye
33/160/158/133/153/144, right eye 362/385/387/263/373/380, mouth
13/14/78/308) were checked against **real detector output**, not trusted
from a tutorial: running `FaceLandmarker` on a real face photo (Google's
own MediaPipe demo asset, `storage.googleapis.com/mediapipe-assets/portrait.jpg`,
a public-domain US-government portrait, downloaded once for this offline
check — not project-collected data, saved at
`data/synthetic/sample_face_portrait.jpg`) confirmed:
  - horizontal corner pairs (33/133, 362/263) sit ~35px apart on that
    photo, upper/lower-lid pairs ~7-8px apart — correctly much smaller,
    giving EAR ≈ 0.20-0.21 for an open eye (plausible, non-degenerate);
  - cropping a 16-point contour bounding box around each eye and saving
    it as an image visually contains the eye (confirmed by direct visual
    inspection during development — left/right assignment is not
    swapped);
  - mouth landmarks gave MAR ≈ 0.19 on that (closed-mouth) photo,
    consistent with a small ratio for a closed mouth.

See `notebooks/07_live_inference.ipynb` Section 1 for the reproducible
check and the annotated figure (`results/week7_landmark_overlay.png`).
No public-domain photo with a *closed* eye was available, so the
closed-eye case is not separately verified this way — stated as a gap,
not silently assumed to also be correct.

## Adaptive / per-user EAR thresholding — the research gap this project claims to address

A single global EAR cutoff (commonly ~0.2-0.25 in tutorials) does not
transfer across users: eyelid shape, eye size, camera angle/distance all
shift a person's *open-eye* EAR up or down, so a fixed cutoff either
never fires for someone with a naturally low open-eye EAR or fires
constantly for someone with a naturally high one.

**Method implemented (`src.ear.EARCalibrator`):**
1. For the first `calib_seconds` (default 3.0s) after the session
   starts, every frame's EAR is accumulated — the demo states the
   assumption to the user that they should keep eyes open during this
   window (it is not auto-detected).
2. Calibration is summarized as the **90th percentile** of the collected
   samples, not the mean/median — a blink during calibration pulls the
   mean/median down, but is a small minority of a few seconds of frames,
   so a high percentile is robust to a handful of low outliers and
   estimates the subject's typical open-eye EAR.
3. The live closure threshold is `baseline_open_ear * closure_ratio`
   (default ratio 0.75), so the absolute cutoff floats per-user instead
   of being one hardcoded constant.

Demonstrated converging to a plausible value on the one available face
(baseline 0.208, threshold 0.156 — see `notebooks/07_live_inference.ipynb`
Section 4 and the manual `scripts/run_demo.py` run below). **Not yet
demonstrated to out-perform a fixed global threshold**, because that
comparison needs real footage from more than one person — a stated
limitation, not a claimed result.

## PERCLOS: window, thresholds, and why (`src/perclos.py`)

PERCLOS (PERcentage of eye CLOSure) is the measure identified by the
Carnegie Mellon Research Institute driving-simulator studies (Wierwille
et al., 1994) and the FHWA-sponsored validation (Dinges & Grace, 1998,
*"PERCLOS: A Valid Psychophysiological Measure of Alertness As Assessed
by Psychomotor Vigilance"*) as the best-performing eye-closure-based
drowsiness predictor they compared — the closest thing this field has to
a standard measure, which is why it is this project's headline
statistic.

Chosen configuration (`PERCLOSConfig` defaults):

| parameter | value | justification |
|---|---|---|
| `window_seconds` | 10.0 | The original CMU/FHWA studies used 1-3 minute windows over multi-hour sessions. This project's synthetic demo sequences run ~30-60s total, so a 1-3 minute window would rarely fill even once. 10s is a **demo-scale adaptation**, stated as such, not a claim that 10s is the validated research window. |
| `enter_threshold` | 0.15 | 15% closure is a widely-cited "drowsy" cutoff for P80-style PERCLOS in the literature. |
| `exit_threshold` | 0.10 | This project's own addition on top of the cited studies (hysteresis), not from them — added so the alert does not clear the instant PERCLOS dips slightly below the enter threshold. |
| `min_enter_seconds` / `min_exit_seconds` | 1.0 / 2.0 | A second (temporal) debounce layer on top of the two-threshold hysteresis, so a single noisy frame cannot flip the alert. |

## Synthetic sequence: what it is, and what it is not

**This is a constructed sequence, not real driving footage.** Every
frame is a genuine MRL camera capture of a real eye; the *order and
timing* — which stills appear for how long, in what sequence — is
authored by `scripts/make_synthetic_sequence.py` to simulate: an alert
period, a few normal blinks (0.2-0.3s closures), a sustained ~2.5-3s
microsleep, recovery, a second microsleep, and a final alert period — for
two subjects (`s0001`, `s0030`, chosen for having >1000 real stills in
both eye states each) concatenated into one ~58.8s / 588-frame,
10fps session. Per-frame ground truth (`open`/`closed`) is known exactly
by construction and written to `data/synthetic/sequence_manifest.csv`
alongside the rendered `data/synthetic/sequence.mp4` (648KB — well under
the ~10MB commit budget, committed as-is).

Because these are eye crops with no face, this sequence validates the
**classifier-driven PERCLOS path only** — see the constraint above.

## PERCLOS validation against synthetic ground truth

From `notebooks/07_live_inference.ipynb` Section 2 (`results/week7_perclos_timeline.png`):

- **Frame-level accuracy** of the Week-6 classifier on this sequence
  (treating each still as a fresh eye crop): **0.9031** — closely
  matching Week-6's own held-out MRL test accuracy (0.8965), confirming
  the classifier is not degrading just because stills were re-sequenced
  into a video.
- **Alert-state agreement** (classifier-driven tracker vs ground-truth
  tracker, same config, frame-by-frame): **0.7109** — markedly lower
  than frame accuracy.
- **Ground-truth alert transitions:** onset at t=16.7s (off at 36.6s),
  onset again at t=46.1s.
- **Classifier-driven alert transitions:** onset at **t=9.2s** — over
  7 seconds *before* the real (authored) microsleep — and never clears
  before the sequence ends.

**Honest reading:** the classifier's ~10% per-frame error rate (45 false
"closed" / 460 true-open frames, 12 false "open" / 128 true-closed
frames — a balanced FP/FN split) is enough independent noise that
scattered misclassifications during a long alert period accumulate
inside the 10s rolling window and cross the 15% enter threshold *before*
the genuine microsleep does. This is a real finding about combining a
noisy per-frame classifier with a simple windowed-mean threshold, not a
bug being hidden: independent frame noise and genuine sustained closure
both raise the same statistic, so a false alarm can arrive first. The
two-threshold hysteresis and dwell-time debouncing already in
`src/perclos.py` mitigate frame-to-frame flicker; they do not by
themselves prevent a false alarm building up over several seconds of
noisy input. A natural next step — smoothing the per-frame label itself
(e.g. requiring N consecutive closed predictions) before it reaches the
PERCLOS window — is noted as future work, deliberately **not**
implemented here to avoid retroactively tuning against this one
sequence.

## Classifier-driven vs EAR-driven PERCLOS — compared where possible

The two cannot be compared on the same ground-truth-labelled sequence,
for a structural reason: the only sequence with known closure timing
(the synthetic one) is built from face-less MRL crops, so EAR is
undefined on it (`FaceEyeDetector` correctly reports 0/588 faces found).
What *is* confirmed: the EAR/MAR code path runs correctly and produces
sane numbers on a real face (EAR≈0.204, MAR≈0.187 on the one available
public-domain photo — see landmark section above), and `EARCalibrator`
converges to a threshold in the expected ~3s. What is **not** validated
is EAR-driven PERCLOS's temporal behavior against a real blink/microsleep
sequence — that requires a real video of a face with actual eye closures
over time, i.e. the user's own webcam/recorded footage. Synthetically
animating EAR values on a static photo to backfill a plot would produce
a fabricated "validation" and was deliberately not done.

## Measured fps (this development machine, single CPU thread)

From `notebooks/07_live_inference.ipynb` Section 5 (also independently
reproduced via `scripts/run_demo.py`):

| path | fps | ms/frame |
|---|---|---|
| classifier-only fallback (no face; the synthetic sequence's path) | 186.4 | 5.4 |
| full mediapipe pipeline (landmarks + EAR/MAR + 2x classifier) | 46.8 | 21.4 |
| haar backend (face+eye boxes + classifier, no EAR/MAR) | 17.6 | 56.9 |

These are not a benchmark of any specific deployment target (phone,
embedded board, etc.) — just what this one dev machine achieves, useful
as a sanity check that the full pipeline is comfortably real-time-capable
(46.8 fps ≫ typical webcam 15-30fps) on ordinary hardware.

`scripts/run_demo.py`'s own manual run on the synthetic video (headless,
`--save-video`, no display) additionally confirmed:
```
processed 588 frames in 6.71s (87.6 fps achieved)
face detected on 0/588 frames; fallback whole-frame-as-eye-crop on 588/588
final PERCLOS[classifier]=35.64% alert=True
EAR never calibrated (no face/landmarks detected long enough)
```
and, run against a repeated-frame single-face test video (80 frames):
```
processed 80 frames in 2.83s (28.3 fps achieved)
face detected on 80/80 frames; fallback whole-frame-as-eye-crop on 0/80
final PERCLOS[classifier]=0.00% alert=False
EAR baseline_open=0.208 threshold=0.156
final PERCLOS[EAR]=0.00% alert=False
```
(fps differs slightly from the notebook's isolated timing loop above
because this run also includes cv2 video I/O and overlay drawing, not
just inference.)

## Honest limitations

- **EAR/MAR/PERCLOS on a real driver's face is unvalidated** until the
  user records real webcam/driving footage. Everything in this project
  confirms the *mechanism* is wired correctly (landmark indices correct,
  EAR/MAR numerically sane, calibration converges, PERCLOS windowing and
  hysteresis behave as designed on known ground truth) — not that it
  will correctly flag a real drowsy driver.
- **`--source webcam` is implemented but untested** in this environment
  — no camera exists here, and `cv2.VideoCapture(0)` was never invoked
  (it would hang). It uses the exact same detection/EAR/PERCLOS code
  path as `--source video`, so it should work unmodified, but "should"
  is not "verified."
- **PERCLOS window (10s) is a demo-scale adaptation**, not the
  1-3 minute window used in the original CMU/FHWA validation studies.
- **The adaptive EAR threshold's benefit over a fixed global threshold
  is not demonstrated** — only one real face was available to test EAR
  on at all; a multi-subject comparison needs real footage from more
  than one person.
- **Classifier-driven PERCLOS alert *timing* is unreliable** at this
  classifier's ~10% error rate combined with a simple windowed-mean
  threshold (see validation section above) — frame-level accuracy is
  good, but alert-onset timing arrived over 7s early on the one
  synthetic sequence tested. This is reported, not hidden, and is the
  single most important limitation of the current pipeline.
- **fps numbers are single-machine, single-thread** measurements, not a
  deployment benchmark.

## Reproducing this week's work

```
.venv/bin/python scripts/make_synthetic_sequence.py     # writes data/synthetic/
.venv/bin/python scripts/run_demo.py --source video --video data/synthetic/sequence.mp4 --save-video /tmp/demo_out.mp4
jupyter nbconvert --to notebook --execute notebooks/07_live_inference.ipynb --inplace
```
