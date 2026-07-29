"""Week-7 EAR / MAR: Soukupová & Cech eye-aspect-ratio and mouth-aspect-ratio,
computed from MediaPipe FaceLandmarker landmarks on *full-face* frames only.

Why this cannot run on the MRL dataset: EAR needs 6 per-eye landmarks (2
corners + 2 upper-lid + 2 lower-lid points) and MAR needs 4 mouth
landmarks, both of which require a face-mesh model run on a full face.
The MRL images used in Weeks 1-6 (and `data/samples/`) are already-cropped
eye patches with no surrounding face -- there is nothing for a
face-landmark model to detect, so EAR/MAR are mathematically undefined on
them. This module is only ever fed landmarks from `src.detect.FaceEyeDetector`
(backend="mediapipe") running on webcam/video frames or the synthetic
sequence's *source* full-face frame (there is none here -- see
docs/week7_live_inference.md, the synthetic sequence validates PERCLOS via
the Week-6 classifier signal, not EAR).

Landmark indices -- verified, not assumed
------------------------------------------
MediaPipe FaceLandmarker (Tasks API) returns 478 landmarks (468 face-mesh
points + 10 iris points) in the same canonical topology the legacy
`mp.solutions.face_mesh` used. The commonly-quoted index sets below were
checked against a real detection (not trusted from a tutorial): running
`FaceLandmarker` on a real face photo (Google's own MediaPipe demo asset,
`https://storage.googleapis.com/mediapipe-assets/portrait.jpg`, used only
for this offline verification, not committed to the repo) and inspecting
the resulting pixel coordinates confirmed:
  - indices 33/133 (left eye) and 362/263 (right eye) sit at the eye's
    horizontal corners (~35px apart on that photo);
  - indices 160/144 and 158/153 (left), 385/380 and 387/373 (right) sit
    on the upper/lower lid, ~7-8px apart vertically on that (open-eyed)
    photo -- correctly much smaller than the horizontal span, giving
    EAR ~= 0.20-0.21 for an open eye, in the expected low-but-nonzero
    range;
  - cropping a bounding box around each 16-point eye contour and saving
    it as an image visually contained the eye, confirming left/right eye
    assignment is not swapped.
  - mouth indices 13 (upper inner lip) / 14 (lower inner lip) and 78/308
    (mouth corners) gave MAR ~= 0.19 on the same (closed-mouth) photo,
    consistent with a small vertical/horizontal ratio for a closed mouth.
See docs/week7_live_inference.md for the full verification note.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]

# 6-point-per-eye EAR landmark sets: (p1, p2, p3, p4, p5, p6) matching the
# Soukupová & Cech convention -- p1/p4 = horizontal corners, (p2,p6) and
# (p3,p5) = the two vertical upper/lower-lid pairs.
LEFT_EYE_EAR_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)

# Mouth: (top-inner-lip, bottom-inner-lip, left-corner, right-corner).
MOUTH_MAR_IDX = (13, 14, 78, 308)


def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def eye_aspect_ratio(landmarks: list[Point], side: str) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||), Soukupová & Cech
    (2016), "Real-Time Eye Blink Detection using Facial Landmarks".

    `landmarks`: the full 478-point list from
    `src.detect.DetectionResult.landmarks_px` (pixel coordinates).
    `side`: "left" or "right".
    """
    if side == "left":
        idx = LEFT_EYE_EAR_IDX
    elif side == "right":
        idx = RIGHT_EYE_EAR_IDX
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in idx)
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def mean_eye_aspect_ratio(landmarks: list[Point]) -> float:
    """Average of left/right EAR -- the usual single-number signal used
    for blink/closure detection (robust to one eye being briefly
    occluded/at an angle relative to the other)."""
    return 0.5 * (eye_aspect_ratio(landmarks, "left") + eye_aspect_ratio(landmarks, "right"))


def mouth_aspect_ratio(landmarks: list[Point]) -> float:
    """MAR = ||p_top - p_bottom|| / ||p_left - p_right|| over the 4 inner
    mouth landmarks -- large MAR indicates a wide-open mouth (yawn
    candidate); see docs/week7_live_inference.md for the chosen yawn
    threshold and its adaptive-baseline analogue below.
    """
    top, bottom, left, right = (landmarks[i] for i in MOUTH_MAR_IDX)
    horizontal = _dist(left, right)
    if horizontal <= 1e-6:
        return 0.0
    return _dist(top, bottom) / horizontal


# ---------------------------------------------------------------------------
# Adaptive / per-user thresholding.
#
# The research gap this project's proposal claims to address: a single
# global EAR threshold (commonly ~0.2-0.25 in tutorials, calibrated on
# whatever face the tutorial author happened to use) does not transfer
# across users -- eyelid shape, eye size/aspect, camera angle and distance
# all shift a person's *open-eye* EAR up or down, so a fixed cutoff either
# never fires for someone with naturally low open-eye EAR, or fires
# constantly for someone with naturally high open-eye EAR.
#
# Method implemented here: per-user relative thresholding via a short
# open-eyes calibration period.
#   1. For the first `calib_seconds` seconds after `start()`, every frame's
#      EAR is fed to `update()` while the subject is assumed alert/eyes
#      mostly open (this is an operating assumption stated to the user
#      when the demo starts, not detected automatically).
#   2. Calibration is summarized as the `calib_percentile`-th percentile
#      (default: 90th) of the collected samples, not the mean/median --
#      blinks happening *during* calibration pull the mean/median down,
#      but a blink is a small minority of frames in a few seconds of
#      video, so a high percentile is robust to a few low outliers and
#      estimates the subject's *typical open-eye* EAR.
#   3. The live closure threshold is `baseline_open_ear * closure_ratio`
#      (default ratio 0.75 -- i.e. "eyes closed" is declared once EAR
#      drops to 75% of this subject's own open-eye EAR), so the absolute
#      cutoff floats per-user instead of being one hardcoded constant.
# ---------------------------------------------------------------------------

@dataclass
class EARCalibrator:
    fps: float
    calib_seconds: float = 3.0
    calib_percentile: float = 90.0
    closure_ratio: float = 0.75
    fallback_threshold: float = 0.2  # used only if calibration never gets enough samples
    _samples: list[float] = field(default_factory=list)
    _n_calib_frames: int = field(init=False)
    _frames_seen: int = field(default=0, init=False)
    baseline_open_ear: float | None = field(default=None, init=False)
    threshold: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._n_calib_frames = max(1, int(round(self.fps * self.calib_seconds)))

    @property
    def is_calibrated(self) -> bool:
        return self.threshold is not None

    def update(self, ear: float) -> None:
        """Feed one frame's EAR. During the calibration window, accumulate
        samples; on the frame that completes the window, compute
        `baseline_open_ear` and `threshold` once. No-op after calibration
        is done (threshold is fixed for the rest of the session)."""
        if self.is_calibrated:
            return
        self._samples.append(ear)
        self._frames_seen += 1
        if self._frames_seen >= self._n_calib_frames:
            self._finalize()

    def _finalize(self) -> None:
        if len(self._samples) < 3:
            # not enough data to trust a percentile; fall back to the
            # literature-typical fixed threshold, clearly recorded as such.
            self.baseline_open_ear = self.fallback_threshold / self.closure_ratio
            self.threshold = self.fallback_threshold
            return
        self.baseline_open_ear = float(np.percentile(self._samples, self.calib_percentile))
        self.threshold = self.baseline_open_ear * self.closure_ratio

    def is_closed(self, ear: float) -> bool | None:
        """True/False once calibrated; `None` while still calibrating (the
        caller should not yet make closure decisions)."""
        if not self.is_calibrated:
            return None
        return ear < self.threshold

    def progress(self) -> float:
        """Calibration progress in [0, 1] for UI display."""
        return min(1.0, self._frames_seen / self._n_calib_frames)
