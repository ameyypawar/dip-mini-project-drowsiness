"""Week-7 live inference: face + eye localization for full-face frames
(webcam / recorded video), with two selectable backends.

This module answers "where is the face and where are the eyes" for a
*full-face* frame -- it is not used on the MRL eye-crop dataset (Weeks
1-6), which has no face to localize in the first place.

Backends
--------
"mediapipe" (primary): `mediapipe.tasks.python.vision.FaceLandmarker`
    (Tasks API -- MediaPipe 1.0.0 removed the legacy `mp.solutions.face_mesh`
    graph, so this is the only face-landmark path available). Gives 478
    landmarks (468 mesh + 10 iris points), from which face box, eye boxes,
    and the 6-point-per-eye / 4-point-mouth EAR/MAR landmarks (src/ear.py)
    are all derived. Landmark indices are documented and were verified in
    src/ear.py against real detector output on a face photo (see
    docs/week7_live_inference.md) -- not trusted from a tutorial blindly.

"haar" (fallback): vendored Haar cascades. No landmarks, so only face box
    + eye boxes are available -- EAR/MAR are not computable from this
    backend, only the Week-6 eye-state classifier can run (on the eye
    crops this backend still extracts).

Trap avoided (see project brief): OpenCV 5.0.0 ships an *empty*
`cv2/data/haarcascades/` directory -- `cv2.data.haarcascades + '...'`
silently returns a classifier that loads but detects nothing. Cascades
here are loaded from the vendored `assets/cascades/*.xml` paths and
`.empty()` is asserted at construction time with a clear error naming the
path, so this failure mode cannot happen silently again.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CASCADE_DIR = REPO_ROOT / "assets" / "cascades"
FACE_CASCADE_PATH = CASCADE_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = CASCADE_DIR / "haarcascade_eye.xml"
FACE_LANDMARKER_PATH = REPO_ROOT / "assets" / "face_landmarker.task"

Box = tuple[int, int, int, int]  # (x, y, w, h) in pixel coords


@dataclass
class DetectionResult:
    """One frame's detection output, backend-agnostic.

    `landmarks_px` is only populated by the "mediapipe" backend (list of
    478 (x, y) pixel-coord tuples); it is `None` for "haar", which has no
    landmarks and therefore cannot feed src.ear (EAR/MAR require them).
    """

    backend: str
    face_box: Box | None
    left_eye_box: Box | None
    right_eye_box: Box | None
    landmarks_px: list[tuple[float, float]] | None = None


def _clip_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Box:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return (x0, y0, x1 - x0, y1 - y0)


def crop_gray(gray: np.ndarray, box: Box) -> np.ndarray:
    """Grayscale crop for `box`=(x, y, w, h), clipped to `gray`'s bounds."""
    h, w = gray.shape[:2]
    x, y, bw, bh = box
    x, y, bw, bh = _clip_box(x, y, x + bw, y + bh, w, h)
    return gray[y:y + bh, x:x + bw]


# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker eye/mouth landmark index sets.
# 6-point EAR subsets and mouth indices are defined + verified in src/ear.py
# (single source of truth); the wider contour lists below are only used
# here to get a tight *bounding box* around each eye for cropping (more
# points than the 6-point EAR subset -> tighter box covering upper+lower
# lid), not for the EAR ratio itself.
# ---------------------------------------------------------------------------

LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]


def _eye_box_from_landmarks(
    landmarks_px: list[tuple[float, float]], contour: list[int], img_w: int, img_h: int,
    margin_x_frac: float = 0.3, margin_y_frac: float = 0.6,
) -> Box:
    pts = np.array([landmarks_px[i] for i in contour])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    mx = (x1 - x0) * margin_x_frac
    my = (y1 - y0) * margin_y_frac
    return _clip_box(int(x0 - mx), int(y0 - my), int(x1 + mx), int(y1 + my), img_w, img_h)


class FaceEyeDetector:
    """Face + eye localizer with a "mediapipe" (primary) or "haar"
    (fallback) backend, sharing one `detect(frame_bgr) -> DetectionResult`
    interface and one `get_eye_crops(frame_bgr, result) -> (left, right)`
    grayscale-crop interface regardless of backend.

    `running_mode`: "video" (default) uses `detect_for_video` with a
    monotonically increasing `timestamp_ms` (required for temporal
    smoothing in MediaPipe's graph -- pass frame-index * (1000/fps) from
    the caller). "image" uses stateless per-frame `detect`, only correct
    for single unrelated images (see notebook's single-face-photo cell).
    """

    def __init__(self, backend: str = "mediapipe", running_mode: str = "video", num_faces: int = 1):
        if backend not in ("mediapipe", "haar"):
            raise ValueError(f"unknown backend {backend!r}, expected 'mediapipe' or 'haar'")
        self.backend = backend
        self.running_mode = running_mode
        self._mp_landmarker = None
        self._face_cascade: cv2.CascadeClassifier | None = None
        self._eye_cascade: cv2.CascadeClassifier | None = None

        if backend == "mediapipe":
            self._init_mediapipe(running_mode, num_faces)
        else:
            self._init_haar()

    def _init_mediapipe(self, running_mode: str, num_faces: int) -> None:
        from mediapipe.tasks.python import vision, BaseOptions

        if not FACE_LANDMARKER_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at {FACE_LANDMARKER_PATH} "
                "-- expected the vendored asset."
            )
        mode = vision.RunningMode.VIDEO if running_mode == "video" else vision.RunningMode.IMAGE
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            running_mode=mode,
            num_faces=num_faces,
        )
        self._mp_landmarker = vision.FaceLandmarker.create_from_options(opts)

    def _init_haar(self) -> None:
        face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        if face_cascade.empty():
            raise RuntimeError(
                f"Face cascade failed to load from vendored path {FACE_CASCADE_PATH}. "
                "Do NOT fall back to cv2.data.haarcascades -- OpenCV 5.0.0 ships that "
                "directory empty and CascadeClassifier silently 'succeeds' with a "
                "classifier that detects nothing."
            )
        eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
        if eye_cascade.empty():
            raise RuntimeError(
                f"Eye cascade failed to load from vendored path {EYE_CASCADE_PATH}. "
                "Do NOT fall back to cv2.data.haarcascades -- see face-cascade error "
                "for why."
            )
        self._face_cascade = face_cascade
        self._eye_cascade = eye_cascade

    # -- detection -----------------------------------------------------

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> DetectionResult:
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame_bgr, timestamp_ms)
        return self._detect_haar(frame_bgr)

    def _detect_mediapipe(self, frame_bgr: np.ndarray, timestamp_ms: int | None) -> DetectionResult:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "video":
            if timestamp_ms is None:
                raise ValueError("running_mode='video' requires timestamp_ms")
            result = self._mp_landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self._mp_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return DetectionResult(backend="mediapipe", face_box=None, left_eye_box=None, right_eye_box=None)

        lm = result.face_landmarks[0]
        landmarks_px = [(pt.x * w, pt.y * h) for pt in lm]

        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        face_box = _clip_box(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), w, h)

        left_box = _eye_box_from_landmarks(landmarks_px, LEFT_EYE_CONTOUR, w, h)
        right_box = _eye_box_from_landmarks(landmarks_px, RIGHT_EYE_CONTOUR, w, h)

        return DetectionResult(
            backend="mediapipe",
            face_box=face_box,
            left_eye_box=left_box,
            right_eye_box=right_box,
            landmarks_px=landmarks_px,
        )

    def _detect_haar(self, frame_bgr: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return DetectionResult(backend="haar", face_box=None, left_eye_box=None, right_eye_box=None)

        # largest detected face
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        face_box = _clip_box(fx, fy, fx + fw, fy + fh, w, h)

        face_roi = gray[fy:fy + fh, fx:fx + fw]
        eyes = self._eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        # keep at most 2 eyes, sorted left-to-right in face-local x
        eyes = sorted(eyes, key=lambda b: b[0])[:2]

        left_box = right_box = None
        if len(eyes) >= 1:
            ex, ey, ew, eh = eyes[0]
            left_box = _clip_box(fx + ex, fy + ey, fx + ex + ew, fy + ey + eh, w, h)
        if len(eyes) >= 2:
            ex, ey, ew, eh = eyes[1]
            right_box = _clip_box(fx + ex, fy + ey, fx + ex + ew, fy + ey + eh, w, h)

        return DetectionResult(backend="haar", face_box=face_box, left_eye_box=left_box, right_eye_box=right_box)

    # -- eye ROI extraction for the Week-6 classifier -------------------

    def get_eye_crops(self, frame_bgr: np.ndarray, result: DetectionResult) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Grayscale left/right eye crops from `frame_bgr` per `result`'s
        boxes, suitable to feed straight into
        `src.classify.predict_eye_state` (which applies
        `preprocess_pipeline` internally -- do not preprocess again here).
        `None` for a side whose box was not found this frame.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        left = crop_gray(gray, result.left_eye_box) if result.left_eye_box else None
        right = crop_gray(gray, result.right_eye_box) if result.right_eye_box else None
        if left is not None and (left.shape[0] < 4 or left.shape[1] < 4):
            left = None
        if right is not None and (right.shape[0] < 4 or right.shape[1] < 4):
            right = None
        return left, right

    def close(self) -> None:
        if self._mp_landmarker is not None:
            self._mp_landmarker.close()
