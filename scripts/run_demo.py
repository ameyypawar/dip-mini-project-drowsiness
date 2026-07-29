#!/usr/bin/env python3
"""run_demo.py -- Week 7 live inference demo.

Runs face/eye localization (src.detect), EAR/MAR (src.ear), the Week-6
eye-state classifier (src.classify.predict_eye_state), and rolling
PERCLOS (src.perclos) over a video source, overlaying face/eye boxes,
EAR/MAR, per-frame eye state + confidence, rolling PERCLOS, and an ALERT
banner when the drowsiness threshold trips.

    --source webcam            (NOT run in this environment -- see below)
    --source video --video PATH

This script never opens a webcam in the environment it was authored in
(no camera exists there and cv2.VideoCapture(0) would hang forever
waiting for a device) -- it was authored and tested exclusively against
`--source video --video data/synthetic/sequence.mp4`. The `--source
webcam` code path is implemented and should work unmodified on a machine
with a camera (same detect/EAR/PERCLOS pipeline either way, source is
just a different cv2.VideoCapture argument), but it is untested here --
this is stated in docs/week7_live_inference.md, not silently assumed.

Two detection outcomes per frame:
  1. A face is detected (real webcam/video of a person): eye ROIs come
     from `FaceEyeDetector.get_eye_crops`, EAR/MAR come from landmarks
     (mediapipe backend only), classifier runs on both eye crops.
  2. No face detected: for the "haar" backend this just means "no driver
     in frame". But the synthetic sequence (data/synthetic/sequence.mp4)
     is itself a stream of already-cropped MRL eye images with no
     surrounding face -- a face detector correctly finds nothing there.
     In that case this script falls back to treating the *entire frame*
     as one eye crop and feeds it straight to the classifier, so PERCLOS
     can still be validated against the synthetic ground truth even
     though no face/EAR pipeline applies to eye-crop-only input (this
     mirrors the project's stated constraint: EAR is never computed on
     MRL crops).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.classify import predict_eye_state
from src.detect import FaceEyeDetector
from src.ear import EARCalibrator, mean_eye_aspect_ratio, mouth_aspect_ratio
from src.perclos import PERCLOSConfig, PERCLOSTracker

ALERT_TEXT = "DROWSINESS ALERT"
GREEN = (0, 200, 0)
RED = (0, 0, 220)
YELLOW = (0, 220, 220)
WHITE = (255, 255, 255)


def draw_box(frame, box, color, label=None):
    if box is None:
        return
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def combine_eye_state(left_pred, right_pred) -> tuple[str | None, float]:
    """Combine (label, confidence) predictions from up to two eye crops
    into one frame-level state. Closed wins if either available eye
    reports closed (a driver with one eye closed is not "open" for
    drowsiness purposes); average confidence when both agree, otherwise
    the confidence of whichever crop drove the "closed" call.
    """
    preds = [p for p in (left_pred, right_pred) if p is not None]
    if not preds:
        return None, 0.0
    if any(label == "closed" for label, _ in preds):
        closed_confs = [conf for label, conf in preds if label == "closed"]
        return "closed", float(np.mean(closed_confs))
    return "open", float(np.mean([conf for _, conf in preds]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["webcam", "video"], required=True)
    ap.add_argument("--video", type=Path, default=None, help="required if --source video")
    ap.add_argument("--backend", choices=["mediapipe", "haar"], default="mediapipe")
    ap.add_argument("--fps", type=float, default=None, help="override fps (defaults to source's reported fps, or 10.0 if unavailable)")
    ap.add_argument("--perclos-window", type=float, default=PERCLOSConfig().window_seconds)
    ap.add_argument("--perclos-enter", type=float, default=PERCLOSConfig().enter_threshold)
    ap.add_argument("--perclos-exit", type=float, default=PERCLOSConfig().exit_threshold)
    ap.add_argument("--calib-seconds", type=float, default=3.0, help="EAR calibration window at session start (assume eyes open)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true", help="show a live cv2 window (requires a GUI-capable environment)")
    ap.add_argument("--save-video", type=Path, default=None, help="write annotated frames to this path")
    args = ap.parse_args()

    if args.source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        if args.video is None:
            print("ERROR: --source video requires --video PATH", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(str(args.video))

    if not cap.isOpened():
        print(f"ERROR: could not open source ({args.source})", file=sys.stderr)
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps or (src_fps if src_fps and src_fps > 1 else 10.0)

    detector = FaceEyeDetector(backend=args.backend, running_mode="video")
    ear_calib = EARCalibrator(fps=fps, calib_seconds=args.calib_seconds)
    perclos_cfg = PERCLOSConfig(window_seconds=args.perclos_window, enter_threshold=args.perclos_enter, exit_threshold=args.perclos_exit)
    perclos_clf = PERCLOSTracker(perclos_cfg)
    perclos_ear = PERCLOSTracker(perclos_cfg)

    writer = None
    frame_idx = 0
    t_wall_start = time.perf_counter()
    n_face_found = 0
    n_fallback_eyecrop = 0

    try:
        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            timestamp_s = frame_idx / fps
            timestamp_ms = int(round(timestamp_s * 1000))

            result = detector.detect(frame_bgr, timestamp_ms=timestamp_ms)

            ear_value = None
            mar_value = None
            state_label, state_conf = None, 0.0

            if result.face_box is not None:
                n_face_found += 1
                left_crop, right_crop = detector.get_eye_crops(frame_bgr, result)
                left_pred = predict_eye_state(left_crop) if left_crop is not None else None
                right_pred = predict_eye_state(right_crop) if right_crop is not None else None
                state_label, state_conf = combine_eye_state(left_pred, right_pred)

                if result.landmarks_px is not None:
                    ear_value = mean_eye_aspect_ratio(result.landmarks_px)
                    mar_value = mouth_aspect_ratio(result.landmarks_px)

                draw_box(frame_bgr, result.face_box, GREEN)
                draw_box(frame_bgr, result.left_eye_box, YELLOW, "L")
                draw_box(frame_bgr, result.right_eye_box, YELLOW, "R")
            else:
                # No face -- fall back to treating the whole frame as one
                # eye crop (this is the path exercised by the synthetic
                # eye-crop sequence; see module docstring).
                n_fallback_eyecrop += 1
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                state_label, state_conf = predict_eye_state(gray)

            is_closed_clf = (state_label == "closed") if state_label is not None else False
            reading_clf = perclos_clf.update(timestamp_s, is_closed_clf)

            if ear_value is not None:
                ear_calib.update(ear_value)
                is_closed_ear = ear_calib.is_closed(ear_value)
                if is_closed_ear is not None:
                    reading_ear = perclos_ear.update(timestamp_s, is_closed_ear)
                else:
                    reading_ear = None
            else:
                reading_ear = None

            # -- overlay -----------------------------------------------
            y = 24
            def put(text, color=WHITE):
                nonlocal y
                cv2.putText(frame_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                y += 22

            if state_label is not None:
                put(f"eye state: {state_label} ({state_conf:.2f})", GREEN if state_label == "open" else RED)
            else:
                put("eye state: n/a", WHITE)

            if ear_value is not None:
                calib_note = "" if ear_calib.is_calibrated else f" [calibrating {ear_calib.progress():.0%}]"
                put(f"EAR: {ear_value:.3f}{calib_note}", WHITE)
                put(f"MAR: {mar_value:.3f}", WHITE)
            else:
                put("EAR/MAR: n/a (no landmarks this frame)", WHITE)

            put(f"PERCLOS[classifier]: {reading_clf.perclos:.2%}", WHITE)
            if reading_ear is not None:
                put(f"PERCLOS[EAR]: {reading_ear.perclos:.2%}", WHITE)

            if reading_clf.alert or (reading_ear is not None and reading_ear.alert):
                cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 40), RED, -1)
                cv2.putText(frame_bgr, ALERT_TEXT, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

            if args.save_video is not None:
                if writer is None:
                    args.save_video.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = frame_bgr.shape[:2]
                    writer = cv2.VideoWriter(str(args.save_video), fourcc, fps, (w, h), isColor=True)
                writer.write(frame_bgr)

            if args.display:
                cv2.imshow("Week 7 - Drowsiness Live Demo", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        detector.close()

    wall_elapsed = time.perf_counter() - t_wall_start
    achieved_fps = frame_idx / wall_elapsed if wall_elapsed > 0 else 0.0
    print(f"\nprocessed {frame_idx} frames in {wall_elapsed:.2f}s ({achieved_fps:.1f} fps achieved)")
    print(f"face detected on {n_face_found}/{frame_idx} frames; fallback whole-frame-as-eye-crop on {n_fallback_eyecrop}/{frame_idx}")
    print(f"final PERCLOS[classifier]={perclos_clf.current_perclos():.2%} alert={perclos_clf.alert}")
    if ear_calib.is_calibrated:
        print(f"EAR baseline_open={ear_calib.baseline_open_ear:.3f} threshold={ear_calib.threshold:.3f}")
        print(f"final PERCLOS[EAR]={perclos_ear.current_perclos():.2%} alert={perclos_ear.alert}")
    else:
        print("EAR never calibrated (no face/landmarks detected long enough)")


if __name__ == "__main__":
    main()
