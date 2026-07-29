"""Week-7 PERCLOS: PERcentage of eye CLOSure over a rolling time window.

PERCLOS is the measure identified by the Carnegie Mellon Research
Institute driving-simulator studies (Wierwille et al. 1994) and the
FHWA-sponsored validation (Dinges & Grace, 1998, "PERCLOS: A Valid
Psychophysiological Measure of Alertness As Assessed by Psychomotor
Vigilance") as the best-performing eye-closure-based drowsiness
predictor among the measures they compared -- it is the closest thing
this field has to a standard measure, which is why it is the headline
statistic here rather than a bespoke one.

This module is deliberately agnostic to *how* "eye closed" is decided
per frame -- `PERCLOSTracker.update(timestamp_s, is_closed)` takes a
plain boolean. That lets the exact same tracker class run on either
signal source this project has:
  - the Week-6 classifier's per-frame open/closed label
    (`src.classify.predict_eye_state`), or
  - `src.ear`'s EAR-below-adaptive-threshold decision
    (`EARCalibrator.is_closed`)
so the two can be run side-by-side over the same sequence and compared
(see notebooks/07_live_inference.ipynb) -- this module does not pick a
winner between them.

Window and thresholds chosen (see docs/week7_live_inference.md for the
full justification and the synthetic-ground-truth validation numbers):
  - `window_seconds=10.0`: the original CMU/FHWA studies used much longer
    windows (1-3 minutes) over multi-hour driving-simulator sessions.
    This project's synthetic demo sequences run ~20-30s total, so a
    1-3 minute window would rarely fill even once; 10s is a demo-scale
    adaptation, not a claim that 10s is the validated research window --
    stated explicitly as a limitation.
  - `enter_threshold=0.15` / `exit_threshold=0.10`: 15% closure is a
    widely-cited "drowsy" cutoff for the P80-style PERCLOS measure in the
    drowsiness-detection literature; a lower *exit* threshold (10%,
    hysteresis) is this project's own addition on top of that, not from
    the cited studies, added specifically so the alert does not flicker
    on/off frame-to-frame right at the boundary.
  - `min_enter_seconds` / `min_exit_seconds`: minimum dwell time above/
    below threshold before the alert flips, a second (temporal) layer of
    debouncing on top of the two-threshold hysteresis.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class PERCLOSConfig:
    window_seconds: float = 10.0
    enter_threshold: float = 0.15
    exit_threshold: float = 0.10
    min_enter_seconds: float = 1.0
    min_exit_seconds: float = 2.0


@dataclass
class PERCLOSReading:
    timestamp_s: float
    perclos: float
    n_frames_in_window: float
    alert: bool


class PERCLOSTracker:
    """Rolling-window PERCLOS + hysteresis alert state machine.

    Frames are (timestamp_s, is_closed) pairs; `timestamp_s` must be
    non-decreasing across calls (video/webcam frame times or a synthetic
    sequence's per-frame `t = frame_idx / fps`). The window is defined on
    *time*, not frame count, so it behaves the same regardless of the
    source fps.
    """

    def __init__(self, config: PERCLOSConfig = PERCLOSConfig()):
        self.config = config
        self._buffer: deque[tuple[float, bool]] = deque()
        self.alert: bool = False
        self._enter_streak_start: float | None = None
        self._exit_streak_start: float | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self.alert = False
        self._enter_streak_start = None
        self._exit_streak_start = None

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.config.window_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def _perclos(self) -> float:
        if not self._buffer:
            return 0.0
        n_closed = sum(1 for _, closed in self._buffer if closed)
        return n_closed / len(self._buffer)

    def current_perclos(self) -> float:
        """Public accessor for the current window's PERCLOS value (e.g.
        for a final end-of-run summary print, without reaching into the
        private buffer)."""
        return self._perclos()

    def _update_alert(self, now: float, perclos: float) -> None:
        cfg = self.config
        if not self.alert:
            if perclos >= cfg.enter_threshold:
                if self._enter_streak_start is None:
                    self._enter_streak_start = now
                elif now - self._enter_streak_start >= cfg.min_enter_seconds:
                    self.alert = True
                    self._enter_streak_start = None
            else:
                self._enter_streak_start = None
        else:
            if perclos <= cfg.exit_threshold:
                if self._exit_streak_start is None:
                    self._exit_streak_start = now
                elif now - self._exit_streak_start >= cfg.min_exit_seconds:
                    self.alert = False
                    self._exit_streak_start = None
            else:
                self._exit_streak_start = None

    def update(self, timestamp_s: float, is_closed: bool) -> PERCLOSReading:
        self._buffer.append((timestamp_s, bool(is_closed)))
        self._evict_old(timestamp_s)
        perclos = self._perclos()
        self._update_alert(timestamp_s, perclos)
        return PERCLOSReading(
            timestamp_s=timestamp_s,
            perclos=perclos,
            n_frames_in_window=len(self._buffer),
            alert=self.alert,
        )


def run_perclos_over_sequence(
    timestamps_s: list[float], is_closed: list[bool], config: PERCLOSConfig = PERCLOSConfig(),
) -> list[PERCLOSReading]:
    """Convenience batch entry point: replay a full (timestamps, is_closed)
    sequence through a fresh `PERCLOSTracker` and return every reading --
    used by the notebook to build the PERCLOS timeline plot and by
    `scripts/run_demo.py`'s offline-video mode.
    """
    tracker = PERCLOSTracker(config)
    return [tracker.update(t, c) for t, c in zip(timestamps_s, is_closed)]
