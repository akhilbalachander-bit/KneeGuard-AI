"""MediaPipe pose extraction for KneeGuard AI.

Wraps the MediaPipe Tasks ``PoseLandmarker`` so the rest of the app only ever
sees plain :class:`~kneeguard.biomechanics.Landmark` objects and
:class:`~kneeguard.biomechanics.BiomechanicsReport` summaries.

Handles both inputs the product accepts:

* **Photo** — a single frame at the bottom of a squat or the moment of landing.
* **Video** — a short drop-jump / squat clip. Frames are sampled, the landing
  frame is located from the hip-descent trough, and peak valgus is reported.

An annotated overlay of the landing frame is returned as a base64 PNG so the
dashboard can show the athlete exactly what was measured.
"""

from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from . import config
from .biomechanics import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    VALGUS_HIGH_DEG,
    BiomechanicsReport,
    FrameMetrics,
    Landmark,
    find_landing_frame,
    frame_metrics,
    summarise,
)

log = logging.getLogger(__name__)

_landmarker = None
_landmarker_lock = threading.Lock()

# Skeleton edges drawn on the overlay (lower body + trunk only — that is what
# the risk model actually uses, and a full skeleton clutters the frame).
_OVERLAY_EDGES = (
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
)


class PoseModelUnavailable(RuntimeError):
    """Raised when the MediaPipe landmarker bundle is missing or unloadable."""


def pose_model_available() -> bool:
    return config.POSE_MODEL_PATH.exists()


def _get_landmarker():
    """Lazily build a single shared IMAGE-mode landmarker.

    MediaPipe landmarkers are not thread-safe, so every ``detect`` call is
    serialised behind ``_landmarker_lock``. A drop-jump clip is ~100 frames, so
    the serialisation cost is irrelevant next to model-load cost.
    """
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    if not config.POSE_MODEL_PATH.exists():
        raise PoseModelUnavailable(
            f"Pose model bundle not found at {config.POSE_MODEL_PATH}. "
            "Run: python scripts/fetch_pose_model.py"
        )

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - depends on install
        raise PoseModelUnavailable(f"mediapipe is not installed: {exc}") from exc

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(config.POSE_MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = vision.PoseLandmarker.create_from_options(options)
    log.info("Loaded MediaPipe pose landmarker from %s", config.POSE_MODEL_PATH)
    return _landmarker


def _detect(bgr_frame: np.ndarray) -> list[Landmark] | None:
    """Run pose detection on one BGR frame; return 33 landmarks or ``None``."""
    import mediapipe as mp

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    landmarker = _get_landmarker()
    with _landmarker_lock:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None
    return [
        Landmark(
            x=float(lm.x),
            y=float(lm.y),
            z=float(lm.z),
            visibility=float(getattr(lm, "visibility", 1.0) or 0.0),
        )
        for lm in result.pose_landmarks[0]
    ]


def _decode_image(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode the uploaded image.")
    return frame


def _annotate(frame: np.ndarray, landmarks: list[Landmark], metrics: FrameMetrics) -> str:
    """Draw the measured skeleton on a frame and return it as a base64 PNG."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # Keep the overlay a sensible size for the dashboard.
    scale = min(1.0, 720 / max(height, width))
    if scale < 1.0:
        canvas = cv2.resize(canvas, (int(width * scale), int(height * scale)))
        height, width = canvas.shape[:2]

    def point(index: int) -> tuple[int, int]:
        lm = landmarks[index]
        return int(lm.x * width), int(lm.y * height)

    for start, end in _OVERLAY_EDGES:
        cv2.line(canvas, point(start), point(end), (210, 210, 210), 2, cv2.LINE_AA)

    for side, leg, hip_i, knee_i, ankle_i in (
        ("left", metrics.left, L_HIP, L_KNEE, L_ANKLE),
        ("right", metrics.right, R_HIP, R_KNEE, R_ANKLE),
    ):
        if leg is None:
            continue
        high = leg.valgus_angle_deg >= VALGUS_HIGH_DEG
        colour = (60, 60, 235) if high else (90, 200, 90)  # BGR
        cv2.line(canvas, point(hip_i), point(knee_i), colour, 4, cv2.LINE_AA)
        cv2.line(canvas, point(knee_i), point(ankle_i), colour, 4, cv2.LINE_AA)
        # Reference line: where the knee *should* track (hip -> ankle).
        cv2.line(canvas, point(hip_i), point(ankle_i), (0, 200, 255), 1, cv2.LINE_AA)

        kx, ky = point(knee_i)
        cv2.circle(canvas, (kx, ky), 7, colour, -1, cv2.LINE_AA)
        label = f"{leg.valgus_angle_deg:.0f}deg"
        offset = 14 if side == "left" else -70
        cv2.putText(
            canvas,
            label,
            (kx + offset, ky - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )

    for index in (L_HIP, R_HIP, L_ANKLE, R_ANKLE):
        cv2.circle(canvas, point(index), 5, (235, 235, 235), -1, cv2.LINE_AA)

    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:  # pragma: no cover - encoding a valid ndarray does not fail
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def analyse_image(data: bytes) -> BiomechanicsReport:
    """Analyse a single photo of a squat / landing position."""
    frame = _decode_image(data)
    landmarks = _detect(frame)
    if landmarks is None:
        report = summarise([], source="image")
        report.notes = [
            "No person detected in the photo. Use a front-facing, full-body shot "
            "with hips, knees and ankles all visible."
        ]
        return report

    metrics = frame_metrics(landmarks, frame_index=0, timestamp_s=0.0)
    report = summarise([metrics], source="image")
    report.overlay_image_b64 = _annotate(frame, landmarks, metrics)
    if report.frames_analysed:
        report.notes.append(
            "Single-frame scan: landing stiffness is estimated from this pose only. "
            "Upload a 2-3 second clip for a full drop-jump assessment."
        )
    return report


def analyse_video(path: Path) -> BiomechanicsReport:
    """Analyse a short jump / squat clip and report the worst landing frame."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not read the uploaded video.")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // config.MAX_VIDEO_FRAMES) if total else 1

        metrics_list: list[FrameMetrics] = []
        frames_by_index: dict[int, tuple[np.ndarray, list[Landmark]]] = {}

        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                landmarks = _detect(frame)
                if landmarks is not None:
                    metrics = frame_metrics(
                        landmarks, frame_index=index, timestamp_s=index / fps
                    )
                    metrics_list.append(metrics)
                    frames_by_index[index] = (frame.copy(), landmarks)
            index += 1
            if len(metrics_list) >= config.MAX_VIDEO_FRAMES:
                break
    finally:
        capture.release()

    usable = [m for m in metrics_list if m.left or m.right]
    if not usable:
        report = summarise([], source="video")
        report.notes = [
            "No person detected in the clip. Film front-on, full body in frame, "
            "with good lighting."
        ]
        return report

    report = summarise(usable, source="video")

    landing = usable[find_landing_frame(usable)]
    overlay_source = frames_by_index.get(landing.frame_index)
    if overlay_source is not None:
        frame, landmarks = overlay_source
        report.overlay_image_b64 = _annotate(frame, landmarks, landing)
    report.notes.append(
        f"Landing detected at {report.landing_timestamp_s:.2f}s "
        f"({report.frames_analysed} frames analysed)."
    )
    return report
