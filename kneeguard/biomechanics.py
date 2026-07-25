"""Knee biomechanics computed from pose landmarks.

Everything in this module is pure geometry on landmark coordinates — no
MediaPipe imports — so the metrics can be unit-tested against synthetic poses.

Landmark indices follow the MediaPipe Pose 33-point topology:

    11 / 12  shoulders (left / right)
    23 / 24  hips
    25 / 26  knees
    27 / 28  ankles
    29 / 30  heels
    31 / 32  foot index (toes)

Odd indices are the subject's LEFT side, even indices the subject's RIGHT side
(anatomical naming, not image-space naming).

Clinical background
-------------------
* **Knee valgus / FPPA** — the frontal-plane projection angle is the standard
  2D screen for dynamic knee valgus. The knee drifting *medially* relative to
  the hip→ankle line loads the ACL in the exact combination (valgus + internal
  rotation + low flexion) that produces non-contact tears. >15 deg is the
  commonly cited high-risk threshold and is what this app flags.
* **KASR** — knee-to-ankle separation ratio from the drop-vertical-jump screen.
  Knees tracking inside the ankles (ratio < ~0.9) is the same collapse pattern
  seen from a distance metric instead of an angle.
* **Landing knee flexion** — stiff, extended landings (< ~45 deg of flexion)
  transmit ground reaction force through the joint rather than the posterior
  chain, spiking anterior tibial shear on the ACL.
* **Lateral trunk lean** — shifts the centre of mass over the stance limb and
  increases the knee abduction moment.
* **Deep flexion under load** — relevant to PCL: a hard landing or fall onto a
  sharply flexed knee drives the tibia posteriorly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --- Landmark indices -------------------------------------------------------

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

REQUIRED_LANDMARKS = (
    L_SHOULDER,
    R_SHOULDER,
    L_HIP,
    R_HIP,
    L_KNEE,
    R_KNEE,
    L_ANKLE,
    R_ANKLE,
)

# --- Clinical thresholds ----------------------------------------------------

VALGUS_ELEVATED_DEG = 8.0
VALGUS_HIGH_DEG = 15.0
KASR_COLLAPSE = 0.90
STIFF_LANDING_FLEXION_DEG = 45.0
DEEP_FLEXION_DEG = 110.0

# Landing depth can only be judged if the athlete actually loaded the leg. A
# person standing upright shows near-zero knee flexion, which is not a stiff
# landing — it is no landing at all, and scoring it as one would penalise every
# athlete who uploaded a standing photo.
MIN_LOADING_FLEXION_DEG = 25.0
TRUNK_LEAN_HIGH_DEG = 12.0
ASYMMETRY_HIGH_DEG = 8.0
MIN_VISIBILITY = 0.5

# Frontal-plane measures (valgus/FPPA, KASR, trunk lean) are only valid when
# the athlete is filmed roughly front-on. Below this score the camera angle is
# too oblique and those measures are suppressed rather than reported as fact.
FRONTAL_VIEW_MIN = 0.45


@dataclass(frozen=True)
class Landmark:
    """A single normalised pose landmark."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class LegMetrics:
    """Per-limb frontal- and sagittal-plane measures."""

    side: str
    valgus_angle_deg: float
    knee_flexion_deg: float
    visibility: float

    @property
    def valgus_flag(self) -> str:
        if self.valgus_angle_deg >= VALGUS_HIGH_DEG:
            return "high"
        if self.valgus_angle_deg >= VALGUS_ELEVATED_DEG:
            return "elevated"
        return "normal"


@dataclass
class FrameMetrics:
    """All biomechanical measures extracted from a single frame."""

    frame_index: int = 0
    timestamp_s: float = 0.0
    left: LegMetrics | None = None
    right: LegMetrics | None = None
    kasr: float | None = None
    trunk_lean_deg: float | None = None
    hip_height: float = 0.0
    confidence: float = 0.0
    frontal_score: float = 0.0

    @property
    def is_frontal(self) -> bool:
        return self.frontal_score >= FRONTAL_VIEW_MIN

    @property
    def peak_valgus_deg(self) -> float:
        values = [leg.valgus_angle_deg for leg in (self.left, self.right) if leg]
        return max(values) if values else 0.0

    @property
    def worst_side(self) -> str | None:
        legs = [leg for leg in (self.left, self.right) if leg]
        if not legs:
            return None
        return max(legs, key=lambda leg: leg.valgus_angle_deg).side

    @property
    def min_knee_flexion_deg(self) -> float:
        values = [leg.knee_flexion_deg for leg in (self.left, self.right) if leg]
        return min(values) if values else 0.0

    @property
    def max_knee_flexion_deg(self) -> float:
        values = [leg.knee_flexion_deg for leg in (self.left, self.right) if leg]
        return max(values) if values else 0.0

    @property
    def valgus_asymmetry_deg(self) -> float:
        if not (self.left and self.right):
            return 0.0
        return abs(self.left.valgus_angle_deg - self.right.valgus_angle_deg)


@dataclass
class BiomechanicsReport:
    """Aggregated, risk-engine-ready summary of a scan."""

    source: str  # "image" | "video"
    frames_analysed: int
    peak_valgus_deg: float
    valgus_side: str | None
    landing_flexion_deg: float
    max_flexion_deg: float
    kasr: float | None
    trunk_lean_deg: float | None
    asymmetry_deg: float
    confidence: float
    frontal_score: float = 0.0
    landing_frame_index: int = 0
    landing_timestamp_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    overlay_image_b64: str | None = None

    @property
    def frontal_view(self) -> bool:
        """Whether frontal-plane measures from this scan can be trusted."""
        return self.frontal_score >= FRONTAL_VIEW_MIN

    @property
    def valgus_flag(self) -> str:
        if not self.frontal_view:
            return "unmeasured"
        if self.peak_valgus_deg >= VALGUS_HIGH_DEG:
            return "high"
        if self.peak_valgus_deg >= VALGUS_ELEVATED_DEG:
            return "elevated"
        return "normal"

    @property
    def landing_assessed(self) -> bool:
        """Whether the athlete visibly loaded the leg enough to judge landing depth."""
        return self.max_flexion_deg >= MIN_LOADING_FLEXION_DEG

    @property
    def stiff_landing(self) -> bool:
        return (
            self.landing_assessed
            and 0 < self.landing_flexion_deg < STIFF_LANDING_FLEXION_DEG
        )

    @property
    def deep_flexion_loading(self) -> bool:
        return self.max_flexion_deg >= DEEP_FLEXION_DEG

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "frames_analysed": self.frames_analysed,
            "peak_valgus_deg": round(self.peak_valgus_deg, 1),
            "valgus_side": self.valgus_side,
            "valgus_flag": self.valgus_flag,
            "landing_flexion_deg": round(self.landing_flexion_deg, 1),
            "max_flexion_deg": round(self.max_flexion_deg, 1),
            "landing_assessed": self.landing_assessed,
            "stiff_landing": self.stiff_landing,
            "deep_flexion_loading": self.deep_flexion_loading,
            "kasr": round(self.kasr, 2) if self.kasr is not None else None,
            "trunk_lean_deg": (
                round(self.trunk_lean_deg, 1) if self.trunk_lean_deg is not None else None
            ),
            "asymmetry_deg": round(self.asymmetry_deg, 1),
            "confidence": round(self.confidence, 2),
            "frontal_score": round(self.frontal_score, 2),
            "frontal_view": self.frontal_view,
            "landing_frame_index": self.landing_frame_index,
            "landing_timestamp_s": round(self.landing_timestamp_s, 2),
            "notes": self.notes,
            "thresholds": {
                "valgus_elevated_deg": VALGUS_ELEVATED_DEG,
                "valgus_high_deg": VALGUS_HIGH_DEG,
                "kasr_collapse": KASR_COLLAPSE,
                "stiff_landing_flexion_deg": STIFF_LANDING_FLEXION_DEG,
            },
        }


# --- Geometry helpers -------------------------------------------------------


def angle_between(a: Sequence[float], vertex: Sequence[float], c: Sequence[float]) -> float:
    """Interior angle at ``vertex`` formed by ``a`` and ``c``, in degrees."""
    v1 = tuple(ai - vi for ai, vi in zip(a, vertex))
    v2 = tuple(ci - vi for ci, vi in zip(c, vertex))
    n1 = math.sqrt(sum(component * component for component in v1))
    n2 = math.sqrt(sum(component * component for component in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(v1, v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def signed_valgus_angle(
    hip: Landmark,
    knee: Landmark,
    ankle: Landmark,
    midline_x: float,
) -> float:
    """Frontal-plane projection angle, signed so positive means *valgus*.

    The magnitude is the deviation of the hip-knee-ankle angle from a straight
    line in the image plane. The sign comes from whether the knee sits on the
    midline side of the hip→ankle line (valgus, knee caving in) or the outside
    (varus, knee bowing out).
    """
    deviation = 180.0 - angle_between((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))

    # Perpendicular offset of the knee from the hip->ankle line (2D cross product).
    dx, dy = ankle.x - hip.x, ankle.y - hip.y
    cross = dx * (knee.y - hip.y) - dy * (knee.x - hip.x)
    if abs(cross) < 1e-9:
        return 0.0

    # Where does the midline sit relative to that same line? If the knee falls
    # on the same side as the midline, the knee has collapsed inward.
    midline_cross = dx * (knee.y - hip.y) - dy * (midline_x - hip.x)
    sign = 1.0 if (cross > 0) == (midline_cross > 0) else -1.0
    return sign * deviation


def knee_flexion_angle(hip: Landmark, knee: Landmark, ankle: Landmark) -> float:
    """Degrees of knee flexion (0 = fully extended straight leg)."""
    return 180.0 - angle_between(
        (hip.x, hip.y, hip.z), (knee.x, knee.y, knee.z), (ankle.x, ankle.y, ankle.z)
    )


def knee_to_ankle_separation_ratio(
    l_knee: Landmark, r_knee: Landmark, l_ankle: Landmark, r_ankle: Landmark
) -> float | None:
    """KASR: horizontal knee separation divided by ankle separation.

    Values below ~0.9 mean the knees are tracking inside the ankles — the
    classic drop-jump valgus collapse signature.
    """
    ankle_sep = abs(l_ankle.x - r_ankle.x)
    if ankle_sep < 1e-6:
        return None
    return abs(l_knee.x - r_knee.x) / ankle_sep


def frontal_view_score(
    l_shoulder: Landmark,
    r_shoulder: Landmark,
    l_hip: Landmark,
    r_hip: Landmark,
) -> float:
    """How front-on the athlete is, from 0 (pure side view) to 1 (square on).

    Shoulder and hip width collapse toward zero as the subject rotates toward
    the camera's sagittal plane, while torso length stays roughly constant.
    The width-to-torso ratio is therefore a cheap, scale-free view estimate.
    """
    shoulder_width = abs(l_shoulder.x - r_shoulder.x)
    hip_width = abs(l_hip.x - r_hip.x)
    torso = math.hypot(
        (l_shoulder.x + r_shoulder.x) / 2 - (l_hip.x + r_hip.x) / 2,
        (l_shoulder.y + r_shoulder.y) / 2 - (l_hip.y + r_hip.y) / 2,
    )
    if torso < 1e-6:
        return 0.0
    ratio = (shoulder_width + hip_width) / (2 * torso)
    # A square-on standing athlete lands around 0.5-0.7; treat 0.7 as fully frontal.
    return max(0.0, min(1.0, ratio / 0.7))


def trunk_lean_angle(
    l_shoulder: Landmark, r_shoulder: Landmark, l_hip: Landmark, r_hip: Landmark
) -> float:
    """Lateral trunk lean from vertical, in degrees."""
    shoulder_mid_x = (l_shoulder.x + r_shoulder.x) / 2
    shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_mid_x = (l_hip.x + r_hip.x) / 2
    hip_mid_y = (l_hip.y + r_hip.y) / 2
    dx = shoulder_mid_x - hip_mid_x
    dy = hip_mid_y - shoulder_mid_y  # image y grows downward
    if abs(dy) < 1e-9:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


# --- Frame-level extraction -------------------------------------------------


def _leg_metrics(
    side: str,
    hip: Landmark,
    knee: Landmark,
    ankle: Landmark,
    midline_x: float,
) -> LegMetrics | None:
    visibility = min(hip.visibility, knee.visibility, ankle.visibility)
    if visibility < MIN_VISIBILITY:
        return None
    return LegMetrics(
        side=side,
        valgus_angle_deg=max(0.0, signed_valgus_angle(hip, knee, ankle, midline_x)),
        knee_flexion_deg=knee_flexion_angle(hip, knee, ankle),
        visibility=visibility,
    )


def frame_metrics(
    landmarks: Sequence[Landmark],
    frame_index: int = 0,
    timestamp_s: float = 0.0,
) -> FrameMetrics:
    """Compute every per-frame measure from one set of pose landmarks."""
    if len(landmarks) < 33:
        raise ValueError(f"expected 33 pose landmarks, got {len(landmarks)}")

    l_hip, r_hip = landmarks[L_HIP], landmarks[R_HIP]
    l_knee, r_knee = landmarks[L_KNEE], landmarks[R_KNEE]
    l_ankle, r_ankle = landmarks[L_ANKLE], landmarks[R_ANKLE]
    l_shoulder, r_shoulder = landmarks[L_SHOULDER], landmarks[R_SHOULDER]

    midline_x = (l_hip.x + r_hip.x) / 2
    frontal = frontal_view_score(l_shoulder, r_shoulder, l_hip, r_hip)

    left = _leg_metrics("left", l_hip, l_knee, l_ankle, midline_x)
    right = _leg_metrics("right", r_hip, r_knee, r_ankle, midline_x)

    # KASR and trunk lean are frontal-plane measures — from an oblique or side
    # view they are geometrically meaningless, so report nothing rather than a
    # confidently wrong number.
    kasr = None
    if left and right and frontal >= FRONTAL_VIEW_MIN:
        kasr = knee_to_ankle_separation_ratio(l_knee, r_knee, l_ankle, r_ankle)

    trunk_lean = None
    if (
        min(l_shoulder.visibility, r_shoulder.visibility) >= MIN_VISIBILITY
        and frontal >= FRONTAL_VIEW_MIN
    ):
        trunk_lean = trunk_lean_angle(l_shoulder, r_shoulder, l_hip, r_hip)

    confidence = sum(landmarks[i].visibility for i in REQUIRED_LANDMARKS) / len(
        REQUIRED_LANDMARKS
    )

    return FrameMetrics(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        left=left,
        right=right,
        kasr=kasr,
        trunk_lean_deg=trunk_lean,
        hip_height=(l_hip.y + r_hip.y) / 2,
        confidence=confidence,
        frontal_score=frontal,
    )


# --- Sequence aggregation ---------------------------------------------------


def find_landing_frame(frames: Sequence[FrameMetrics]) -> int:
    """Index of the frame that best represents ground contact / deepest landing.

    A drop-jump or squat clip descends to a lowest hip position; the landing
    itself is the moment of greatest knee flexion in the neighbourhood of that
    trough, which is where valgus collapse peaks.
    """
    if not frames:
        return 0
    if len(frames) == 1:
        return 0

    # Image-space y grows downward, so the largest hip_height is the lowest hip.
    lowest = max(range(len(frames)), key=lambda i: frames[i].hip_height)
    window = max(2, len(frames) // 10)
    lo = max(0, lowest - window)
    hi = min(len(frames), lowest + window + 1)
    candidates = range(lo, hi)
    return max(candidates, key=lambda i: frames[i].max_knee_flexion_deg)


def summarise(
    frames: Iterable[FrameMetrics],
    source: str,
) -> BiomechanicsReport:
    """Reduce a sequence of frame measurements into a single scan report."""
    frames = [f for f in frames if f.left or f.right]
    if not frames:
        return BiomechanicsReport(
            source=source,
            frames_analysed=0,
            peak_valgus_deg=0.0,
            valgus_side=None,
            landing_flexion_deg=0.0,
            max_flexion_deg=0.0,
            kasr=None,
            trunk_lean_deg=None,
            asymmetry_deg=0.0,
            confidence=0.0,
            notes=["No usable pose detected — no biomechanical adjustment applied."],
        )

    landing_idx = find_landing_frame(frames)
    landing = frames[landing_idx]

    frontal_score = max(f.frontal_score for f in frames)
    frontal_frames = [f for f in frames if f.is_frontal]

    kasr_values = [f.kasr for f in frames if f.kasr is not None]
    lean_values = [f.trunk_lean_deg for f in frames if f.trunk_lean_deg is not None]

    notes: list[str] = []

    if not frontal_frames:
        # Sagittal / oblique footage: flexion is still readable, valgus is not.
        return BiomechanicsReport(
            source=source,
            frames_analysed=len(frames),
            peak_valgus_deg=0.0,
            valgus_side=None,
            landing_flexion_deg=landing.max_knee_flexion_deg,
            max_flexion_deg=max(f.max_knee_flexion_deg for f in frames),
            kasr=None,
            trunk_lean_deg=None,
            asymmetry_deg=0.0,
            confidence=sum(f.confidence for f in frames) / len(frames),
            frontal_score=frontal_score,
            landing_frame_index=landing.frame_index,
            landing_timestamp_s=landing.timestamp_s,
            notes=[
                "Camera angle is too side-on to measure knee valgus. Knee flexion "
                f"was still read ({landing.max_knee_flexion_deg:.0f}deg at landing); "
                "re-film front-on, full body in frame, for a valgus assessment.",
            ],
        )

    peak_frame = max(frontal_frames, key=lambda f: f.peak_valgus_deg)
    peak_valgus = peak_frame.peak_valgus_deg
    if peak_valgus >= VALGUS_HIGH_DEG:
        notes.append(
            f"Knee valgus of {peak_valgus:.1f}deg on the {peak_frame.worst_side} leg exceeds the "
            f"{VALGUS_HIGH_DEG:.0f}deg inward-collapse threshold."
        )
    elif peak_valgus >= VALGUS_ELEVATED_DEG:
        notes.append(
            f"Mild inward knee travel ({peak_valgus:.1f}deg) on the {peak_frame.worst_side} leg."
        )
    else:
        notes.append(f"Knee tracking stayed within normal limits ({peak_valgus:.1f}deg).")

    landing_flexion = landing.max_knee_flexion_deg
    deepest_flexion = max(f.max_knee_flexion_deg for f in frames)
    if deepest_flexion < MIN_LOADING_FLEXION_DEG:
        notes.append(
            "The athlete never bent the knee far enough to judge landing depth — "
            "capture the moment of ground contact, not a standing pose."
        )
    elif 0 < landing_flexion < STIFF_LANDING_FLEXION_DEG:
        notes.append(
            f"Stiff landing: only {landing_flexion:.0f}deg of knee flexion absorbing impact."
        )

    min_kasr = min(kasr_values) if kasr_values else None
    if min_kasr is not None and min_kasr < KASR_COLLAPSE:
        notes.append(
            f"Knees tracked inside the ankles (KASR {min_kasr:.2f}) — medial collapse pattern."
        )

    mean_lean = max(lean_values) if lean_values else None
    if mean_lean is not None and mean_lean >= TRUNK_LEAN_HIGH_DEG:
        notes.append(
            f"Lateral trunk lean of {mean_lean:.0f}deg increases the knee abduction moment."
        )

    asymmetry = peak_frame.valgus_asymmetry_deg
    if asymmetry >= ASYMMETRY_HIGH_DEG:
        notes.append(f"Left/right valgus asymmetry of {asymmetry:.1f}deg.")

    return BiomechanicsReport(
        source=source,
        frames_analysed=len(frames),
        peak_valgus_deg=peak_valgus,
        valgus_side=peak_frame.worst_side,
        landing_flexion_deg=landing_flexion,
        max_flexion_deg=max(f.max_knee_flexion_deg for f in frames),
        kasr=min_kasr,
        trunk_lean_deg=mean_lean,
        asymmetry_deg=asymmetry,
        confidence=sum(f.confidence for f in frames) / len(frames),
        frontal_score=frontal_score,
        landing_frame_index=landing.frame_index,
        landing_timestamp_s=landing.timestamp_s,
        notes=notes,
    )
