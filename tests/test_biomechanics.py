"""Geometry tests for the knee biomechanics measures.

Poses are constructed by hand so the expected answer is known independently of
MediaPipe. Image coordinates are normalised: x grows to the right, y grows
*downward*, and a front-facing athlete has their left side at larger x.
"""

from __future__ import annotations

import math

import pytest

from kneeguard.biomechanics import (
    FRONTAL_VIEW_MIN,
    KASR_COLLAPSE,
    MIN_LOADING_FLEXION_DEG,
    VALGUS_HIGH_DEG,
    Landmark,
    angle_between,
    find_landing_frame,
    frame_metrics,
    frontal_view_score,
    knee_flexion_angle,
    knee_to_ankle_separation_ratio,
    signed_valgus_angle,
    summarise,
    trunk_lean_angle,
)


def build_pose(
    *,
    knee_offset: float = 0.0,
    knee_y: float = 0.74,
    hip_y: float = 0.52,
    ankle_y: float = 0.94,
    hip_half: float = 0.075,
    ankle_half: float = 0.075,
    shoulder_half: float = 0.105,
    shoulder_y: float = 0.30,
    visibility: float = 1.0,
) -> list[Landmark]:
    """A front-facing pose. ``knee_offset`` moves both knees toward the midline."""
    marks = [Landmark(0.5, 0.5, 0.0, visibility) for _ in range(33)]
    marks[11] = Landmark(0.5 + shoulder_half, shoulder_y, 0.0, visibility)
    marks[12] = Landmark(0.5 - shoulder_half, shoulder_y, 0.0, visibility)
    marks[23] = Landmark(0.5 + hip_half, hip_y, 0.0, visibility)
    marks[24] = Landmark(0.5 - hip_half, hip_y, 0.0, visibility)
    marks[25] = Landmark(0.5 + hip_half - knee_offset, knee_y, 0.0, visibility)
    marks[26] = Landmark(0.5 - hip_half + knee_offset, knee_y, 0.0, visibility)
    marks[27] = Landmark(0.5 + ankle_half, ankle_y, 0.0, visibility)
    marks[28] = Landmark(0.5 - ankle_half, ankle_y, 0.0, visibility)
    return marks


# --- Pure geometry ---------------------------------------------------------


def test_angle_between_right_angle():
    assert angle_between((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)


def test_angle_between_straight_line():
    assert angle_between((0, 1), (0, 0), (0, -1)) == pytest.approx(180.0)


def test_angle_between_degenerate_returns_zero():
    assert angle_between((0, 0), (0, 0), (1, 0)) == 0.0


def test_straight_leg_has_zero_valgus():
    hip = Landmark(0.575, 0.52)
    knee = Landmark(0.575, 0.74)
    ankle = Landmark(0.575, 0.94)
    assert signed_valgus_angle(hip, knee, ankle, midline_x=0.5) == pytest.approx(0.0, abs=0.5)


def test_knee_toward_midline_is_positive_valgus():
    hip = Landmark(0.575, 0.52)
    knee = Landmark(0.545, 0.74)  # pulled toward the 0.5 midline
    ankle = Landmark(0.575, 0.94)
    assert signed_valgus_angle(hip, knee, ankle, midline_x=0.5) > 5.0


def test_knee_away_from_midline_is_negative_varus():
    hip = Landmark(0.575, 0.52)
    knee = Landmark(0.615, 0.74)  # bowed outward, away from midline
    ankle = Landmark(0.575, 0.94)
    assert signed_valgus_angle(hip, knee, ankle, midline_x=0.5) < 0.0


def test_valgus_sign_is_mirrored_for_the_other_leg():
    """The same inward displacement must read as valgus on both legs."""
    left = signed_valgus_angle(
        Landmark(0.575, 0.52), Landmark(0.545, 0.74), Landmark(0.575, 0.94), midline_x=0.5
    )
    right = signed_valgus_angle(
        Landmark(0.425, 0.52), Landmark(0.455, 0.74), Landmark(0.425, 0.94), midline_x=0.5
    )
    assert left > 0 and right > 0
    assert left == pytest.approx(right, abs=0.01)


def test_knee_flexion_extended_leg_is_zero():
    flexion = knee_flexion_angle(
        Landmark(0.5, 0.5), Landmark(0.5, 0.7), Landmark(0.5, 0.9)
    )
    assert flexion == pytest.approx(0.0, abs=0.5)


def test_knee_flexion_right_angle():
    flexion = knee_flexion_angle(
        Landmark(0.5, 0.5), Landmark(0.5, 0.7), Landmark(0.7, 0.7)
    )
    assert flexion == pytest.approx(90.0, abs=0.5)


def test_kasr_knees_over_ankles_is_one():
    ratio = knee_to_ankle_separation_ratio(
        Landmark(0.58, 0.7), Landmark(0.42, 0.7), Landmark(0.58, 0.9), Landmark(0.42, 0.9)
    )
    assert ratio == pytest.approx(1.0)


def test_kasr_knees_inside_ankles_is_below_one():
    ratio = knee_to_ankle_separation_ratio(
        Landmark(0.54, 0.7), Landmark(0.46, 0.7), Landmark(0.58, 0.9), Landmark(0.42, 0.9)
    )
    assert ratio < KASR_COLLAPSE


def test_kasr_none_when_ankles_coincide():
    assert knee_to_ankle_separation_ratio(
        Landmark(0.54, 0.7), Landmark(0.46, 0.7), Landmark(0.5, 0.9), Landmark(0.5, 0.9)
    ) is None


def test_trunk_lean_upright_is_zero():
    lean = trunk_lean_angle(
        Landmark(0.6, 0.3), Landmark(0.4, 0.3), Landmark(0.6, 0.55), Landmark(0.4, 0.55)
    )
    assert lean == pytest.approx(0.0, abs=0.5)


def test_trunk_lean_detects_lateral_shift():
    lean = trunk_lean_angle(
        Landmark(0.70, 0.3), Landmark(0.50, 0.3), Landmark(0.6, 0.55), Landmark(0.4, 0.55)
    )
    assert lean > 15.0


# --- View detection --------------------------------------------------------


def test_frontal_view_score_high_when_square_on():
    score = frontal_view_score(
        Landmark(0.605, 0.30), Landmark(0.395, 0.30),
        Landmark(0.575, 0.52), Landmark(0.425, 0.52),
    )
    assert score >= FRONTAL_VIEW_MIN


def test_frontal_view_score_low_when_side_on():
    """Side-on: shoulders and hips collapse to nearly a single x."""
    score = frontal_view_score(
        Landmark(0.505, 0.30), Landmark(0.495, 0.30),
        Landmark(0.503, 0.52), Landmark(0.497, 0.52),
    )
    assert score < FRONTAL_VIEW_MIN


def test_side_on_footage_reports_no_valgus_rather_than_a_wrong_one():
    side_on = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    side_on[11] = Landmark(0.505, 0.30)
    side_on[12] = Landmark(0.495, 0.30)
    side_on[23] = Landmark(0.503, 0.52)
    side_on[24] = Landmark(0.497, 0.52)
    side_on[25] = Landmark(0.470, 0.74)
    side_on[26] = Landmark(0.530, 0.74)
    side_on[27] = Landmark(0.503, 0.94)
    side_on[28] = Landmark(0.497, 0.94)

    report = summarise([frame_metrics(side_on)], source="image")
    assert report.frontal_view is False
    assert report.valgus_flag == "unmeasured"
    assert report.peak_valgus_deg == 0.0
    assert report.kasr is None


# --- Frame metrics ---------------------------------------------------------


def test_frame_metrics_requires_full_landmark_set():
    with pytest.raises(ValueError):
        frame_metrics([Landmark(0.5, 0.5)] * 10)


def test_low_visibility_legs_are_dropped():
    metrics = frame_metrics(build_pose(visibility=0.1))
    assert metrics.left is None
    assert metrics.right is None


def test_clean_pose_flags_normal():
    metrics = frame_metrics(build_pose(knee_offset=0.0))
    assert metrics.left is not None
    assert metrics.left.valgus_flag == "normal"
    assert metrics.kasr == pytest.approx(1.0, abs=0.01)


def test_collapsed_pose_flags_high_valgus():
    metrics = frame_metrics(build_pose(knee_offset=0.04))
    assert metrics.peak_valgus_deg > VALGUS_HIGH_DEG
    assert metrics.kasr < KASR_COLLAPSE


# --- Sequence handling -----------------------------------------------------


def test_find_landing_frame_picks_the_deepest_descent():
    frames = []
    for i in range(11):
        descent = math.sin(math.pi * i / 10)
        frames.append(
            frame_metrics(
                build_pose(hip_y=0.52 + 0.14 * descent, knee_y=0.74 + 0.07 * descent),
                frame_index=i,
                timestamp_s=i / 30,
            )
        )
    # The hip trough is at i == 5; the landing frame should be in its vicinity.
    assert 3 <= find_landing_frame(frames) <= 7


def test_summarise_reports_peak_not_mean_valgus():
    frames = [
        frame_metrics(build_pose(knee_offset=0.0), 0, 0.0),
        frame_metrics(build_pose(knee_offset=0.045), 1, 0.03),
        frame_metrics(build_pose(knee_offset=0.0), 2, 0.06),
    ]
    report = summarise(frames, source="video")
    assert report.peak_valgus_deg > VALGUS_HIGH_DEG
    assert report.valgus_flag == "high"


def test_summarise_with_no_usable_frames_is_safe():
    report = summarise([], source="video")
    assert report.frames_analysed == 0
    assert report.peak_valgus_deg == 0.0
    assert report.notes


def test_standing_pose_is_not_scored_as_a_stiff_landing():
    """A near-straight leg means no landing was captured, not a bad one."""
    report = summarise([frame_metrics(build_pose(knee_y=0.74))], source="image")
    assert report.max_flexion_deg < MIN_LOADING_FLEXION_DEG
    assert report.landing_assessed is False
    assert report.stiff_landing is False


def test_shallow_landing_is_scored_as_stiff():
    marks = build_pose(knee_y=0.72)
    # A little forward knee travel: real flexion, but well short of the 45 deg
    # a properly absorbed landing needs.
    marks[25] = Landmark(marks[25].x, 0.72, -0.05)
    marks[26] = Landmark(marks[26].x, 0.72, -0.05)
    marks[27] = Landmark(marks[27].x, 0.90, 0.0)
    marks[28] = Landmark(marks[28].x, 0.90, 0.0)
    report = summarise([frame_metrics(marks)], source="image")
    assert report.landing_assessed is True
    assert report.stiff_landing is True


def test_report_dict_is_json_safe():
    report = summarise([frame_metrics(build_pose(knee_offset=0.03))], source="image")
    payload = report.to_dict()
    for key in ("peak_valgus_deg", "valgus_flag", "kasr", "notes", "frontal_view"):
        assert key in payload
