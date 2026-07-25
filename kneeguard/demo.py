"""Synthetic movement scans for demoing without a camera.

These build pose landmark sequences directly and push them through the *same*
:mod:`kneeguard.biomechanics` code a real MediaPipe upload uses — only the
landmark source differs. Every scan produced here is tagged as synthetic in its
notes so it can never be mistaken for a measurement of a real athlete.
"""

from __future__ import annotations

import math

from .biomechanics import (
    BiomechanicsReport,
    Landmark,
    frame_metrics,
    summarise,
)

FPS = 30.0
FRAMES = 24


DEMO_SCENARIOS: dict[str, dict] = {
    "clean_landing": {
        "label": "Clean drop-jump landing",
        "description": "Knees track over the toes, deep soft landing. Textbook mechanics.",
        "valgus_travel": 0.003,
        "depth": 0.16,
        "knee_forward": 0.10,
    },
    "mild_valgus": {
        "label": "Mild knee collapse",
        "description": "Knees drift inward under load — an early warning pattern.",
        "valgus_travel": 0.014,
        "depth": 0.13,
        "knee_forward": 0.08,
    },
    "valgus_collapse": {
        "label": "Valgus collapse + stiff landing",
        "description": "Pronounced inward knee travel with a shallow, stiff landing.",
        "valgus_travel": 0.036,
        "depth": 0.06,
        "knee_forward": 0.03,
    },
}


def _drop_jump_frame(
    phase: float,
    valgus_travel: float,
    depth: float,
    knee_forward: float,
) -> list[Landmark]:
    """One frame of a synthetic front-on drop jump.

    ``phase`` runs 0 -> 1 over the clip. A half-sine drives the descent so the
    hips reach their lowest point mid-clip, which is the landing the summariser
    is designed to find.
    """
    descent = math.sin(math.pi * phase)  # 0 at start/end, 1 at deepest point

    hip_y = 0.50 + depth * descent
    knee_y = 0.70 + depth * 0.55 * descent
    ankle_y = 0.93

    # Hips and ankles hold a stable stance; the knees are what collapse inward.
    hip_half, ankle_half = 0.075, 0.075
    knee_half = hip_half - valgus_travel * descent

    # A small forward knee travel keeps the sagittal geometry realistic so the
    # flexion angle behaves like a real squat rather than a stick figure.
    knee_z = -knee_forward * descent

    landmarks = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    landmarks[11] = Landmark(0.5 + 0.105, 0.50 - 0.20 + depth * 0.5 * descent)
    landmarks[12] = Landmark(0.5 - 0.105, 0.50 - 0.20 + depth * 0.5 * descent)
    landmarks[23] = Landmark(0.5 + hip_half, hip_y)
    landmarks[24] = Landmark(0.5 - hip_half, hip_y)
    landmarks[25] = Landmark(0.5 + knee_half, knee_y, knee_z)
    landmarks[26] = Landmark(0.5 - knee_half, knee_y, knee_z)
    landmarks[27] = Landmark(0.5 + ankle_half, ankle_y)
    landmarks[28] = Landmark(0.5 - ankle_half, ankle_y)
    landmarks[29] = Landmark(0.5 + ankle_half, ankle_y + 0.02)
    landmarks[30] = Landmark(0.5 - ankle_half, ankle_y + 0.02)
    landmarks[31] = Landmark(0.5 + ankle_half, ankle_y + 0.04)
    landmarks[32] = Landmark(0.5 - ankle_half, ankle_y + 0.04)
    return landmarks


def synthetic_scan(scenario: str) -> BiomechanicsReport:
    """Build a full :class:`BiomechanicsReport` for a named demo scenario."""
    if scenario not in DEMO_SCENARIOS:
        raise ValueError(f"unknown demo scenario {scenario!r}")
    spec = DEMO_SCENARIOS[scenario]

    frames = [
        frame_metrics(
            _drop_jump_frame(
                i / (FRAMES - 1),
                spec["valgus_travel"],
                spec["depth"],
                spec["knee_forward"],
            ),
            frame_index=i,
            timestamp_s=i / FPS,
        )
        for i in range(FRAMES)
    ]

    report = summarise(frames, source="video")
    report.notes.insert(
        0,
        f"DEMO SCAN ({spec['label']}) — synthetic landmark data, not a real athlete. "
        "Upload a clip to scan a real one.",
    )
    return report
