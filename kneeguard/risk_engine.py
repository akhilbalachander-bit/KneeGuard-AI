"""Fuse workload risk with biomechanical findings into a ligament scorecard.

The two signals are combined in **log-odds space**: the trained workload model
supplies a base probability, and each biomechanical finding contributes an
additive adjustment to its logit. That keeps the fusion monotone (a worse scan
can never lower the score), keeps the result a valid probability, and leaves
every adjustment individually inspectable — which matters more here than raw
accuracy, because the athlete is being told *why*.

Adjustment magnitudes are hand-set from the relative strength of each finding
in the screening literature, not fitted; they are exposed in the API response
so nothing is hidden behind a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .biomechanics import (
    DEEP_FLEXION_DEG,
    KASR_COLLAPSE,
    STIFF_LANDING_FLEXION_DEG,
    TRUNK_LEAN_HIGH_DEG,
    VALGUS_ELEVATED_DEG,
    ASYMMETRY_HIGH_DEG,
    BiomechanicsReport,
)
from .exercises import select_plan, to_dict as exercise_to_dict
from .workload_model import (
    LIGAMENTS,
    AthleteInput,
    WorkloadRiskModel,
    acwr,
    inverse_logit,
    logit,
)

# Ceiling on the total biomechanical adjustment per ligament, in log-odds. A
# screening scan should be able to move the score substantially but never
# swamp the workload evidence entirely.
MAX_ADJUSTMENT = 2.2

# Risk bands use the reserved *status* palette, not series colours — the band is
# a state (good / warning / serious / critical), not an identity. Each band also
# carries an icon and a label so the state never depends on colour alone.
RISK_BANDS = (
    (85.0, "critical", "#d03b3b", "critical"),
    (70.0, "high", "#ec835a", "serious"),
    (40.0, "moderate", "#fab219", "warning"),
    (0.0, "low", "#0ca30c", "good"),
)

LIGAMENT_NAMES = {
    "acl": "Anterior Cruciate Ligament",
    "mcl": "Medial Collateral Ligament",
    "pcl": "Posterior Cruciate Ligament",
}

LIGAMENT_MECHANISMS = {
    "acl": (
        "Non-contact pivoting, sudden deceleration, or landing flat-footed with the "
        "knee caving inward (valgus)."
    ),
    "mcl": (
        "A direct blow to the outside of the knee, or the knee being forced inward "
        "while the cleat stays planted."
    ),
    "pcl": (
        "A direct force to the front of an already-bent knee — a sliding tackle or "
        "falling hard onto a flexed knee."
    ),
}


def band_for(index: float) -> tuple[str, str, str]:
    """Return ``(band label, colour, status role)`` for a 0-100 risk index."""
    for threshold, label, colour, status in RISK_BANDS:
        if index >= threshold:
            return label, colour, status
    return "low", "#0ca30c", "good"


@dataclass
class Adjustment:
    """One biomechanical finding and what it did to a ligament's score."""

    finding: str
    detail: str
    log_odds: float


@dataclass
class LigamentScore:
    ligament: str
    risk_index: float
    baseline_index: float
    probability: float
    band: str
    colour: str
    status: str
    drivers: list[dict]
    adjustments: list[Adjustment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ligament": self.ligament.upper(),
            "name": LIGAMENT_NAMES[self.ligament],
            "mechanism": LIGAMENT_MECHANISMS[self.ligament],
            "risk_index": round(self.risk_index, 1),
            "workload_only_index": round(self.baseline_index, 1),
            "biomechanical_shift": round(self.risk_index - self.baseline_index, 1),
            "modelled_probability": round(self.probability, 4),
            "band": self.band,
            "colour": self.colour,
            "status": self.status,
            "drivers": self.drivers,
            "adjustments": [
                {
                    "finding": a.finding,
                    "detail": a.detail,
                    "log_odds": round(a.log_odds, 3),
                }
                for a in self.adjustments
            ],
        }


def _biomechanical_adjustments(
    scan: BiomechanicsReport | None,
) -> dict[str, list[Adjustment]]:
    """Translate scan findings into per-ligament log-odds adjustments."""
    result: dict[str, list[Adjustment]] = {lig: [] for lig in LIGAMENTS}
    if scan is None or scan.frames_analysed == 0:
        return result

    # A low-confidence scan should nudge, not shout.
    weight = max(0.0, min(1.0, scan.confidence))

    if scan.frontal_view and scan.peak_valgus_deg > VALGUS_ELEVATED_DEG:
        excess = scan.peak_valgus_deg - VALGUS_ELEVATED_DEG
        side = scan.valgus_side or "loaded"
        detail = (
            f"Peak knee valgus {scan.peak_valgus_deg:.1f} deg on the {side} leg "
            f"({excess:.1f} deg past the {VALGUS_ELEVATED_DEG:.0f} deg screening threshold)."
        )
        result["acl"].append(Adjustment("Inward knee collapse", detail, 0.085 * excess * weight))
        result["mcl"].append(Adjustment("Inward knee collapse", detail, 0.045 * excess * weight))

    if scan.frontal_view and scan.kasr is not None and scan.kasr < KASR_COLLAPSE:
        gap = KASR_COLLAPSE - scan.kasr
        detail = (
            f"Knee-to-ankle separation ratio {scan.kasr:.2f} — the knees tracked inside "
            "the ankles through the landing."
        )
        result["acl"].append(Adjustment("Medial collapse (KASR)", detail, 2.4 * gap * weight))
        result["mcl"].append(Adjustment("Medial collapse (KASR)", detail, 1.4 * gap * weight))

    if scan.stiff_landing:
        gap = STIFF_LANDING_FLEXION_DEG - scan.landing_flexion_deg
        detail = (
            f"Only {scan.landing_flexion_deg:.0f} deg of knee flexion at landing "
            f"(target is at least {STIFF_LANDING_FLEXION_DEG:.0f} deg)."
        )
        result["acl"].append(Adjustment("Stiff landing", detail, 0.030 * gap * weight))
        result["pcl"].append(Adjustment("Stiff landing", detail, 0.012 * gap * weight))

    if scan.frontal_view and scan.trunk_lean_deg and scan.trunk_lean_deg > TRUNK_LEAN_HIGH_DEG:
        excess = scan.trunk_lean_deg - TRUNK_LEAN_HIGH_DEG
        detail = (
            f"Lateral trunk lean of {scan.trunk_lean_deg:.0f} deg shifts the centre of "
            "mass over the stance leg and raises the knee abduction moment."
        )
        result["acl"].append(Adjustment("Lateral trunk lean", detail, 0.035 * excess * weight))
        result["mcl"].append(Adjustment("Lateral trunk lean", detail, 0.025 * excess * weight))

    if scan.frontal_view and scan.asymmetry_deg >= ASYMMETRY_HIGH_DEG:
        detail = (
            f"{scan.asymmetry_deg:.1f} deg difference in valgus between legs — one side "
            "is absorbing load differently."
        )
        result["acl"].append(Adjustment("Left/right asymmetry", detail, 0.25 * weight))
        result["mcl"].append(Adjustment("Left/right asymmetry", detail, 0.15 * weight))

    if scan.deep_flexion_loading:
        excess = scan.max_flexion_deg - DEEP_FLEXION_DEG
        detail = (
            f"Knee reached {scan.max_flexion_deg:.0f} deg of flexion under load — the "
            "position in which a posterior force loads the PCL."
        )
        result["pcl"].append(Adjustment("Deep flexion under load", detail, 0.022 * excess * weight))

    return result


def _deficits(
    athlete: AthleteInput, scan: BiomechanicsReport | None
) -> set[str]:
    """Everything the action plan should respond to."""
    found: set[str] = set()

    ratio = acwr(athlete.minutes_last_7d, athlete.minutes_prior_28d)
    if ratio > 1.3:
        found.add("workload_spike")
    if athlete.consecutive_days >= 4:
        found.add("consecutive_days")
    if athlete.soreness >= 6:
        found.add("fatigue")
    if athlete.sleep_hours < 7.0:
        found.update({"sleep", "fatigue"})
    if athlete.prior_injury:
        found.add("prior_injury")
    if athlete.contact_events >= 5:
        found.update({"contact", "collisions"})
    if athlete.slide_tackles >= 3:
        found.add("slide_tackles")
    if athlete.surface.startswith("turf"):
        found.update({"turf", "traction"})
    if athlete.surface == "grass_wet":
        found.add("traction")

    if scan and scan.frames_analysed:
        if scan.frontal_view and scan.peak_valgus_deg > VALGUS_ELEVATED_DEG:
            found.add("valgus")
        if scan.frontal_view and scan.kasr is not None and scan.kasr < KASR_COLLAPSE:
            found.add("kasr_collapse")
        if scan.stiff_landing:
            found.add("stiff_landing")
        if scan.frontal_view and scan.asymmetry_deg >= ASYMMETRY_HIGH_DEG:
            found.add("asymmetry")
        if scan.deep_flexion_loading:
            found.add("deep_flexion")

    return found


def assess(
    athlete: AthleteInput,
    model: WorkloadRiskModel,
    scan: BiomechanicsReport | None = None,
) -> dict:
    """Produce the full scorecard: per-ligament risk, findings, and action plan."""
    athlete.validate()

    baseline = model.predict(athlete)
    adjustments = _biomechanical_adjustments(scan)

    scores: dict[str, LigamentScore] = {}
    for ligament in LIGAMENTS:
        base = baseline[ligament]
        total = sum(a.log_odds for a in adjustments[ligament])
        total = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, total))
        fused_probability = inverse_logit(logit(base.probability) + total)
        index = model.index_for_probability(ligament, fused_probability)
        band, colour, status = band_for(index)
        scores[ligament] = LigamentScore(
            ligament=ligament,
            risk_index=index,
            baseline_index=base.risk_index,
            probability=fused_probability,
            band=band,
            colour=colour,
            status=status,
            drivers=base.drivers,
            adjustments=adjustments[ligament],
        )

    ligament_risk = {lig: score.risk_index for lig, score in scores.items()}
    plan = select_plan(_deficits(athlete, scan), ligament_risk)

    # The headline gauge is the worst ligament — that is the one that will end
    # a season, and averaging would hide it.
    overall_index = max(ligament_risk.values())
    overall_band, overall_colour, overall_status = band_for(overall_index)
    primary = max(ligament_risk, key=lambda k: ligament_risk[k])

    return {
        "overall": {
            "risk_index": round(overall_index, 1),
            "band": overall_band,
            "colour": overall_colour,
            "status": overall_status,
            "primary_ligament": primary.upper(),
        },
        "ligaments": [scores[lig].to_dict() for lig in LIGAMENTS],
        "action_plan": [exercise_to_dict(exercise) for exercise in plan],
        "workload": {
            "acwr": round(acwr(athlete.minutes_last_7d, athlete.minutes_prior_28d), 2),
            "acwr_safe_ceiling": 1.3,
            "surface": athlete.surface,
            "inputs": athlete.to_dict(),
        },
        "scan": scan.to_dict() if scan else None,
        "model_metrics": model.metrics,
    }
