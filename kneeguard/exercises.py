"""Prevention exercise library and action-plan selection.

Every entry is drawn from an established injury-prevention protocol — the FIFA
11+ warm-up, the Santa Monica PEP program, and the Nordic hamstring literature
— rather than invented. Each is tagged with the specific deficit it addresses
so the plan responds to what the scan and questionnaire actually found.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Exercise:
    key: str
    name: str
    target: str
    dose: str
    rationale: str
    addresses: tuple[str, ...]
    ligaments: tuple[str, ...]


LIBRARY: tuple[Exercise, ...] = (
    Exercise(
        key="nordic_curl",
        name="Nordic Hamstring Curl",
        target="Eccentric hamstring strength",
        dose="3 sets x 5 reps, 2x per week, lowering slowly over 4-5 seconds",
        rationale=(
            "Strong eccentric hamstrings resist the tibia sliding forward under the femur "
            "— the exact motion the ACL is left to stop when the hamstrings fatigue."
        ),
        addresses=("valgus", "fatigue", "workload_spike", "prior_injury"),
        ligaments=("acl",),
    ),
    Exercise(
        key="lateral_band_walk",
        name="Lateral Band Walk (Monster Walk)",
        target="Gluteus medius / hip abductors",
        dose="3 sets x 15 steps each direction, band above the knees, before every session",
        rationale=(
            "Weak hip abductors let the femur drop into adduction and internal rotation, "
            "which is what pulls the knee inward into valgus during landing and cutting."
        ),
        addresses=("valgus", "kasr_collapse", "asymmetry"),
        ligaments=("acl", "mcl"),
    ),
    Exercise(
        key="single_leg_landing",
        name="Single-Leg Drop Landing (knee-over-toe cue)",
        target="Neuromuscular landing control",
        dose="3 sets x 6 landings per leg, hold each landing 2 seconds before the next rep",
        rationale=(
            "Retrains the landing pattern directly: land softly, hips back, knee tracking "
            "over the second toe rather than collapsing toward the midline."
        ),
        addresses=("valgus", "kasr_collapse", "stiff_landing", "asymmetry"),
        ligaments=("acl", "mcl"),
    ),
    Exercise(
        key="soft_landing_depth_drop",
        name="Depth Drop to Soft Landing",
        target="Impact absorption / landing stiffness",
        dose="4 sets x 5 drops from a 30 cm box, aiming for a silent landing",
        rationale=(
            "A stiff, near-straight-legged landing sends ground reaction force through the "
            "joint instead of the muscles. Cueing a deeper, quieter landing moves that load "
            "into the quadriceps and glutes."
        ),
        addresses=("stiff_landing",),
        ligaments=("acl", "pcl"),
    ),
    Exercise(
        key="copenhagen_plank",
        name="Copenhagen Adductor Plank",
        target="Adductor / medial hip strength",
        dose="3 sets x 15 seconds per side, progressing to 30 seconds",
        rationale=(
            "Builds medial-chain strength so the leg can resist a valgus force from the "
            "outside of the knee, which is the classic MCL mechanism in a tackle."
        ),
        addresses=("contact", "valgus", "traction"),
        ligaments=("mcl",),
    ),
    Exercise(
        key="spanish_squat",
        name="Spanish Squat (banded isometric)",
        target="Quadriceps strength at depth",
        dose="4 sets x 30 second holds at 60 degrees of knee flexion",
        rationale=(
            "The quadriceps pull the tibia forward, directly opposing the posterior force "
            "that injures the PCL when a bent knee is driven backwards in a sliding tackle "
            "or a hard fall."
        ),
        addresses=("deep_flexion", "collisions", "slide_tackles"),
        ligaments=("pcl",),
    ),
    Exercise(
        key="single_leg_rdl",
        name="Single-Leg Romanian Deadlift",
        target="Posterior chain + single-leg balance",
        dose="3 sets x 8 reps per leg, controlled tempo",
        rationale=(
            "Trains hip hinge and balance on one leg, so the athlete decelerates through "
            "the hips instead of dumping load into the knee."
        ),
        addresses=("asymmetry", "fatigue", "valgus"),
        ligaments=("acl", "mcl"),
    ),
    Exercise(
        key="fifa11_agility",
        name="FIFA 11+ Part 3: Planted-Foot Cutting Drills",
        target="Change-of-direction technique",
        dose="2 sets x 6 cuts per side, at the start of every session on turf",
        rationale=(
            "High-traction surfaces let the cleat hold while the body keeps rotating. "
            "Practising a wider, lower, multi-step cut reduces the torque that reaches "
            "the knee before the shoe releases."
        ),
        addresses=("traction", "turf"),
        ligaments=("acl", "mcl"),
    ),
    Exercise(
        key="load_management",
        name="Planned Load Reduction",
        target="Acute:chronic workload ratio",
        dose="Cut this week's minutes by 25-30% and take one full rest day before the next match",
        rationale=(
            "A sharp spike in weekly minutes over the four-week average is the most "
            "modifiable injury risk factor there is. Bringing the ratio back under 1.3 "
            "lowers risk across all three ligaments at once."
        ),
        addresses=("workload_spike", "consecutive_days", "fatigue"),
        ligaments=("acl", "mcl", "pcl"),
    ),
    Exercise(
        key="sleep_recovery",
        name="Sleep Extension Protocol",
        target="Neuromuscular recovery",
        dose="Target 8-9 hours per night; treat it as part of training, not optional",
        rationale=(
            "Under about 7 hours of sleep, reaction time and landing control degrade "
            "measurably — the same control deficits that show up as knee valgus."
        ),
        addresses=("sleep", "fatigue"),
        ligaments=("acl", "mcl", "pcl"),
    ),
)

BY_KEY = {exercise.key: exercise for exercise in LIBRARY}


def select_plan(
    deficits: set[str],
    ligament_risk: dict[str, float],
    limit: int = 3,
) -> list[Exercise]:
    """Pick the exercises that best match the detected deficits.

    Scoring is deliberately simple and transparent: an exercise earns a point
    per deficit it addresses, weighted up if it targets the ligament that came
    back highest. Ties break toward the library's clinical ordering.
    """
    highest = max(ligament_risk, key=lambda k: ligament_risk[k]) if ligament_risk else None

    scored: list[tuple[float, int, Exercise]] = []
    for position, exercise in enumerate(LIBRARY):
        matches = deficits & set(exercise.addresses)
        if not matches:
            continue
        score = float(len(matches))
        if highest and highest in exercise.ligaments:
            score += 1.5
        # Weight by how elevated the ligaments this exercise protects actually are.
        score += sum(ligament_risk.get(lig, 0.0) for lig in exercise.ligaments) / 200.0
        scored.append((-score, position, exercise))

    scored.sort()
    plan = [exercise for _, _, exercise in scored[:limit]]

    if not plan:
        # Nothing flagged: give the athlete the baseline prevention protocol.
        plan = [BY_KEY["nordic_curl"], BY_KEY["lateral_band_walk"], BY_KEY["single_leg_landing"]]
    return plan


def to_dict(exercise: Exercise) -> dict:
    return {
        "key": exercise.key,
        "name": exercise.name,
        "target": exercise.target,
        "dose": exercise.dose,
        "rationale": exercise.rationale,
        "ligaments": [lig.upper() for lig in exercise.ligaments],
    }
