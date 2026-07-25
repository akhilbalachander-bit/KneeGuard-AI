"""Workload-based ligament risk model (scikit-learn).

There is no public per-athlete ligament-injury dataset with the fields this app
collects, so the model is trained on a **synthetic cohort** generated from a
risk function built out of published sports-medicine risk factors. The value of
the learned model is that it turns those factors into a single calibrated,
monotone probability with per-feature attributions — not that it discovered
anything from data. This is stated plainly in the UI and the README.

Risk factors encoded in the generator
-------------------------------------
* **Acute:chronic workload ratio (ACWR)** — a spike above ~1.3 (this week's
  load far exceeding the four-week average) is the best-established modifiable
  predictor of soft-tissue injury.
* **Playing surface** — artificial turf produces higher rotational traction
  between cleat and surface, raising torsional load on the ACL. Wet natural
  grass sits at the other extreme: low traction, more slips and uncontrolled
  landings. Both ends of that range are hard on the MCL, which is why the MCL
  term uses distance from a neutral traction value rather than traction itself.
* **Fatigue** — soreness and short sleep degrade neuromuscular control, and
  landing mechanics deteriorate measurably late in matches.
* **Sex** — female athletes tear the ACL at 2-8x the rate of male athletes in
  matched sports, attributed to Q-angle, hormonal, and neuromuscular factors.
* **Prior injury** — a previously injured knee is among the strongest single
  predictors of re-injury.
* **Contact and sliding tackles** — direct blows drive MCL and PCL injury; the
  PCL specifically fails under a posterior force on a flexed knee, the exact
  loading of a sliding tackle or a fall onto a bent knee.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config

LIGAMENTS = ("acl", "mcl", "pcl")

FEATURE_NAMES = (
    "age",
    "female",
    "minutes_last_7d",
    "minutes_prior_28d",
    "acwr",
    "acwr_excess",
    "consecutive_days",
    "is_turf",
    "is_wet",
    "surface_traction",
    "traction_mismatch",
    "soreness",
    "sleep_hours",
    "prior_injury",
    "contact_events",
    "slide_tackles",
)

# Cleat-surface rotational traction. Higher = more grip = more torque through
# the knee before the shoe releases.
SURFACE_TRACTION = {
    "turf_dry": 1.00,
    "turf_wet": 0.85,
    "grass_dry": 0.70,
    "grass_wet": 0.55,
}
NEUTRAL_TRACTION = 0.75

SURFACE_LABELS = {
    "turf_dry": "Dry artificial turf",
    "turf_wet": "Wet artificial turf",
    "grass_dry": "Dry natural grass",
    "grass_wet": "Wet natural grass",
}

# Human-readable names used in the "top risk drivers" panel.
FEATURE_LABELS = {
    "age": "Age",
    "female": "Sex (ACL incidence)",
    "minutes_last_7d": "Minutes played this week",
    "minutes_prior_28d": "Minutes played previous 4 weeks",
    "acwr": "Acute:chronic workload ratio",
    "acwr_excess": "Workload spike above safe ratio",
    "consecutive_days": "Consecutive days played",
    "is_turf": "Artificial turf exposure",
    "is_wet": "Wet playing surface",
    "surface_traction": "Cleat-surface rotational traction",
    "traction_mismatch": "Traction away from neutral footing",
    "soreness": "Muscle soreness",
    "sleep_hours": "Sleep per night",
    "prior_injury": "Previous knee injury",
    "contact_events": "Physical collisions per week",
    "slide_tackles": "Sliding tackles per week",
}

# How to phrase a feature when it is *pushing risk up*. Some features raise risk
# when high (soreness), others when low (sleep), so the driver panel needs the
# direction, not just the coefficient.
_DRIVER_PHRASES = {
    "acwr_excess": ("Workload spike: {value:.2f} above the safe 1.3 ratio", "high"),
    "acwr": ("Acute:chronic workload ratio of {value:.2f}", "high"),
    "minutes_last_7d": ("{value:.0f} minutes played in the last 7 days", "high"),
    "minutes_prior_28d": ("{value:.0f} minutes in the previous 4 weeks", "high"),
    "consecutive_days": ("{value:.0f} consecutive days played", "high"),
    "soreness": ("Muscle soreness reported at {value:.0f}/10", "high"),
    "sleep_hours": ("Only {value:.1f} h of sleep per night", "low"),
    "prior_injury": ("Previous knee injury on record", "flag"),
    "female": ("Female athlete (2-8x baseline ACL incidence)", "flag"),
    "age": ("Age {value:.0f} (adolescent growth phase)", "low"),
    "is_turf": ("Playing on artificial turf", "flag"),
    "is_wet": ("Wet playing surface", "flag"),
    "surface_traction": ("High cleat-surface rotational traction ({value:.2f})", "high"),
    "traction_mismatch": ("Footing {value:.2f} away from neutral traction", "high"),
    "contact_events": ("{value:.0f} physical collisions this week", "high"),
    "slide_tackles": ("{value:.0f} sliding tackles this week", "high"),
}


def describe_driver(feature: str, value: float) -> str:
    """Render a risk driver as a sentence a coach can act on."""
    template, _ = _DRIVER_PHRASES.get(feature, ("{label}: {value:.1f}", "high"))
    return template.format(value=value, label=FEATURE_LABELS.get(feature, feature))


@dataclass
class AthleteInput:
    """The workload questionnaire, in the units the athlete enters them."""

    age: int = 17
    sex: str = "male"  # "male" | "female" | "unspecified"
    minutes_last_7d: int = 180
    minutes_prior_28d: int = 600
    consecutive_days: int = 2
    surface: str = "grass_dry"
    soreness: int = 4  # 1-10
    sleep_hours: float = 8.0
    prior_injury: bool = False
    contact_events: int = 3
    slide_tackles: int = 1

    def validate(self) -> None:
        if self.surface not in SURFACE_TRACTION:
            raise ValueError(
                f"surface must be one of {sorted(SURFACE_TRACTION)}, got {self.surface!r}"
            )
        if not 1 <= self.soreness <= 10:
            raise ValueError("soreness must be between 1 and 10")
        if not 8 <= self.age <= 60:
            raise ValueError("age must be between 8 and 60")
        for name in ("minutes_last_7d", "minutes_prior_28d", "consecutive_days",
                     "contact_events", "slide_tackles"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0 <= self.sleep_hours <= 16:
            raise ValueError("sleep_hours must be between 0 and 16")

    def to_dict(self) -> dict:
        return asdict(self)


def acwr(minutes_last_7d: float, minutes_prior_28d: float) -> float:
    """Acute:chronic workload ratio (this week vs. the 4-week weekly average).

    A player returning from a lay-off has no chronic base; guarding the divisor
    keeps that case finite instead of infinite, and the resulting high ratio is
    the correct signal.
    """
    chronic_weekly = max(minutes_prior_28d / 4.0, 30.0)
    return minutes_last_7d / chronic_weekly


def build_features(athlete: AthleteInput) -> np.ndarray:
    """Turn the questionnaire into the model's feature vector."""
    traction = SURFACE_TRACTION[athlete.surface]
    ratio = acwr(athlete.minutes_last_7d, athlete.minutes_prior_28d)
    values = {
        "age": float(athlete.age),
        "female": 1.0 if athlete.sex == "female" else 0.0,
        "minutes_last_7d": float(athlete.minutes_last_7d),
        "minutes_prior_28d": float(athlete.minutes_prior_28d),
        "acwr": ratio,
        "acwr_excess": max(0.0, ratio - 1.3),
        "consecutive_days": float(athlete.consecutive_days),
        "is_turf": 1.0 if athlete.surface.startswith("turf") else 0.0,
        "is_wet": 1.0 if athlete.surface.endswith("wet") else 0.0,
        "surface_traction": traction,
        "traction_mismatch": abs(traction - NEUTRAL_TRACTION),
        "soreness": float(athlete.soreness),
        "sleep_hours": float(athlete.sleep_hours),
        "prior_injury": 1.0 if athlete.prior_injury else 0.0,
        "contact_events": float(athlete.contact_events),
        "slide_tackles": float(athlete.slide_tackles),
    }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=float)


# --- Synthetic cohort -------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_cohort(n: int = 24_000, seed: int = config.RANDOM_SEED):
    """Sample a synthetic squad-season cohort and its ligament-event labels.

    Returns ``(X, y)`` where ``y`` has one binary column per ligament: did this
    athlete-week end in a ligament event (sprain or tear)?
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(17.5, 2.6, n).clip(13, 30)
    female = rng.integers(0, 2, n).astype(float)
    minutes_prior_28d = rng.gamma(6.0, 110.0, n).clip(0, 2400)
    load_multiplier = rng.lognormal(0.0, 0.42, n)
    minutes_last_7d = (minutes_prior_28d / 4.0 * load_multiplier).clip(0, 900)
    consecutive_days = rng.poisson(2.2, n).clip(0, 14).astype(float)
    surface_idx = rng.choice(len(SURFACE_TRACTION), n, p=[0.28, 0.12, 0.42, 0.18])
    surface_keys = list(SURFACE_TRACTION)
    traction = np.array([SURFACE_TRACTION[surface_keys[i]] for i in surface_idx])
    is_turf = np.array([1.0 if surface_keys[i].startswith("turf") else 0.0 for i in surface_idx])
    is_wet = np.array([1.0 if surface_keys[i].endswith("wet") else 0.0 for i in surface_idx])

    # Soreness rises with load and falls with sleep — not independent noise.
    ratio = minutes_last_7d / np.maximum(minutes_prior_28d / 4.0, 30.0)
    sleep_hours = rng.normal(7.6, 1.15, n).clip(3.5, 11)
    soreness = (
        2.0 + 2.4 * np.tanh(ratio - 1.0) + 0.45 * consecutive_days
        - 0.35 * (sleep_hours - 7.6) + rng.normal(0, 1.3, n)
    ).clip(1, 10)

    prior_injury = (rng.random(n) < 0.17).astype(float)
    contact_events = rng.poisson(3.1, n).clip(0, 25).astype(float)
    slide_tackles = rng.poisson(1.3, n).clip(0, 15).astype(float)

    acwr_excess = np.maximum(0.0, ratio - 1.3)
    traction_mismatch = np.abs(traction - NEUTRAL_TRACTION)
    youth = np.clip((19.0 - age) / 6.0, 0, 1)  # adolescent growth-phase risk

    # Latent log-odds of a ligament event in the next 7 days.
    acl_logit = (
        -4.35
        + 1.70 * acwr_excess
        + 0.16 * (soreness - 4.0)
        + 0.11 * consecutive_days
        + 0.78 * female
        + 0.85 * prior_injury
        + 2.10 * (traction - NEUTRAL_TRACTION)
        - 0.17 * (sleep_hours - 7.6)
        + 0.42 * youth
    )
    mcl_logit = (
        -4.20
        + 0.20 * (contact_events - 3.1)
        + 1.90 * traction_mismatch
        + 0.13 * (soreness - 4.0)
        + 0.55 * prior_injury
        + 0.70 * acwr_excess
        + 0.10 * slide_tackles
    )
    pcl_logit = (
        -5.30
        + 0.42 * slide_tackles
        + 0.16 * (contact_events - 3.1)
        + 0.08 * (soreness - 4.0)
        + 0.45 * prior_injury
    )

    X = np.column_stack(
        [
            age,
            female,
            minutes_last_7d,
            minutes_prior_28d,
            ratio,
            acwr_excess,
            consecutive_days,
            is_turf,
            is_wet,
            traction,
            traction_mismatch,
            soreness,
            sleep_hours,
            prior_injury,
            contact_events,
            slide_tackles,
        ]
    )
    probabilities = np.column_stack(
        [_sigmoid(acl_logit), _sigmoid(mcl_logit), _sigmoid(pcl_logit)]
    )
    y = (rng.random(probabilities.shape) < probabilities).astype(int)
    return X, y


# Relative sensitivity of each ligament to the findings a pose scan can produce.
# The ACL has the most screenable failure modes (valgus, KASR, stiff landing,
# trunk lean, asymmetry); the PCL has the fewest.
SCAN_SENSITIVITY = {"acl": 1.00, "mcl": 0.65, "pcl": 0.35}


def sample_scan_adjustments(
    n: int, ligament: str, rng: np.random.Generator
) -> np.ndarray:
    """Plausible spread of biomechanical log-odds adjustments across a squad.

    The reference distribution used for percentile scoring has to represent
    athletes assessed the *same way* KneeGuard assesses them — workload plus a
    movement scan. Without this, an athlete at the top of the workload
    distribution would already sit at 100 and the scan could never move the
    number, which would make the biomechanical half of the product invisible.

    The mixture reflects what drop-jump screening actually finds: most athletes
    land cleanly, a minority show mild collapse, and a small group show frank
    valgus collapse.
    """
    scale = SCAN_SENSITIVITY[ligament]
    draw = rng.random(n)
    adjustment = np.zeros(n)

    mild = (draw >= 0.55) & (draw < 0.85)
    severe = draw >= 0.85
    adjustment[mild] = rng.uniform(0.05, 0.85, mild.sum())
    adjustment[severe] = rng.uniform(0.85, 2.2, severe.sum())
    return adjustment * scale


# --- Model bundle -----------------------------------------------------------


@dataclass
class LigamentPrediction:
    ligament: str
    probability: float
    risk_index: float
    drivers: list[dict]


class WorkloadRiskModel:
    """Per-ligament logistic regression with percentile-calibrated scoring.

    Logistic regression is deliberate here: it stays monotone in each risk
    factor (a coach can never make the score go down by reporting *more*
    soreness), it calibrates well, and its standardised coefficients give the
    per-feature attributions shown in the dashboard.
    """

    def __init__(self, pipelines: dict, cohort_scores: dict, metrics: dict):
        self.pipelines = pipelines
        self.cohort_scores = {k: np.asarray(v) for k, v in cohort_scores.items()}
        self.metrics = metrics

    # -- training --

    @classmethod
    def train(cls, n: int = 24_000, seed: int = config.RANDOM_SEED) -> "WorkloadRiskModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = generate_cohort(n=n, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )

        pipelines: dict = {}
        cohort_scores: dict = {}
        metrics: dict = {"n_train": int(len(X_train)), "n_test": int(len(X_test))}

        for i, ligament in enumerate(LIGAMENTS):
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, C=1.0)),
                ]
            )
            pipeline.fit(X_train, y_train[:, i])
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics[ligament] = {
                "roc_auc": float(roc_auc_score(y_test[:, i], probabilities)),
                "brier": float(brier_score_loss(y_test[:, i], probabilities)),
                "event_rate": float(y_test[:, i].mean()),
            }
            pipelines[ligament] = pipeline

            # Reference distribution for percentile scoring. Built from the full
            # cohort with a plausible movement-scan adjustment applied, so the
            # percentile scale has headroom for what a scan can contribute.
            cohort_probability = pipeline.predict_proba(X)[:, 1]
            reference_rng = np.random.default_rng(seed + i + 1)
            adjusted = _sigmoid(
                np.log(cohort_probability / (1 - cohort_probability))
                + sample_scan_adjustments(len(cohort_probability), ligament, reference_rng)
            )
            cohort_scores[ligament] = np.sort(adjusted)

        return cls(pipelines, cohort_scores, metrics)

    # -- inference --

    def _risk_index(self, ligament: str, probability: float) -> float:
        """Percentile of this athlete's modelled probability within the cohort.

        The dashboard's 0-100 number is a *relative* risk index, not a
        probability of injury. A 78 means "higher modelled risk than 78% of
        athlete-weeks in the reference cohort".
        """
        scores = self.cohort_scores[ligament]
        rank = float(np.searchsorted(scores, probability, side="right"))
        return round(100.0 * rank / len(scores), 1)

    def _drivers(self, ligament: str, features: np.ndarray, top_k: int = 4) -> list[dict]:
        """Standardised per-feature contributions to the log-odds."""
        pipeline = self.pipelines[ligament]
        scaler = pipeline.named_steps["scale"]
        coefficients = pipeline.named_steps["lr"].coef_[0]
        standardised = (features - scaler.mean_) / scaler.scale_
        contributions = coefficients * standardised

        order = np.argsort(-contributions)
        drivers: list[dict] = []
        for index in order:
            if len(drivers) >= top_k or contributions[index] <= 0.02:
                break
            feature = FEATURE_NAMES[index]
            value = float(features[index])
            # Binary flags only make sense as drivers when they are actually set.
            if feature in ("female", "prior_injury", "is_turf", "is_wet") and value < 0.5:
                continue
            drivers.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "description": describe_driver(feature, value),
                    "value": round(value, 2),
                    "contribution": round(float(contributions[index]), 3),
                }
            )
        return drivers

    def predict(self, athlete: AthleteInput) -> dict[str, LigamentPrediction]:
        features = build_features(athlete)
        row = features.reshape(1, -1)
        results = {}
        for ligament in LIGAMENTS:
            probability = float(self.pipelines[ligament].predict_proba(row)[0, 1])
            results[ligament] = LigamentPrediction(
                ligament=ligament,
                probability=probability,
                risk_index=self._risk_index(ligament, probability),
                drivers=self._drivers(ligament, features),
            )
        return results

    def index_for_probability(self, ligament: str, probability: float) -> float:
        """Public percentile mapping, used after biomechanical fusion."""
        return self._risk_index(ligament, probability)

    # -- persistence --

    def save(self, path: Path | None = None) -> Path:
        import joblib

        path = Path(path or config.RISK_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipelines": self.pipelines,
                "cohort_scores": {k: v for k, v in self.cohort_scores.items()},
                "metrics": self.metrics,
                "feature_names": list(FEATURE_NAMES),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "WorkloadRiskModel":
        import joblib

        path = Path(path or config.RISK_MODEL_PATH)
        bundle = joblib.load(path)
        return cls(bundle["pipelines"], bundle["cohort_scores"], bundle["metrics"])

    @classmethod
    def load_or_train(cls, path: Path | None = None) -> "WorkloadRiskModel":
        path = Path(path or config.RISK_MODEL_PATH)
        if path.exists():
            try:
                return cls.load(path)
            except Exception:  # noqa: BLE001 - a stale bundle should not be fatal
                pass
        model = cls.train()
        model.save(path)
        return model


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def inverse_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
