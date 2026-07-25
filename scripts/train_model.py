#!/usr/bin/env python3
"""Train the workload risk model and write the bundle to models/.

    python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kneeguard.workload_model import LIGAMENTS, AthleteInput, WorkloadRiskModel  # noqa: E402


def main() -> int:
    print("Generating synthetic cohort and fitting per-ligament models...")
    model = WorkloadRiskModel.train()
    path = model.save()

    print(f"\nSaved model bundle -> {path}\n")
    print(f"{'ligament':<10}{'ROC AUC':>10}{'Brier':>10}{'event rate':>13}")
    print("-" * 43)
    for ligament in LIGAMENTS:
        metrics = model.metrics[ligament]
        print(
            f"{ligament.upper():<10}{metrics['roc_auc']:>10.3f}"
            f"{metrics['brier']:>10.4f}{metrics['event_rate']:>13.3%}"
        )

    print("\nSanity check — low-risk vs. high-risk athlete-week:\n")
    scenarios = {
        "low risk": AthleteInput(
            age=17, sex="male", minutes_last_7d=140, minutes_prior_28d=620,
            consecutive_days=1, surface="grass_dry", soreness=2, sleep_hours=9.0,
            prior_injury=False, contact_events=1, slide_tackles=0,
        ),
        "high risk": AthleteInput(
            age=16, sex="female", minutes_last_7d=430, minutes_prior_28d=520,
            consecutive_days=6, surface="turf_dry", soreness=9, sleep_hours=5.5,
            prior_injury=True, contact_events=9, slide_tackles=5,
        ),
    }
    for name, athlete in scenarios.items():
        predictions = model.predict(athlete)
        summary = "  ".join(
            f"{lig.upper()} {predictions[lig].risk_index:>5.1f}" for lig in LIGAMENTS
        )
        print(f"  {name:<10} {summary}")

    print("\nTop ACL drivers for the high-risk athlete:")
    for driver in model.predict(scenarios["high risk"])["acl"].drivers:
        print(f"  {driver['description']:<52} {driver['contribution']:+.2f}")

    print("\n" + json.dumps({"n_train": model.metrics["n_train"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
