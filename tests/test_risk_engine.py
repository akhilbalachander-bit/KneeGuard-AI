"""Behavioural tests for the workload model, fusion engine, and action plan."""

from __future__ import annotations

import math

import pytest

from kneeguard.biomechanics import BiomechanicsReport, frame_metrics, summarise
from kneeguard.demo import DEMO_SCENARIOS, synthetic_scan
from kneeguard.exercises import LIBRARY, select_plan
from kneeguard.risk_engine import MAX_ADJUSTMENT, assess, band_for
from kneeguard.workload_model import (
    FEATURE_NAMES,
    LIGAMENTS,
    AthleteInput,
    WorkloadRiskModel,
    acwr,
    build_features,
    inverse_logit,
    logit,
)

from .test_biomechanics import build_pose


@pytest.fixture(scope="module")
def model() -> WorkloadRiskModel:
    # Trained rather than loaded so the suite never depends on a stale bundle.
    return WorkloadRiskModel.train(n=6000)


def athlete(**overrides) -> AthleteInput:
    defaults = dict(
        age=17,
        sex="male",
        minutes_last_7d=250,
        minutes_prior_28d=900,
        consecutive_days=2,
        surface="grass_dry",
        soreness=3,
        sleep_hours=8.0,
        prior_injury=False,
        contact_events=2,
        slide_tackles=1,
    )
    defaults.update(overrides)
    return AthleteInput(**defaults)


# --- Workload features -----------------------------------------------------


def test_acwr_matches_definition():
    assert acwr(400, 800) == pytest.approx(2.0)


def test_acwr_guards_against_zero_chronic_load():
    """A player back from a lay-off must not produce an infinite ratio."""
    assert math.isfinite(acwr(300, 0))
    assert acwr(300, 0) > 1.3


def test_build_features_matches_declared_order():
    features = build_features(athlete())
    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("soreness")] == 3.0


def test_turf_and_wet_flags_are_encoded():
    features = build_features(athlete(surface="turf_wet"))
    assert features[FEATURE_NAMES.index("is_turf")] == 1.0
    assert features[FEATURE_NAMES.index("is_wet")] == 1.0


def test_invalid_surface_is_rejected():
    with pytest.raises(ValueError):
        athlete(surface="sand").validate()


def test_invalid_soreness_is_rejected():
    with pytest.raises(ValueError):
        athlete(soreness=99).validate()


def test_negative_minutes_are_rejected():
    with pytest.raises(ValueError):
        athlete(minutes_last_7d=-5).validate()


# --- Model behaviour -------------------------------------------------------


def test_risk_index_is_monotone_in_soreness(model):
    scores = [
        assess(athlete(soreness=level), model)["ligaments"][0]["risk_index"]
        for level in (1, 4, 7, 10)
    ]
    assert scores == sorted(scores), scores


def test_risk_index_is_monotone_in_workload_spike(model):
    scores = [
        assess(athlete(minutes_last_7d=minutes), model)["ligaments"][0]["risk_index"]
        for minutes in (120, 250, 400, 600)
    ]
    assert scores == sorted(scores), scores


def test_turf_raises_acl_risk_over_dry_grass(model):
    grass = assess(athlete(surface="grass_dry"), model)["ligaments"][0]["risk_index"]
    turf = assess(athlete(surface="turf_dry"), model)["ligaments"][0]["risk_index"]
    assert turf > grass


def test_slide_tackles_raise_pcl_risk(model):
    def pcl(count):
        report = assess(athlete(slide_tackles=count), model)
        return next(l for l in report["ligaments"] if l["ligament"] == "PCL")["risk_index"]

    assert pcl(6) > pcl(0)


def test_prior_injury_raises_risk(model):
    clean = assess(athlete(prior_injury=False), model)["overall"]["risk_index"]
    injured = assess(athlete(prior_injury=True), model)["overall"]["risk_index"]
    assert injured > clean


def test_model_metrics_are_reported(model):
    for ligament in LIGAMENTS:
        metrics = model.metrics[ligament]
        assert 0.5 < metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 0.25


# --- Bands -----------------------------------------------------------------


@pytest.mark.parametrize(
    "index,expected",
    [(0, "low"), (39.9, "low"), (40, "moderate"), (69.9, "moderate"),
     (70, "high"), (84.9, "high"), (85, "critical"), (100, "critical")],
)
def test_band_thresholds(index, expected):
    assert band_for(index)[0] == expected


def test_every_band_carries_a_status_role():
    for index in (10, 50, 75, 95):
        label, colour, status = band_for(index)
        assert colour.startswith("#")
        assert status in {"good", "warning", "serious", "critical"}


# --- Biomechanical fusion --------------------------------------------------


def test_scan_absent_leaves_score_unchanged(model):
    report = assess(athlete(), model, None)
    for ligament in report["ligaments"]:
        assert ligament["biomechanical_shift"] == 0.0
        assert ligament["adjustments"] == []


def test_clean_scan_does_not_raise_risk(model):
    clean = synthetic_scan("clean_landing")
    baseline = assess(athlete(), model, None)["overall"]["risk_index"]
    scanned = assess(athlete(), model, clean)["overall"]["risk_index"]
    assert scanned == pytest.approx(baseline, abs=0.5)


def test_valgus_collapse_raises_acl_risk(model):
    collapse = synthetic_scan("valgus_collapse")
    baseline = assess(athlete(), model, None)["ligaments"][0]
    scanned = assess(athlete(), model, collapse)["ligaments"][0]
    assert scanned["ligament"] == "ACL"
    assert scanned["risk_index"] > baseline["risk_index"]
    assert scanned["adjustments"]


def test_fusion_is_monotone_across_scenarios(model):
    scores = [
        assess(athlete(), model, synthetic_scan(name))["ligaments"][0]["risk_index"]
        for name in ("clean_landing", "mild_valgus", "valgus_collapse")
    ]
    assert scores == sorted(scores), scores


def test_biomechanical_adjustment_is_capped(model):
    """Even an absurd scan must not swamp the workload evidence entirely."""
    extreme = summarise(
        [frame_metrics(build_pose(knee_offset=0.070), i, i / 30) for i in range(5)],
        source="video",
    )
    report = assess(athlete(), model, extreme)
    acl = report["ligaments"][0]
    total = sum(a["log_odds"] for a in acl["adjustments"])
    assert total > MAX_ADJUSTMENT  # the raw findings exceed the ceiling...

    baseline_probability = assess(athlete(), model, None)["ligaments"][0]["modelled_probability"]
    ceiling = inverse_logit(logit(baseline_probability) + MAX_ADJUSTMENT)
    assert acl["modelled_probability"] <= ceiling + 1e-6  # ...but the fusion clamps


def test_unusable_scan_contributes_nothing(model):
    empty = BiomechanicsReport(
        source="video", frames_analysed=0, peak_valgus_deg=0.0, valgus_side=None,
        landing_flexion_deg=0.0, max_flexion_deg=0.0, kasr=None, trunk_lean_deg=None,
        asymmetry_deg=0.0, confidence=0.0,
    )
    report = assess(athlete(), model, empty)
    for ligament in report["ligaments"]:
        assert ligament["biomechanical_shift"] == 0.0


def test_overall_is_the_worst_ligament_not_the_mean(model):
    report = assess(athlete(soreness=9, surface="turf_dry"), model)
    worst = max(l["risk_index"] for l in report["ligaments"])
    assert report["overall"]["risk_index"] == pytest.approx(worst)


# --- Action plan -----------------------------------------------------------


def test_plan_returns_three_exercises(model):
    report = assess(athlete(soreness=8, minutes_last_7d=500), model)
    assert len(report["action_plan"]) == 3


def test_plan_responds_to_valgus_findings(model):
    report = assess(athlete(), model, synthetic_scan("valgus_collapse"))
    keys = {e["name"] for e in report["action_plan"]}
    assert any("Landing" in name or "Band Walk" in name for name in keys)


def test_plan_responds_to_slide_tackles(model):
    report = assess(athlete(slide_tackles=8, contact_events=9), model)
    names = {e["name"] for e in report["action_plan"]}
    assert "Spanish Squat (banded isometric)" in names


def test_plan_falls_back_when_nothing_is_flagged():
    plan = select_plan(set(), {"acl": 5.0, "mcl": 4.0, "pcl": 2.0})
    assert len(plan) == 3


def test_every_exercise_declares_a_ligament_and_deficit():
    for exercise in LIBRARY:
        assert exercise.ligaments
        assert exercise.addresses
        assert exercise.dose


# --- Demo scenarios --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DEMO_SCENARIOS))
def test_demo_scan_is_labelled_synthetic(name):
    report = synthetic_scan(name)
    assert report.frames_analysed > 0
    assert "DEMO SCAN" in report.notes[0]


def test_demo_scenarios_land_in_their_intended_bands():
    assert synthetic_scan("clean_landing").valgus_flag == "normal"
    assert synthetic_scan("mild_valgus").valgus_flag == "elevated"
    assert synthetic_scan("valgus_collapse").valgus_flag == "high"


def test_unknown_demo_scenario_raises():
    with pytest.raises(ValueError):
        synthetic_scan("nope")


# --- Persistence -----------------------------------------------------------


def test_model_round_trips_through_disk(model, tmp_path):
    path = tmp_path / "bundle.joblib"
    model.save(path)
    reloaded = WorkloadRiskModel.load(path)
    subject = athlete(soreness=6)
    assert reloaded.predict(subject)["acl"].probability == pytest.approx(
        model.predict(subject)["acl"].probability
    )
