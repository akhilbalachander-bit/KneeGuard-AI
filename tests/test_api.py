"""End-to-end tests for the HTTP surface."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from kneeguard.api import app
from kneeguard.config import POSE_MODEL_PATH


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["risk_model_ready"] is True


def test_reference_lists_surfaces_and_bands(client):
    body = client.get("/api/reference").json()
    assert {s["value"] for s in body["surfaces"]} == {
        "grass_dry", "grass_wet", "turf_dry", "turf_wet"
    }
    assert [b["label"] for b in body["bands"]] == ["low", "moderate", "high", "critical"]
    assert body["demo_scenarios"]


def test_index_serves_the_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "KneeGuard AI" in response.text


def test_assess_with_defaults(client):
    response = client.post("/api/assess", json={})
    assert response.status_code == 200
    body = response.json()
    assert {l["ligament"] for l in body["ligaments"]} == {"ACL", "MCL", "PCL"}
    assert 0 <= body["overall"]["risk_index"] <= 100
    assert len(body["action_plan"]) == 3
    assert body["disclaimer"]


def test_assess_includes_explanation_by_default(client):
    body = client.post("/api/assess", json={"soreness": 8}).json()
    explanation = body["explanation"]
    assert explanation["generated_by"] in {"claude", "rules"}
    assert explanation["headline"]
    assert explanation["this_week"]


def test_assess_can_skip_explanation(client):
    body = client.post("/api/assess", json={"explain": False}).json()
    assert body["explanation"] is None


def test_assess_rejects_out_of_range_input(client):
    assert client.post("/api/assess", json={"soreness": 42}).status_code == 422
    assert client.post("/api/assess", json={"age": 3}).status_code == 422
    assert client.post("/api/assess", json={"surface": "sand"}).status_code == 422


def test_assess_rejects_unknown_scan_id(client):
    response = client.post("/api/assess", json={"scan_id": "does-not-exist"})
    assert response.status_code == 404


def test_demo_scan_then_assess_fuses_the_findings(client):
    scan = client.post("/api/scan/demo/valgus_collapse").json()
    assert scan["scan"]["valgus_flag"] == "high"

    fused = client.post("/api/assess", json={"scan_id": scan["scan_id"]}).json()
    acl = next(l for l in fused["ligaments"] if l["ligament"] == "ACL")
    assert acl["biomechanical_shift"] > 0
    assert acl["adjustments"]
    assert fused["scan"]["frames_analysed"] > 0


def test_demo_scan_rejects_unknown_scenario(client):
    assert client.post("/api/scan/demo/nope").status_code == 404


def test_scan_rejects_unsupported_file_type(client):
    response = client.post(
        "/api/scan", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert response.status_code in {415, 503}


def test_scan_rejects_empty_upload(client):
    response = client.post(
        "/api/scan", files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    )
    assert response.status_code in {400, 503}


@pytest.mark.skipif(not POSE_MODEL_PATH.exists(), reason="pose model bundle not downloaded")
def test_scan_a_real_photo_end_to_end(client):
    from kneeguard.config import SAMPLES_DIR

    sample = SAMPLES_DIR / "sample_pose.jpg"
    if not sample.exists():
        pytest.skip("sample photo not present")

    with sample.open("rb") as handle:
        response = client.post(
            "/api/scan", files={"file": (sample.name, handle, "image/jpeg")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["scan"]["frames_analysed"] == 1
    assert body["overlay_image"].startswith("data:image/png;base64,")

    fused = client.post("/api/assess", json={"scan_id": body["scan_id"]})
    assert fused.status_code == 200


@pytest.mark.skipif(not POSE_MODEL_PATH.exists(), reason="pose model bundle not downloaded")
def test_scan_of_an_image_without_a_person_is_handled(client):
    """A blank frame must produce an explanatory scan, not a 500."""
    import cv2
    import numpy as np

    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", blank)
    assert ok

    response = client.post(
        "/api/scan", files={"file": ("blank.jpg", io.BytesIO(buffer.tobytes()), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scan"]["frames_analysed"] == 0
    assert body["scan"]["notes"]

    # And it must contribute nothing to the score rather than erroring.
    fused = client.post("/api/assess", json={"scan_id": body["scan_id"]}).json()
    for ligament in fused["ligaments"]:
        assert ligament["biomechanical_shift"] == 0.0
