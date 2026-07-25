"""Runtime configuration and shared paths for KneeGuard AI."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
WEB_DIR = ROOT_DIR / "web"
SAMPLES_DIR = ROOT_DIR / "samples"

# Trained scikit-learn workload risk bundle (produced by scripts/train_model.py).
RISK_MODEL_PATH = MODELS_DIR / "workload_risk_model.joblib"

# MediaPipe pose landmarker bundle (fetched by scripts/fetch_pose_model.py).
POSE_MODEL_PATH = Path(
    os.environ.get("KNEEGUARD_POSE_MODEL", MODELS_DIR / "pose_landmarker_lite.task")
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Upload limits — a drop-jump clip only needs a couple of seconds.
MAX_UPLOAD_BYTES = int(os.environ.get("KNEEGUARD_MAX_UPLOAD_BYTES", 40 * 1024 * 1024))
MAX_VIDEO_FRAMES = int(os.environ.get("KNEEGUARD_MAX_VIDEO_FRAMES", 120))

# Claude API (optional). Without a key the app falls back to a deterministic,
# rule-derived narrative so the demo never depends on network access.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.environ.get("KNEEGUARD_CLAUDE_MODEL", "claude-opus-5")

# Supabase (optional). Accounts and saved history switch themselves off when
# these are unset, so the app still runs standalone.
#
# Only the *anon* key belongs here. It is designed to reach the browser and is
# safe there because Row Level Security decides what each user can read. The
# service_role key must never be used by this app: it bypasses RLS entirely.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()


# Cross-origin frontends (e.g. the dashboard on Vercel, this API on Render).
# Comma-separated exact origins. Empty means same-origin only and no CORS
# headers at all — the safe default, since a wide-open API would let any site
# drive someone's camera uploads through it.
CORS_ALLOW_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]


def supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


RANDOM_SEED = 20260725
