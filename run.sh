#!/usr/bin/env bash
# KneeGuard AI — one-command setup and launch.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8000}"

echo "==> Installing dependencies"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt

echo "==> Fetching the MediaPipe pose model (skipped if already present)"
"$PYTHON" scripts/fetch_pose_model.py

if [ ! -f models/workload_risk_model.joblib ]; then
  echo "==> Training the workload risk model"
  "$PYTHON" scripts/train_model.py
fi

echo
echo "==> KneeGuard AI on http://127.0.0.1:${PORT}"
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "    (ANTHROPIC_API_KEY unset — explanations will use the rule-based writer)"
fi
echo

exec "$PYTHON" -m uvicorn kneeguard.api:app --host 0.0.0.0 --port "$PORT"
