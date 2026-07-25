#!/usr/bin/env bash
# KneeGuard AI — one-command setup and launch.
#
#   ./run.sh            serve over HTTP on localhost
#   ./run.sh --https    serve over HTTPS with a self-signed cert
#
# Why --https matters: browsers only expose getUserMedia in a secure context.
# http://localhost counts as secure, but http://192.168.x.x does not — so to
# point a phone at your laptop and use its camera, you need TLS.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8000}"
USE_HTTPS=0

for arg in "$@"; do
  case "$arg" in
    --https) USE_HTTPS=1 ;;
    --help|-h)
      sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "==> Installing dependencies"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt

echo "==> Fetching the MediaPipe pose model (skipped if already present)"
"$PYTHON" scripts/fetch_pose_model.py

if [ ! -f models/workload_risk_model.joblib ]; then
  echo "==> Training the workload risk model"
  "$PYTHON" scripts/train_model.py
fi

SSL_ARGS=()
SCHEME="http"

if [ "$USE_HTTPS" = "1" ]; then
  CERT_DIR="certs"
  mkdir -p "$CERT_DIR"
  if [ ! -f "$CERT_DIR/key.pem" ] || [ ! -f "$CERT_DIR/cert.pem" ]; then
    if ! command -v openssl >/dev/null 2>&1; then
      echo "openssl is required for --https but was not found." >&2
      exit 1
    fi
    echo "==> Generating a self-signed certificate (valid 365 days)"
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
      -subj "/CN=kneeguard.local" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null
  fi
  SSL_ARGS=(--ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem")
  SCHEME="https"
fi

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"

echo
echo "==> KneeGuard AI on ${SCHEME}://127.0.0.1:${PORT}"
if [ "$USE_HTTPS" = "1" ] && [ -n "$LAN_IP" ]; then
  echo "    From your phone: ${SCHEME}://${LAN_IP}:${PORT}"
  echo "    The certificate is self-signed, so accept the browser warning once."
elif [ -n "$LAN_IP" ]; then
  echo "    Camera capture needs a secure context: use localhost, or re-run"
  echo "    with ./run.sh --https to use a phone at http://${LAN_IP}:${PORT}"
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "    (ANTHROPIC_API_KEY unset — explanations will use the rule-based writer)"
fi
echo

exec "$PYTHON" -m uvicorn kneeguard.api:app \
  --host 0.0.0.0 --port "$PORT" "${SSL_ARGS[@]}"
