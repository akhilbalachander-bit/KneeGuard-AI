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

# Best-effort LAN address, so the script can print a URL a phone can reach.
# macOS has ipconfig(8) and no `hostname -I`; Linux is the other way round.
lan_ip() {
  local candidate
  if command -v ipconfig >/dev/null 2>&1; then
    for iface in en0 en1 en2; do
      candidate="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
      [ -n "$candidate" ] && { echo "$candidate"; return; }
    done
  fi
  candidate="$(hostname -I 2>/dev/null | awk '{print $1}')"
  [ -n "$candidate" ] && { echo "$candidate"; return; }
  if command -v ip >/dev/null 2>&1; then
    ip route get 1.1.1.1 2>/dev/null \
      | awk '{for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }}'
  fi
}

echo "==> Installing dependencies"
# Best-effort: on Debian/Ubuntu (and Codespaces) pip is installed by the system
# package manager and cannot uninstall itself, which under `set -e` would abort
# the whole script before anything useful happened. A stale pip is harmless.
"$PYTHON" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true

if ! "$PYTHON" -m pip install --quiet -r requirements.txt; then
  echo >&2
  echo "Dependency install failed." >&2
  echo >&2
  echo "If Python reported an 'externally-managed-environment' (PEP 668)," >&2
  echo "run this inside a virtualenv:" >&2
  echo >&2
  echo "    python3 -m venv .venv" >&2
  echo "    source .venv/bin/activate" >&2
  echo "    ./run.sh${1:+ $1}" >&2
  echo >&2
  exit 1
fi

echo "==> Fetching the MediaPipe pose model (skipped if already present)"
"$PYTHON" scripts/fetch_pose_model.py

if [ ! -f models/workload_risk_model.joblib ]; then
  echo "==> Training the workload risk model"
  "$PYTHON" scripts/train_model.py
fi

SSL_ARGS=()
SCHEME="http"
LAN_IP="$(lan_ip || true)"

if [ "$USE_HTTPS" = "1" ]; then
  CERT_DIR="certs"
  mkdir -p "$CERT_DIR"

  # The cert must name the address the phone actually types, or the browser
  # reports a name mismatch on top of the self-signed warning and may refuse
  # to let you continue. Re-issue whenever the LAN address has moved.
  SAN="DNS:localhost,IP:127.0.0.1"
  [ -n "$LAN_IP" ] && SAN="${SAN},IP:${LAN_IP}"

  NEEDS_CERT=0
  if [ ! -f "$CERT_DIR/key.pem" ] || [ ! -f "$CERT_DIR/cert.pem" ]; then
    NEEDS_CERT=1
  elif [ -n "$LAN_IP" ] && ! openssl x509 -in "$CERT_DIR/cert.pem" -noout -text 2>/dev/null \
        | grep -q "IP Address:${LAN_IP}\b"; then
    echo "==> LAN address changed to ${LAN_IP}; re-issuing the certificate"
    NEEDS_CERT=1
  fi

  if [ "$NEEDS_CERT" = "1" ]; then
    if ! command -v openssl >/dev/null 2>&1; then
      echo "openssl is required for --https but was not found." >&2
      exit 1
    fi
    echo "==> Generating a self-signed certificate (valid 365 days)"
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
      -subj "/CN=kneeguard.local" \
      -addext "subjectAltName=${SAN}" 2>/dev/null
  fi

  SSL_ARGS=(--ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem")
  SCHEME="https"
fi

echo
echo "  ┌─ KneeGuard AI ──────────────────────────────────────────"

if [ "${CODESPACES:-}" = "true" ]; then
  # In a codespace the LAN address is the container's internal IP, which is
  # unreachable, and --https is pointless because GitHub already terminates
  # TLS on the forwarded URL. Give the instructions that actually apply.
  if [ -n "${CODESPACE_NAME:-}" ] && [ -n "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-}" ]; then
    echo "  │  Open:  https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
  else
    echo "  │  Open the forwarded port ${PORT} from the Ports tab."
  fi
  echo "  │"
  echo "  │  If the Ports tab says 'No forwarded ports', click"
  echo "  │  'Forward a Port' and enter ${PORT}."
  echo "  │"
  echo "  │  For a phone: right-click that port -> Port Visibility -> Public."
  echo "  │  Do NOT use --https here; the forwarded URL is already HTTPS."
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "  │"
    echo "  │  ANTHROPIC_API_KEY unset — explanations use the rule-based writer."
  fi
  echo "  └─────────────────────────────────────────────────────────"
  echo
  exec "$PYTHON" -m uvicorn kneeguard.api:app \
    --host 0.0.0.0 --port "$PORT" "${SSL_ARGS[@]}"
fi

echo "  │  This computer:  ${SCHEME}://localhost:${PORT}"
if [ "$USE_HTTPS" = "1" ]; then
  if [ -n "$LAN_IP" ]; then
    echo "  │  Phone / tablet: ${SCHEME}://${LAN_IP}:${PORT}"
    echo "  │"
    echo "  │  The certificate is self-signed, so the phone will warn once."
    echo "  │  Tap Advanced -> Proceed. Both devices must be on the same Wi-Fi."
  else
    echo "  │  Could not detect a LAN address — find this machine's IP and use"
    echo "  │  ${SCHEME}://<that-ip>:${PORT} on the phone."
  fi
else
  echo "  │"
  echo "  │  Camera capture only works on localhost over plain HTTP."
  if [ -n "$LAN_IP" ]; then
    echo "  │  To use a phone camera at ${LAN_IP}, restart with:  ./run.sh --https"
  else
    echo "  │  To use a phone camera, restart with:  ./run.sh --https"
  fi
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "  │"
  echo "  │  ANTHROPIC_API_KEY unset — explanations use the rule-based writer."
fi
echo "  └─────────────────────────────────────────────────────────"
echo

exec "$PYTHON" -m uvicorn kneeguard.api:app \
  --host 0.0.0.0 --port "$PORT" "${SSL_ARGS[@]}"
