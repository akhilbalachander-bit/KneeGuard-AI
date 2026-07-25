#!/usr/bin/env python3
"""Check every part of the KneeGuard stack and report what is broken.

MediaPipe loads its native libraries lazily — on the first scan, not at
startup — so a broken install looks like a healthy server that fails the
moment you upload something. This runs each layer in order and prints the
real error instead of an HTTP 500.

    python scripts/doctor.py
"""

from __future__ import annotations

import platform
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> bool:
    """Run one check, recording the real exception rather than swallowing it."""
    try:
        detail = fn()
        results.append((PASS, name, detail or ""))
        return True
    except Exception as exc:  # noqa: BLE001 - reporting is the whole point
        results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
        check.failures.append((name, traceback.format_exc()))
        return False


check.failures = []


def main() -> int:
    print("KneeGuard AI — environment check\n")
    print(f"  Python   {sys.version.split()[0]}")
    print(f"  Platform {platform.platform()}\n")

    # --- Python packages ---------------------------------------------------
    def import_numpy():
        import numpy
        return numpy.__version__

    def import_cv2():
        import cv2
        return cv2.__version__

    def import_sklearn():
        import sklearn
        return sklearn.__version__

    def import_mediapipe():
        import mediapipe
        return mediapipe.__version__

    check("import numpy", import_numpy)
    check("import cv2 (OpenCV)", import_cv2)
    check("import sklearn", import_sklearn)
    mediapipe_ok = check("import mediapipe", import_mediapipe)

    # --- Model files -------------------------------------------------------
    def pose_bundle():
        from kneeguard import config
        path = config.POSE_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python scripts/fetch_pose_model.py"
            )
        return f"{path.stat().st_size:,} bytes"

    def risk_model():
        from kneeguard.workload_model import WorkloadRiskModel
        model = WorkloadRiskModel.load_or_train()
        return f"ACL ROC AUC {model.metrics['acl']['roc_auc']:.3f}"

    bundle_ok = check("pose model bundle", pose_bundle)
    check("workload risk model", risk_model)

    def supabase_config():
        from kneeguard import config
        if not config.SUPABASE_URL and not config.SUPABASE_ANON_KEY:
            return "not configured (accounts disabled)"
        if config.SUPABASE_URL_WARNING:
            raise ValueError(config.SUPABASE_URL_WARNING)
        if not config.SUPABASE_URL:
            raise ValueError("SUPABASE_ANON_KEY is set but SUPABASE_URL is empty")
        if not config.SUPABASE_ANON_KEY:
            raise ValueError("SUPABASE_URL is set but SUPABASE_ANON_KEY is empty")
        if not config.SUPABASE_URL.endswith(".supabase.co"):
            raise ValueError(
                f"SUPABASE_URL is {config.SUPABASE_URL!r}; expected "
                "https://<project-ref>.supabase.co"
            )
        key = config.SUPABASE_ANON_KEY
        kind = ("publishable" if key.startswith("sb_publishable_")
                else "legacy anon" if key.startswith("ey")
                else "UNRECOGNISED")
        if key.startswith("sb_secret_") or key.startswith("service_role"):
            raise ValueError(
                "SUPABASE_ANON_KEY holds a SECRET key. That bypasses Row Level "
                "Security and must never reach the browser. Use the Publishable key."
            )
        return f"{config.SUPABASE_URL} ({kind} key)"

    check("supabase configuration", supabase_config)

    # --- The step that actually breaks in containers ------------------------
    # Creating the landmarker is what dlopen()s libGLESv2/libEGL.
    if mediapipe_ok and bundle_ok:
        def native_libraries():
            from kneeguard import pose_analysis
            pose_analysis._get_landmarker()
            return "GLES/EGL loaded"

        def end_to_end_scan():
            import cv2
            import numpy as np
            from kneeguard import pose_analysis
            blank = np.full((480, 640, 3), 200, np.uint8)
            encoded = cv2.imencode(".jpg", blank)[1].tobytes()
            report = pose_analysis.analyse_image(encoded)
            # No person in a blank frame; completing without raising is the point.
            return f"completed, {report.frames_analysed} frames analysed"

        if check("MediaPipe native libraries", native_libraries):
            check("end-to-end image scan", end_to_end_scan)

    # --- Report ------------------------------------------------------------
    print("  " + "-" * 58)
    for status, name, detail in results:
        mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  —  {detail}"
        print(line)
    print("  " + "-" * 58)

    failed = [r for r in results if r[0] == FAIL]
    if not failed:
        print("\n  Everything checks out. If scans still fail, the problem is in")
        print("  the browser or the upload, not the server.\n")
        return 0

    print(f"\n  {len(failed)} check(s) failed.\n")

    # The overwhelmingly common cause gets a named fix.
    blob = " ".join(detail for _, _, detail in failed).lower()
    if "libgl" in blob or "libegl" in blob or "cannot open shared object" in blob:
        print("  MediaPipe's native libraries are missing. Fix with:\n")
        print("      sudo apt-get update")
        print("      sudo apt-get install -y libgl1 libglib2.0-0 libgles2 libegl1\n")
        print("  Then restart the server.\n")

    print("  Full tracebacks:\n")
    for name, tb in check.failures:
        print(f"  --- {name} ---")
        print("\n".join("  " + line for line in tb.strip().splitlines()))
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
