#!/usr/bin/env python3
"""Download the MediaPipe pose landmarker bundle used by the scanner.

The bundle is ~6 MB, so it is fetched at setup time rather than committed.

    python scripts/fetch_pose_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kneeguard import config  # noqa: E402


def main() -> int:
    target = config.POSE_MODEL_PATH
    if target.exists():
        print(f"Pose model already present: {target} ({target.stat().st_size:,} bytes)")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.POSE_MODEL_URL}")
    temporary = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(config.POSE_MODEL_URL, temporary)
    temporary.replace(target)
    print(f"Saved {target} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
