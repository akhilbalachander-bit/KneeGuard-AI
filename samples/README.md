# Sample media

This directory is for local test media and is **not** committed — the media
itself is gitignored so no third-party imagery ships with the repo.

## For the end-to-end pose test

`tests/test_api.py::test_scan_a_real_photo_end_to_end` looks for
`samples/sample_pose.jpg` and **skips cleanly if it is absent**. To run it, drop
in any photo of a person with the full body visible:

```bash
curl -o samples/sample_pose.jpg \
  https://storage.googleapis.com/mediapipe-assets/pose.jpg
```

(That is Google's own MediaPipe example asset. Any full-body photo works.)

Everything else in the suite runs without it — the geometry is tested against
hand-built poses with known answers, and `kneeguard/demo.py` generates synthetic
drop-jump landings that exercise the same code path.

## For trying the app

Record yourself: front-on, whole body in frame, two to three seconds. Step off a
box around 30 cm high and land, or do one bodyweight squat to depth. Drop the
file on the scanner panel.
