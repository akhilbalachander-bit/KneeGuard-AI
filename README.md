# KneeGuard AI

Knee ligament injury risk screening for soccer players — ACL, MCL and PCL —
combining **workload modelling** with **computer-vision landing biomechanics**.

Instead of waiting for a tear, an athlete enters a week of match data and
**records a three-second drop-jump straight from the browser**. KneeGuard
measures how the knee actually tracks under load, fuses that with the fatigue
and workload picture, and returns a per-ligament scorecard plus a targeted
prevention plan.

```
        Workload questionnaire      Camera capture / upload / demo
                 │                                │
                 ▼                                ▼
   scikit-learn per-ligament LR        MediaPipe Pose (33 landmarks)
   (ACWR, surface, fatigue,            hip · knee · ankle · shoulder
    contact, prior injury)                        │
                 │                    valgus (FPPA) · KASR · landing
                 │                    flexion · trunk lean · asymmetry
                 └──────────────┬───────────────┘
                                ▼
                  Fusion in log-odds space (capped)
                                │
                 ACL 92  ·  MCL 78  ·  PCL 41
                                │
                                ▼
            Claude explains why + targeted exercise plan
```

## Quick start

```bash
git clone <this repo> && cd KneeGuard-AI
./run.sh                       # installs deps, fetches models, serves on :8000
./run.sh --https               # same, over TLS — needed for phone cameras
```

Then open <http://localhost:8000>. Use `localhost`, not the LAN IP: the browser
only exposes the camera in a secure context (see [The camera tab](#the-camera-tab)).

Manual setup, if you prefer:

```bash
pip install -r requirements.txt
python scripts/fetch_pose_model.py     # ~6 MB MediaPipe bundle
python scripts/train_model.py          # trains + reports holdout metrics
uvicorn kneeguard.api:app --reload
```

Optional — have Claude write the explanations instead of the built-in
rule-based writer:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The app is fully functional without it.

---

## The science

Each ligament fails under a different mechanism, so each gets its own model and
its own set of evidence.

### ACL — Anterior Cruciate Ligament

**Mechanism.** Non-contact pivoting, sudden deceleration, or landing
flat-footed with the knee caving inward (valgus).

**What KneeGuard measures.**

- **Frontal plane projection angle (FPPA)** — the hip→knee→ankle angle projected
  into the image plane, signed by whether the knee deviates *toward* the body
  midline. Past **15°** of inward travel is flagged as elevated ACL shear strain;
  8-15° is flagged as early collapse.
- **Landing knee flexion** — a landing absorbed with under **45°** of flexion
  drives ground reaction force through the joint rather than the posterior chain.
- **Surface traction** — artificial turf produces higher rotational traction
  between cleat and surface, so the shoe holds while the body keeps rotating.
- **Acute:chronic workload ratio, soreness, sleep, sex, prior injury.**

### MCL — Medial Collateral Ligament

**Mechanism.** A direct blow to the outside of the knee, or the knee forced
inward while the cleat stays planted.

**What KneeGuard measures.** Collision and tackle counts, valgus and KASR (the
MCL resists valgus too), and **traction mismatch** — distance from neutral
footing in *either* direction, since both a high-grip dry turf (cleat sticks)
and a low-grip wet pitch (slip, uncontrolled landing) load the medial side.

### PCL — Posterior Cruciate Ligament

**Mechanism.** A direct force to the front of an already-bent knee — a sliding
tackle, or falling hard onto a flexed knee.

**What KneeGuard measures.** Sliding tackle frequency, collision count, and
**deep flexion under load** (past 110°) from the scan.

---

## How the scoring works

### 1. Workload model (`kneeguard/workload_model.py`)

One `StandardScaler → LogisticRegression` pipeline per ligament over 16
features. Logistic regression is deliberate: it stays **monotone** in every risk
factor (reporting *more* soreness can never lower your score), it calibrates
well, and its standardised coefficients give the per-feature attributions shown
in the dashboard's "top workload drivers" panel.

Holdout performance on the synthetic cohort:

| Ligament | ROC AUC | Brier | Cohort event rate |
|----------|---------|-------|-------------------|
| ACL | 0.763 | 0.045 | 5.40% |
| MCL | 0.656 | 0.027 | 2.78% |
| PCL | 0.690 | 0.008 | 0.85% |

### 2. Biomechanics (`kneeguard/biomechanics.py`)

Pure geometry over MediaPipe's 33 landmarks — no ML, so every number is
traceable. For a video, frames are sampled, the **landing frame** is located
from the hip-descent trough, and *peak* (not mean) valgus is reported.

### 3. Fusion (`kneeguard/risk_engine.py`)

Findings are converted to additive **log-odds adjustments** on the workload
probability. That keeps the fusion monotone, keeps the output a valid
probability, caps total scan influence at ±2.2 log-odds so a screening scan can
never entirely swamp the workload evidence, and leaves every adjustment
individually inspectable — the API returns each one with its magnitude.

### 4. Explanation (`kneeguard/explain.py`)

Claude receives **only the already-computed findings** and is instructed to
explain them, never to re-score the athlete or invent a measurement. Without an
API key — or on any API error or refusal — a deterministic rule-based writer
produces the same shape of output, and the response always says which path ran.

---

## Honest limitations

These matter more than the demo, so they are stated plainly rather than buried.

- **The training data is synthetic.** No public per-athlete ligament-injury
  dataset carries these fields, so the cohort is generated from a risk function
  built out of published sports-medicine risk factors. The model turns those
  factors into a calibrated, monotone score with attributions — it did not
  discover anything from real injury data. Swapping in a real cohort means
  replacing `generate_cohort()` and retraining; nothing else changes.
- **The 0-100 number is a percentile, not a probability.** A 78 means "higher
  modelled risk than 78% of athlete-weeks in the reference cohort". Real ACL
  incidence is a few percent per season; any app showing "78% chance of tearing
  your ACL" is lying to you. The raw modelled probability is also returned.
- **Single-camera 2D pose is a screen, not a lab.** FPPA is a validated *2D
  screening* proxy for 3D knee abduction moment, not a measurement of it.
  Marker-based 3D motion capture is the reference standard.
- **Frontal-plane measures need a frontal view.** If the clip is too side-on,
  KneeGuard reports valgus as `unmeasured` and tells you to re-film rather than
  producing a confidently wrong angle.
- **Landing depth needs a landing.** A standing photo shows near-zero knee
  flexion; that is *no landing captured*, not a stiff one, and is scored as such.
- **Sex is included** because ACL incidence in matched sports is 2-8× higher in
  female athletes (Q-angle, hormonal and neuromuscular factors). It is optional
  in the form and defaults to unspecified.
- **This is not a medical device.** It is a training-adjustment aid. Pain,
  swelling, or instability means see a clinician.

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Model readiness and whether Claude is configured |
| `GET` | `/api/reference` | Surfaces, risk bands, demo scenarios, model metrics |
| `POST` | `/api/scan` | Analyse an uploaded photo or clip → `scan_id` + overlay PNG |
| `POST` | `/api/scan/demo/{scenario}` | Synthetic scan for demoing without a camera |
| `POST` | `/api/assess` | Score an athlete, optionally fusing a `scan_id` |

Interactive docs at `/docs`.

```bash
# Scan a clip, then fuse it into an assessment
SCAN=$(curl -s -F "file=@drop_jump.mp4" localhost:8000/api/scan | jq -r .scan_id)

curl -s -X POST localhost:8000/api/assess \
  -H 'content-type: application/json' \
  -d "{\"age\":16,\"sex\":\"female\",\"minutes_last_7d\":330,
       \"minutes_prior_28d\":800,\"consecutive_days\":4,\"surface\":\"turf_dry\",
       \"soreness\":7,\"sleep_hours\":6.5,\"scan_id\":\"$SCAN\"}" | jq
```

Uploaded media is **never written to disk**; scans live in a bounded in-memory
cache only long enough to attach to an assessment.

### Filming guide

Front-on, whole body in frame, decent light. Two to three seconds is plenty:
step off a box (~30 cm) and land, or do a single bodyweight squat to depth.

---

## The camera tab

The movement scan has three inputs: **Camera**, **Upload**, and **Demo**.

The camera tab runs entirely in the browser — `getUserMedia` for the live
preview, `MediaRecorder` for the clip — then posts the captured blob to the same
`/api/scan` endpoint an upload uses. It gives you a framing guide to stand
inside, a 3-2-1 countdown so you have time to get into position, a 3-second
auto-stopping recording with a live timer, and a playback review with
**Retake** / **Analyse this** before anything is sent. A front/rear toggle
covers both filming yourself and filming a teammate. The camera is released the
moment you switch tabs, press *Turn off*, or leave the page.

### The HTTPS catch — read this before demoing on a phone

**Browsers only expose the camera in a secure context.** `http://localhost`
counts as secure; `http://192.168.1.42` does not. So pointing a phone at your
laptop over plain HTTP means `navigator.mediaDevices` is simply *undefined* and
no camera can appear — nothing you can fix in JavaScript.

The app detects this and says so, naming the origin and the fix, rather than
failing silently. To actually use a phone camera:

```bash
./run.sh --https      # generates a self-signed cert, prints the LAN URL
```

Then open the printed `https://<lan-ip>:8000` on the phone and accept the
certificate warning once. Verified end-to-end: over TLS on a non-loopback
address the camera initialises and a capture round-trips to `/api/scan`.

The Upload tab is always available as a fallback — phones let you record with
the native camera app and pick the file, which needs no special context.

### Browser support

`MediaRecorder` output format is negotiated at runtime from
`video/webm;codecs=vp9` → `vp8` → `webm` → `mp4`, so Chrome/Edge/Firefox record
WebM and Safari records MP4. The server decodes both. If recording is
unavailable entirely, **Take photo** still works — a single frame at the bottom
of a squat is a valid (if less informative) scan.

---

## Accounts and saved history (optional)

Sign-in and a per-athlete history of past scorecards, backed by **Supabase**.
Entirely optional — with no Supabase configured, the account controls never
render and the rest of the app is unchanged.

### Setup

1. Create a project at [supabase.com](https://supabase.com) (free tier is fine).
2. **SQL Editor → New query**, paste [`supabase/schema.sql`](supabase/schema.sql),
   and run it. That creates the tables, the indexes, and the Row Level Security
   policies.
3. **Authentication → Providers → Email**: make sure Email is enabled. For a
   hackathon, also turn *off* "Confirm email" so signup is instant — otherwise
   every judge who tries it needs a working inbox.
4. **Project Settings → API**: copy the **Project URL** and the **anon / public**
   key into your `.env`:

   ```bash
   SUPABASE_URL=https://xxxxxxxx.supabase.co
   SUPABASE_ANON_KEY=eyJhbGci...
   ```

5. **Authentication → URL Configuration**: add the address you run the app
   from to **Redirect URLs** — `http://localhost:8000`, your Codespaces
   forwarded URL, and your deployed URL. Sign-up asks Supabase to send people
   back to `window.location.origin`, and Supabase only honours addresses on
   that list.

6. Restart the server. A **Sign in** button appears in the header.

### The confirmation email

If **Confirm email** is on, the link in that email returns to the app rather
than a blank page: the session is established, a **Welcome to KneeGuard AI**
screen explains what to do next, and the auth tokens are stripped out of the
address bar so they cannot be copied out of it or end up in a log.

An expired or already-used link opens the sign-in dialog with an explanation
instead of appearing to do nothing.

### How it is wired

The browser talks to Supabase directly with the Supabase JS client; FastAPI
stays stateless and never sees a password or a token. **Row Level Security is
the entire security model** — every policy in `schema.sql` is scoped to
`auth.uid()`, so a user can only ever read, insert, or delete their own rows.

That is why only the **anon** key is used. It is meant to be public and ships in
the page. The **service_role** key bypasses RLS completely and must never appear
in the frontend, in `.env` here, or in the repo.

### What gets stored — and what does not

Saved: the questionnaire answers, the derived ACWR, the per-ligament indices and
bands, the action plan, the explanation, and the **numeric** summary of the
movement scan (valgus angle, KASR, landing flexion, notes).

**Never saved: the photo or video.** The app does not write uploaded media to
disk, and it does not go to the database either. These are frequently minors
being filmed mid-squat; a screening tool has no business retaining that footage.
A test asserts the payload the browser persists contains no image data.

There is deliberately **no UPDATE policy** on `assessments`: a past assessment
records what was measured at a point in time, and letting it be edited would
make the history untrustworthy.

Note that what *is* stored — age, sex, injury history — is health-adjacent
personal data about minors. If this goes past a hackathon, that deserves a
proper look at consent and retention.

---

## Deploying it

**This is not a static site.** It is a Python server that runs MediaPipe pose
detection and scikit-learn inference on every request. Dropping the `web/`
folder onto GitHub Pages, Netlify drop, or Vercel-static will serve the
dashboard and even let the camera record — then every scan and assessment will
fail, because there is no `/api` behind it. You need a host that runs a
container or a Python process.

Upshot: **push the whole repository and let the host build the `Dockerfile`.**
There is no "files to upload" folder.

One genuine bonus of hosting it: every platform below terminates TLS for you,
so the page is served over HTTPS and the in-browser camera works on phones with
no certificate warnings — the problem `./run.sh --https` exists to solve
locally goes away.

### Why a Dockerfile rather than a plain Python buildpack

MediaPipe's Tasks runtime `dlopen()`s the GLES/EGL client libraries even for
CPU-only inference. On an image without them, the first pose call dies with:

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

`pip install mediapipe` does not pull those in — they are system packages.
Note that `mediapipe` depends on `opencv-contrib-python` (the *full* build), so
`import cv2` needs `libGL.so.1` even though this project asks for headless
OpenCV — both end up installed and the full build wins the import. The
`Dockerfile` installs `libgl1`, `libglib2.0-0`, `libgles2` and `libegl1`, which is the
difference between a working deploy and a container that starts fine and then
500s on the first scan. Most "deploy a Python app" buildpacks give you no way
to add them.

### "Can't I just use GitHub?"

Depends which GitHub product:

| | Runs the Python server? |
|---|---|
| **GitHub Pages** | ❌ Static files only — the dashboard loads, every `/api` call 404s |
| **GitHub Actions** | ⚠️ It's CI. It can *build and deploy* to a host, but hosts nothing itself |
| **GitHub Codespaces** | ✅ Yes — a real Linux container with a public HTTPS URL |

**Codespaces is a genuine option**, and needs no third-party signup.
`.devcontainer/devcontainer.json` is included, so *Code → Codespaces → Create*
installs the system libraries, the Python dependencies, and both models. Then:

```bash
uvicorn kneeguard.api:app --host 0.0.0.0 --port 8000
```

Open the forwarded port 8000 from the **Ports** panel. To use it from a phone,
right-click that port and set **Port Visibility → Public** — the forwarded URL
is HTTPS, so the camera works with no certificate warnings.

Caveats: Codespaces bills against a monthly free allowance and the machine
stops when you disconnect, so it suits development and a quick share rather
than a link that stays up. For a URL that lives on, use a host below.

### Recommended: Hugging Face Spaces (free, no card)

Good fit for a hackathon: free CPU tier with far more RAM than most free
tiers, a public HTTPS URL judges can open, and native Docker support.

1. Create a Space → SDK **Docker** → **Blank**.
2. Push this repository to the Space's git remote.
3. Add this frontmatter to the top of the Space's `README.md`:

   ```yaml
   ---
   title: KneeGuard AI
   sdk: docker
   app_port: 8000
   ---
   ```
4. Optional: add `ANTHROPIC_API_KEY` under Settings → Secrets for
   Claude-written explanations.

### Alternative: Render

`render.yaml` is included, so **New → Blueprint** pointed at the repo picks up
everything. Health check is wired to `/api/health`.

Watch the memory: the free plan caps at 512 MB and MediaPipe + OpenCV +
scikit-learn loaded together sit close to it. If the service restarts mid-scan,
that is what happened — move up a plan.

### Anything else that takes a Dockerfile

Fly.io, Google Cloud Run, Railway, and Azure Container Apps all work from the
same `Dockerfile` with no changes. The container reads `$PORT`, which is what
each of these injects.

### What the image does at build time

Both models are baked in — the ~6 MB MediaPipe bundle is downloaded and the
risk model is trained during `docker build`. The container therefore needs no
network at boot and the first request is not slow.

### Sizing expectations

Analysis is CPU-bound: a 3-second clip is roughly 60–90 frames of pose
inference. On a free shared-CPU tier expect a few seconds per scan, so demo
with short clips. Photo scans are near-instant. Uploads are never written to
disk, and scans live in a bounded in-memory cache, so there is no storage to
provision.

---

## Layout

```
supabase/schema.sql  # tables + Row Level Security policies
kneeguard/
  biomechanics.py    # pure geometry: FPPA, KASR, flexion, trunk lean, view check
  pose_analysis.py   # MediaPipe Tasks wrapper + annotated overlay rendering
  workload_model.py  # synthetic cohort, per-ligament LR, percentile calibration
  risk_engine.py     # log-odds fusion, risk bands, scorecard assembly
  exercises.py       # FIFA 11+ / PEP / Nordic prevention library + selection
  explain.py         # Claude narrative with deterministic fallback
  demo.py            # synthetic drop-jump scans for camera-free demos
  api.py             # FastAPI routes + static hosting
web/                 # dashboard (vanilla HTML/CSS/JS, no build step)
scripts/             # fetch_pose_model.py, train_model.py
tests/               # 88 tests: geometry, model behaviour, fusion, HTTP, WebM
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers the geometry against hand-built poses with known answers, monotonicity of
the risk model in each factor, the fusion cap, band thresholds, WebM decoding
of browser-recorded clips, and the full HTTP surface including malformed uploads
and images with no person in them.

The camera flow itself is verified with Playwright against a fake media device:
photo and clip capture, countdown, recording, submission to `/api/scan`, camera
release on stop, the insecure-origin fallback, and the HTTPS path.

## Prevention exercises

Drawn from the **FIFA 11+** warm-up, the **Santa Monica PEP** program, and the
Nordic hamstring literature — not invented. Each is tagged with the deficit it
addresses, so the plan responds to what the scan and questionnaire actually
found rather than printing a fixed list.
