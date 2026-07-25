# KneeGuard AI — container image.
#
# This app is a Python server, not a static site: it runs MediaPipe pose
# detection and scikit-learn inference per request. Static hosts (GitHub Pages,
# Netlify drop, Vercel static) can serve web/ but every /api/ call will 404.
#
# Build:  docker build -t kneeguard .
# Run:    docker run -p 8000:8000 kneeguard   ->  http://localhost:8000

# Pinned to bookworm on purpose: the apt package names below differ on newer
# Debian (libglib2.0-0 became libglib2.0-0t64 in trixie), so an unpinned
# `-slim` tag would break the build the day the default moves.
FROM python:3.11-slim-bookworm

# Native libraries pip cannot install, and which the app needs in two places:
#
#   libgl1        libGL.so.1     — mediapipe depends on opencv-contrib-python
#                                  (the full build, not headless), so `import
#                                  cv2` fails without it. This one bites even
#                                  though requirements.txt asks for headless
#                                  OpenCV: both get installed and the full
#                                  build wins the import.
#   libgles2      libGLESv2.so.2 — MediaPipe's Tasks runtime dlopen()s these
#   libegl1       libEGL.so.1      even for CPU-only inference.
#   libglib2.0-0                  — OpenCV runtime dependency.
#
# Miss any of them and the container starts cleanly, then fails on the first
# scan. That is the single most common way this deployment breaks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgles2 \
        libegl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so edits to the app don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake both models into the image: the ~6 MB MediaPipe bundle and the trained
# risk model. Doing this at build time means the container needs no network at
# boot and the first request is not slow.
RUN python scripts/fetch_pose_model.py \
    && python scripts/train_model.py

# Drop privileges — nothing here needs root at runtime.
RUN useradd --create-home --uid 10001 kneeguard \
    && chown -R kneeguard:kneeguard /app
USER kneeguard

# Most platforms (Render, Cloud Run, Hugging Face Spaces, Fly) inject $PORT.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; \
urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health').read()"

CMD ["sh", "-c", "exec uvicorn kneeguard.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
