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

# MediaPipe's Tasks runtime dlopen()s the GLES/EGL client libraries even for
# CPU-only inference. Without these the very first landmarker call dies with
# "OSError: libGLESv2.so.2: cannot open shared object file" — the single most
# common way this image fails to start. libglib2.0-0 is for OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgles2 \
        libegl1 \
        libglib2.0-0 \
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
