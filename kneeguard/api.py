"""FastAPI application for KneeGuard AI."""

from __future__ import annotations

import logging
import tempfile
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .biomechanics import BiomechanicsReport
from .demo import DEMO_SCENARIOS, synthetic_scan
from .explain import explain
from .risk_engine import RISK_BANDS, assess
from .schemas import AssessmentRequest, HealthResponse, ScanResponse
from .workload_model import SURFACE_LABELS, WorkloadRiskModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kneeguard")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the risk model at boot so the first assessment is not slow."""
    get_model()
    if not config.POSE_MODEL_PATH.exists():
        log.warning(
            "Pose model missing at %s — the scanner will be unavailable. "
            "Run: python scripts/fetch_pose_model.py",
            config.POSE_MODEL_PATH,
        )
    yield


app = FastAPI(
    lifespan=lifespan,
    title="KneeGuard AI",
    version="0.1.0",
    description=(
        "Knee ligament injury risk screening for soccer players, combining workload "
        "modelling with MediaPipe-based landing biomechanics."
    ),
)

# Scans are held in memory only long enough for the client to attach them to an
# assessment. Nothing about the athlete is persisted to disk — a screening tool
# holding onto minors' video would be a liability, not a feature.
_SCANS: "OrderedDict[str, BiomechanicsReport]" = OrderedDict()
_MAX_CACHED_SCANS = 64

_model: WorkloadRiskModel | None = None

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"}


def get_model() -> WorkloadRiskModel:
    global _model
    if _model is None:
        log.info("Loading workload risk model...")
        _model = WorkloadRiskModel.load_or_train()
    return _model


def _remember(report: BiomechanicsReport) -> str:
    scan_id = uuid.uuid4().hex[:12]
    _SCANS[scan_id] = report
    while len(_SCANS) > _MAX_CACHED_SCANS:
        _SCANS.popitem(last=False)
    return scan_id


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=app.version,
        pose_model_ready=config.POSE_MODEL_PATH.exists(),
        risk_model_ready=_model is not None or config.RISK_MODEL_PATH.exists(),
        claude_configured=bool(config.ANTHROPIC_API_KEY),
        claude_model=config.CLAUDE_MODEL,
    )


@app.get("/api/reference")
def reference() -> dict:
    """Static reference data the frontend needs to render its form and legend."""
    model = get_model()
    return {
        "surfaces": [{"value": key, "label": label} for key, label in SURFACE_LABELS.items()],
        "demo_scenarios": [
            {"key": key, "label": scenario["label"], "description": scenario["description"]}
            for key, scenario in DEMO_SCENARIOS.items()
        ],
        "model_metrics": model.metrics,
        # The anon key is meant for the browser; Row Level Security is what
        # protects the data. Accounts stay hidden when this is not configured.
        "supabase": {
            "enabled": config.supabase_configured(),
            "url": config.SUPABASE_URL,
            "anon_key": config.SUPABASE_ANON_KEY,
        },
        "bands": [
            {"label": label, "min": threshold, "colour": colour, "status": status}
            for threshold, label, colour, status in reversed(RISK_BANDS)
        ],
    }


@app.post("/api/scan", response_model=ScanResponse)
async def scan(file: UploadFile = File(...)) -> ScanResponse:
    """Analyse an uploaded photo or short clip for landing mechanics."""
    from . import pose_analysis

    if not pose_analysis.pose_model_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Pose model not installed on the server. "
                "Run: python scripts/fetch_pose_model.py"
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    content_type = (file.content_type or "").lower()
    suffix = Path(file.filename or "").suffix.lower()
    is_video = content_type in VIDEO_TYPES or suffix in {".mp4", ".mov", ".webm", ".avi", ".m4v"}
    is_image = content_type in IMAGE_TYPES or suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if not (is_video or is_image):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {content_type or suffix!r}. Upload a JPEG/PNG photo or an MP4/MOV clip.",
        )

    try:
        if is_video:
            with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4", delete=True) as handle:
                handle.write(data)
                handle.flush()
                report = pose_analysis.analyse_video(Path(handle.name))
        else:
            report = pose_analysis.analyse_image(data)
    except pose_analysis.PoseModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # An unhandled error here reaches the browser as a bare "500" with no
        # body, which tells the user nothing. Log the traceback and hand back
        # something they can act on or quote.
        log.exception("Scan failed for %r", file.filename)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: {type(exc).__name__}: {exc}. "
                "The full traceback is in the server log."
            ),
        ) from exc

    scan_id = _remember(report)
    return ScanResponse(
        scan_id=scan_id,
        scan=report.to_dict(),
        overlay_image=(
            f"data:image/png;base64,{report.overlay_image_b64}"
            if report.overlay_image_b64
            else None
        ),
    )


@app.post("/api/scan/demo/{scenario}", response_model=ScanResponse)
def scan_demo(scenario: str) -> ScanResponse:
    """Load a synthetic movement scan — for demoing without a camera to hand.

    The landmark sequences are generated, not detected, and every response says
    so. They exercise the identical biomechanics and fusion code path a real
    upload takes.
    """
    if scenario not in DEMO_SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown demo scenario. Available: {sorted(DEMO_SCENARIOS)}",
        )
    report = synthetic_scan(scenario)
    scan_id = _remember(report)
    return ScanResponse(scan_id=scan_id, scan=report.to_dict(), overlay_image=None)


@app.post("/api/assess")
def assess_athlete(request: AssessmentRequest) -> JSONResponse:
    """Score an athlete, optionally fusing in a previously uploaded scan."""
    scan_report: BiomechanicsReport | None = None
    if request.scan_id:
        scan_report = _SCANS.get(request.scan_id)
        if scan_report is None:
            raise HTTPException(
                status_code=404,
                detail="Scan not found or expired. Re-upload the clip and try again.",
            )

    try:
        report = assess(request.to_athlete(), get_model(), scan_report)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    report["explanation"] = explain(report) if request.explain else None
    report["disclaimer"] = (
        "KneeGuard AI is a screening and training-adjustment aid, not a medical device "
        "or a diagnosis. The risk index is a percentile against a synthetic reference "
        "cohort, not a probability of injury. See a qualified clinician for pain, "
        "swelling, or instability."
    )
    return JSONResponse(report)


# --- Static frontend --------------------------------------------------------

if config.WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(config.WEB_DIR / "index.html")
