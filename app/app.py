"""
FastAPI-сервис предсказаний (ЛР4).

- POST /predict: JSON {"spans": ["...", ...]} → {"predictions": [...], "model_version": "...", "request_id": "..."}.
  Та же схема входа и предобработка, что в src.predictor (ЛР2).
- GET /health: 200 + {"status": "healthy", "model_version": "..."} только когда модель загружена;
  503 + {"status": "unhealthy"} иначе. Пути/секреты/детали инфраструктуры в ответ не попадают.
- Модель загружается ОДИН РАЗ при старте; SHA-256 сверяется с models/SHA256SUMS ДО загрузки.
"""
from __future__ import annotations

import os

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import hashlib
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.schemas import (
    HealthResponse,
    Prediction,  # noqa: F401 — реэкспорт для удобства клиентов
    PredictRequest,
    PredictResponse,
)

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import src.predictor as pred_mod  # noqa: E402
from src.predictor import PredictorError, predict  # noqa: E402

MODELS_DIR = Path(os.environ.get("MODELS_DIR", _root / "models"))
CHECKSUMS = MODELS_DIR / "SHA256SUMS"

logger = logging.getLogger("predictor.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

STATE = {"ready": False, "model_version": None, "reason": None}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_and_load() -> str:
    if not CHECKSUMS.exists():
        raise RuntimeError(f"эталон SHA256 не найден: {CHECKSUMS}")
    expected = {}
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1]] = parts[0]

    model_version = None
    for name, exp in expected.items():
        p = MODELS_DIR / name
        if not p.exists():
            raise RuntimeError(f"артефакт отсутствует: {p}")
        actual = _sha256(p)
        if actual != exp:
            raise RuntimeError(f"SHA256 mismatch {name}: {actual[:16]}... != {exp[:16]}...")
        if name.startswith("lab2_span_"):
            model_version = actual[:12]
    if model_version is None:
        raise RuntimeError("в SHA256SUMS нет lab2_span_* артефактов")

    pred_mod._DEFAULT_ART = MODELS_DIR
    pred_mod._VECTORIZER = None
    pred_mod._MODEL = None
    pred_mod._EXPECTED_N_FEATURES = None
    pred_mod._ensure_model()
    return model_version


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: verify+load from %s", MODELS_DIR)
    try:
        version = _verify_and_load()
        STATE.update(ready=True, model_version=version, reason=None)
        logger.info("startup OK: model_version=%s", version)
    except Exception as e:
        STATE.update(ready=False, model_version=None, reason=str(e))
        logger.error("startup FAILED: %s", e)
    yield
    logger.info("shutdown")


app = FastAPI(title="PII span predictor", version="1.0", lifespan=lifespan)


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def _require_ready(request_id: str) -> None:
    if not STATE["ready"]:
        logger.warning("service not ready request_id=%s", request_id)
        # Никаких путей/причин в ответе — только общий статус.
        raise HTTPException(status_code=503, detail="service not ready")


@app.get("/health", response_model=HealthResponse)
def health():
    if not STATE["ready"]:
        # Причина пишется в лог при старте, в ответ не попадает.
        return JSONResponse(status_code=503,
                            content={"status": "unhealthy", "model_version": None})
    return {"status": "healthy", "model_version": STATE["model_version"]}


@app.post("/predict", response_model=PredictResponse)
def do_predict(req: PredictRequest, request: Request, response: Response):
    rid = _request_id(request)
    response.headers["X-Request-ID"] = rid
    _require_ready(rid)
    try:
        preds = predict(req.spans)
    except PredictorError as e:
        logger.warning("predict rejected request_id=%s code=%s msg=%s client=%s",
                       rid, e.code, e, request.client)
        raise HTTPException(
            status_code=422,
            detail={"code": e.code, "message": str(e), "request_id": rid},
            headers={"X-Request-ID": rid},
        ) from e
    logger.info("predict OK request_id=%s n=%d client=%s",
                rid, len(preds), request.client)
    return {
        "predictions": preds,
        "model_version": STATE["model_version"],
        "request_id": rid,
    }


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    rid = request.headers.get("x-request-id", "-")
    logger.warning("http %d %s %s request_id=%s",
                   exc.status_code, request.url.path, exc.detail, rid)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
