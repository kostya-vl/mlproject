"""
FastAPI-сервис предсказаний (ЛР4).

- POST /predict: JSON {"spans": [...]} + заголовок X-API-Key.
  Коды: 200 ok, 400 malformed JSON, 401 bad/missing token, 413 body too large,
        415 unsupported media, 422 schema/NaN/лимиты, 429 rate limit,
        503 not ready / auth not configured, 504 timeout.
- GET  /health: 200 {"status":"healthy","model_version"} | 503 {"status":"unhealthy"}.
- GET  /: демо-страница (токен вводится вручную, не встраивается в HTML/JS,
         не в localStorage, не в URL).
Модель загружается ОДИН РАЗ при старте; SHA-256 сверяется до загрузки.
"""
from __future__ import annotations

import os

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import asyncio
import hashlib
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.auth import require_token
from app.limits import (
    MAX_BODY_BYTES,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT_SEC,
)
from app.middleware import BodySizeLimitMiddleware
from app.ratelimit import rate_limit
from app.schemas import HealthResponse, PredictRequest, PredictResponse

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import src.predictor as pred_mod  # noqa: E402
from src.predictor import PredictorError, predict  # noqa: E402

MODELS_DIR = Path(os.environ.get("MODELS_DIR", _root / "models"))
CHECKSUMS = MODELS_DIR / "SHA256SUMS"

logger = logging.getLogger("predictor.api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

STATE = {"ready": False, "model_version": None, "reason": None, "sem": None}

INDEX_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>PII span predictor — demo</title></head>
<body>
<h1>PII span predictor — demo</h1>
<p>Токен вводится вручную и хранится только в памяти вкладки. Он не встраивается
в HTML/JS, не пишется в localStorage и не передаётся в URL.</p>
<label>API token: <input id="tok" type="password" autocomplete="off"></label><br><br>
<label>Spans (по одному на строку):<br>
<textarea id="spans" rows="6" cols="50">ivan.petrov@example.com
+7 900 123-45-678</textarea></label><br>
<button id="go">Predict</button>
<pre id="out"></pre>
<script>
document.getElementById('go').onclick = async () => {
  const tok = document.getElementById('tok').value;   // только в памяти JS
  const spans = document.getElementById('spans').value.split('\\n').filter(Boolean);
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'content-type': 'application/json', 'X-API-Key': tok},
      body: JSON.stringify({spans})
    });
    const body = await r.json();
    document.getElementById('out').textContent = r.status + '\\n' + JSON.stringify(body, null, 2);
  } catch (e) {
    document.getElementById('out').textContent = 'error: ' + e;
  }
};
</script>
</body></html>"""


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
    STATE["sem"] = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
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
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def _require_ready(rid: str) -> None:
    if not STATE["ready"]:
        logger.warning("service not ready request_id=%s", rid)
        raise HTTPException(status_code=503, detail="service not ready")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/health", response_model=HealthResponse)
def health():
    if not STATE["ready"]:
        return JSONResponse(status_code=503,
                            content={"status": "unhealthy", "model_version": None})
    return {"status": "healthy", "model_version": STATE["model_version"]}


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_token), Depends(rate_limit)],
)
async def do_predict(request: Request, response: Response):
    rid = _request_id(request)
    response.headers["X-Request-ID"] = rid
    _require_ready(rid)

    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        logger.warning("unsupported media request_id=%s ctype=%s", rid, ctype)
        raise HTTPException(status_code=415, detail="content-type must be application/json")

    raw = await request.body()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("malformed JSON request_id=%s err=%s", rid, e)
        raise HTTPException(status_code=400, detail="malformed JSON") from e

    try:
        req = PredictRequest.model_validate(payload)
    except Exception as e:
        logger.warning("schema rejected request_id=%s err=%s", rid, e)
        raise HTTPException(status_code=422, detail="invalid request schema") from e

    sem: asyncio.Semaphore = STATE["sem"]
    async with sem:
        try:
            preds = await asyncio.wait_for(
                run_in_threadpool(predict, req.spans),
                timeout=REQUEST_TIMEOUT_SEC,
            )
        except TimeoutError as e:
            logger.warning("timeout request_id=%s", rid)
            raise HTTPException(status_code=504, detail="prediction timeout") from e
        except PredictorError as e:
            logger.warning("predict rejected request_id=%s code=%s msg=%s",
                           rid, e.code, e)
            raise HTTPException(
                status_code=422,
                detail={"code": e.code, "message": str(e), "request_id": rid},
                headers={"X-Request-ID": rid},
            ) from e

    logger.info("predict OK request_id=%s n=%d", rid, len(preds))
    return {"predictions": preds, "model_version": STATE["model_version"],
            "request_id": rid}


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    rid = request.headers.get("x-request-id", "-")
    logger.warning("http %d %s %s request_id=%s",
                   exc.status_code, request.url.path, exc.detail, rid)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,  # сохраняем WWW-Authenticate, Retry-After, X-Request-ID
    )
