"""
ASGI access-log: JSON в stdout, без содержимого входов/токенов/PII.

Поля: ts, event, request_id, method, route (шаблон), status, duration_ms,
model_version, failure_type. request_id — не label метрики, только в логе.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid

from app.metrics import DURATION, FAILURE_REASONS, FAILURES, REQUESTS

logger = logging.getLogger("predictor.access")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
logger.propagate = True


def _log(payload: dict) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False))


class AccessLogMiddleware:
    def __init__(self, app, get_model_version):
        self.app = app
        self.get_model_version = get_model_version

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start = time.perf_counter()
        status = {"code": 500}
        rid = None
        for k, v in scope.get("headers", []):
            if k == b"x-request-id":
                rid = v.decode("latin-1")
                break
        if not rid:
            rid = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = rid

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                # прокинуть request_id в ответ, если ещё нет
                headers = list(message.get("headers", []))
                if not any(h[0].lower() == b"x-request-id" for h in headers):
                    headers.append((b"x-request-id", rid.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            dur_ms = (time.perf_counter() - start) * 1000.0
            route = self._route(scope)
            code = status["code"]
            method = scope.get("method", "?")
            REQUESTS.labels(method=method, route=route, status=str(code)).inc()
            DURATION.labels(method=method, route=route).observe(dur_ms / 1000.0)
            reason = FAILURE_REASONS.get(code)
            if reason:
                FAILURES.labels(reason=reason).inc()
            _log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "request",
                "request_id": rid,
                "method": method,
                "route": route,
                "status": code,
                "duration_ms": round(dur_ms, 2),
                "model_version": self.get_model_version(),
                "failure_type": reason or "none",
            })

    @staticmethod
    def _route(scope) -> str:
        route = scope.get("route")
        p = getattr(route, "path", None)
        if p:
            return p
        path = scope.get("path", "")
        return path if path in ("/health", "/predict", "/metrics", "/") else "unmatched"
