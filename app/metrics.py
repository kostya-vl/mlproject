"""
Prometheus-метрики и защита /metrics.

Labels — только низкокардинальные: method, route (шаблон пути), status,
reason (фиксированный набор). request_id, тексты спанов, имена файлов и
реальные пути в labels НЕ используются.
"""
from __future__ import annotations

import ipaddress
import os

from fastapi import HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

REQUESTS = Counter(
    "predictor_http_requests_total",
    "HTTP requests",
    ["method", "route", "status"],
)
DURATION = Histogram(
    "predictor_http_request_duration_seconds",
    "Request duration",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
FAILURES = Counter(
    "predictor_failures_total",
    "Rejected/failed requests by reason",
    ["reason"],
)

FAILURE_REASONS = {
    400: "malformed_json",
    401: "auth",
    413: "body_too_large",
    415: "unsupported_media",
    422: "validation",
    429: "rate_limit",
    503: "not_ready",
    504: "timeout",
}


def _route_template(scope: dict) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path  # шаблон, напр. /predict
    p = scope.get("path", "")
    return p if p in ("/health", "/predict", "/metrics", "/") else "unmatched"


def _allowed(request: Request) -> bool:
    token = os.environ.get("METRICS_TOKEN")
    if token:
        auth = request.headers.get("authorization", "")
        return auth == f"Bearer {token}"
    nets_env = os.environ.get("METRICS_ALLOW_NETS", "127.0.0.1/32,::1/128")
    nets = []
    for part in nets_env.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    if not request.client:
        return False
    try:
        ip = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    return any(ip in n for n in nets)


def metrics_endpoint(request: Request) -> Response:
    if not _allowed(request):
        # 404 вместо 403 — не подтверждаем существование endpoint'а
        raise HTTPException(status_code=404, detail="Not Found")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
