"""
In-memory rate limiter: fixed window.

Лимит по умолчанию: 30 запросов / 60 секунд (RATE_LIMIT_MAX / RATE_LIMIT_WINDOW).
Ключ ограничения: sha256(токен)[:16] + IP клиента — токен в памяти/логах
не хранится в открытом виде.
X-Forwarded-For учитывается только если запрос пришёл от TRUSTED_PROXY_IP.
При превышении — 429 + Retry-After.
Ограничение одного процесса: при N workers фактический лимит ≈ N × limit,
при перезапуске состояние сбрасывается (см. README).
"""
from __future__ import annotations

import hashlib
import os
import time

from fastapi import Header, HTTPException, Request

DEFAULT_MAX = 30
DEFAULT_WINDOW_SEC = 60.0

_BUCKETS: dict[str, tuple[float, int]] = {}


def reset() -> None:
    """Сбросить состояние (для тестов)."""
    _BUCKETS.clear()


def _client_ip(request: Request) -> str:
    trusted = os.environ.get("TRUSTED_PROXY_IP")
    if trusted and request.client and request.client.host == trusted:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _key(token: str | None, ip: str) -> str:
    h = hashlib.sha256((token or "").encode("utf-8")).hexdigest()[:16]
    return f"{h}:{ip}"


def rate_limit(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    limit = int(os.environ.get("RATE_LIMIT_MAX", str(DEFAULT_MAX)))
    window = float(os.environ.get("RATE_LIMIT_WINDOW", str(DEFAULT_WINDOW_SEC)))
    key = _key(x_api_key, _client_ip(request))
    now = time.monotonic()
    start, count = _BUCKETS.get(key, (now, 0))
    if now - start > window:
        start, count = now, 0
    count += 1
    _BUCKETS[key] = (start, count)
    if count > limit:
        retry_after = max(1, int(window - (now - start)) + 1)
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
