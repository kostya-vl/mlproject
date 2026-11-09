"""
API-токен аутентификация.

Секрет — ТОЛЬКО из переменной окружения API_TOKEN (в Git/Dockerfile не хранится).
401 — токен отсутствует или неверный.
503 — сервер не сконфигурирован (API_TOKEN не задан), чтобы не пускать «по умолчанию».
Роли не реализованы → 403 не возвращается (см. README).
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


def require_token(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    expected = os.environ.get("API_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="auth not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        # токен не логируем и не возвращаем в ответе
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API token",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key
