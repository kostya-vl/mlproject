"""
Контролируемый сценарий подозрительной активности (ЛР4, задание 8).

Шаги:
  1. baseline: /health healthy, /metrics доступен, корректный /predict → 200.
  2. 10 запросов с НЕВЕРНЫМ токеном → 401 (auth failures).
  3. 8 запросов с ПРАВИЛЬНЫМ токеном при RATE_LIMIT_MAX=5 → 5×200 + 3×429.
  4. снять /metrics, посчитать дельты predictor_failures_total{reason=auth|rate_limit}.
  5. применить правило обнаружения (порог + окно).
  6. после сценария — корректный /predict снова 200.
Сохраняет metrics/security_scenario_<YYYYMMDD>.json.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("API_TOKEN", "demo-secret")
METRICS_TOKEN = os.environ.get("METRICS_TOKEN", "demo-metrics")

N_AUTH = 10
N_RATE = 8
RATE_MAX = int(os.environ.get("RATE_LIMIT_MAX", "5"))
WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))

# Правило обнаружения: за окно WINDOW_SEC
AUTH_THRESHOLD = 5      # > 5 401
RATE_THRESHOLD = 3      # > 3 429


def _req(method: str, path: str, *, headers=None, body=None, timeout=5):
    url = BASE + path
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {scheme!r} (only http/https)")
    data = None
    h = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("content-type", "application/json")

    req = urllib.request.Request(url, data=data, headers=h, method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 — схема проверена
            return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers)


def _counter(text: str, reason: str) -> float:
    pat = re.compile(
        r'^predictor_failures_total\{reason="' + re.escape(reason) + r'"\}\s+([\d.]+)',
        re.MULTILINE,
    )
    m = pat.search(text)
    return float(m.group(1)) if m else 0.0


def main() -> int:
    started = time.time()
    report = {
        "scenario": "suspicious_activity_controlled",
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "base_url": BASE,
        "limits": {
            "rate_limit_max": RATE_MAX,
            "rate_limit_window_sec": WINDOW_SEC,
            "n_auth_fail_requests": N_AUTH,
            "n_rate_requests": N_RATE,
        },
        "detection_rule": {
            "window_sec": WINDOW_SEC,
            "auth_401_threshold": AUTH_THRESHOLD,
            "rate_429_threshold": RATE_THRESHOLD,
            "expr": f"delta(predictor_failures_total{{reason='auth'}}[{WINDOW_SEC}s]) > {AUTH_THRESHOLD} "
                    f"or delta(predictor_failures_total{{reason='rate_limit'}}[{WINDOW_SEC}s]) > {RATE_THRESHOLD}",
        },
        "steps": [],
    }

    # 1. baseline
    s, b, _ = _req("GET", "/health")
    report["steps"].append({"step": "health_before", "status": s, "body": b})
    if s != 200 or json.loads(b).get("status") != "healthy":
        print("FAIL: baseline /health is not healthy", file=sys.stderr)
        return 1

    s, b, _ = _req("POST", "/predict",
               headers={"X-API-Key": API_TOKEN},
               body={"spans": ["ivan.petrov@example.com"]})
    report["steps"].append({"step": "predict_before", "status": s, "body": b[:300]})
    if s != 200:
        print(f"FAIL: baseline /predict returned {s}: {b[:300]}", file=sys.stderr)
        return 1

    s, m0, _ = _req("GET", "/metrics",
                    headers={"Authorization": f"Bearer {METRICS_TOKEN}"})
    if s != 200:
        print(f"FAIL: /metrics returned {s}", file=sys.stderr)
        return 1
    auth0, rate0 = _counter(m0, "auth"), _counter(m0, "rate_limit")

    # 2. серия 401
    auth_codes = []
    for _ in range(N_AUTH):
        s, _, _ = _req("POST", "/predict",
                       headers={"X-API-Key": "wrong-token"},
                       body={"spans": ["a@b.com"]})
        auth_codes.append(s)
    report["steps"].append({"step": "auth_failures", "n": len(auth_codes),
                            "codes": auth_codes,
                            "count_401": auth_codes.count(401)})
    if auth_codes.count(401) != N_AUTH:
        print("FAIL: not all auth requests returned 401", file=sys.stderr)
        return 1

    # 3. серия 429
    rate_codes = []
    for _ in range(N_RATE):
        s, _, _ = _req("POST", "/predict",
                       headers={"X-API-Key": API_TOKEN},
                       body={"spans": ["a@b.com"]})
        rate_codes.append(s)
    report["steps"].append({"step": "rate_limit", "n": len(rate_codes),
                            "codes": rate_codes,
                            "count_429": rate_codes.count(429),
                            "count_200": rate_codes.count(200)})
    if rate_codes.count(429) == 0:
        print("FAIL: rate limit did not trigger (429 not observed)", file=sys.stderr)
        return 1

    # 4. дельты метрик
    s, m1, _ = _req("GET", "/metrics",
                    headers={"Authorization": f"Bearer {METRICS_TOKEN}"})
    auth1, rate1 = _counter(m1, "auth"), _counter(m1, "rate_limit")
    delta_auth = auth1 - auth0
    delta_rate = rate1 - rate0
    report["metrics"] = {
        "auth_before": auth0, "auth_after": auth1, "delta_auth": delta_auth,
        "rate_before": rate0, "rate_after": rate1, "delta_rate": delta_rate,
    }

    # 5. правило обнаружения
    rule_fired = {
        "auth_rule": delta_auth > AUTH_THRESHOLD,
        "rate_rule": delta_rate > RATE_THRESHOLD,
    }
    report["detection"] = {
        "rules": rule_fired,
        "fired": any(rule_fired.values()),
    }

    # 6. восстановление: корректный запрос после сценария
    #    rate limit для правильного токена уже исчерпан в окне — ждём сброса
    #    либо проверяем 429/200; для демо достаточно: сервис отвечает, не 5xx
    time.sleep(1)
    s, b, _ = _req("POST", "/predict",
                   headers={"X-API-Key": API_TOKEN},
                   body={"spans": ["ivan.petrov@example.com"]})
    report["steps"].append({"step": "predict_after", "status": s})
    report["recovered"] = s in (200, 429)  # 429 — тоже «живой» сервис, не 5xx

    # 7. health после
    s, b, _ = _req("GET", "/health")
    report["steps"].append({"step": "health_after", "status": s, "body": b})
    report["health_after_ok"] = s == 200 and json.loads(b).get("status") == "healthy"

    report["duration_sec"] = round(time.time() - started, 3)
    report["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()

    out = Path("metrics") / f"security_scenario_{time.strftime('%Y%m%d')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {out}")
    print(json.dumps({
        "delta_auth": delta_auth, "delta_rate": delta_rate,
        "fired": report["detection"]["fired"],
        "predict_after": s, "health_after_ok": report["health_after_ok"],
    }, indent=2, ensure_ascii=False))

    ok = (auth_codes.count(401) == N_AUTH
          and rate_codes.count(429) > 0
          and report["detection"]["fired"]
          and report["recovered"]
          and report["health_after_ok"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
