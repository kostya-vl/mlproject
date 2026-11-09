import uuid
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REAL = (MODELS_DIR / "lab2_span_tfidf.joblib").exists() and (MODELS_DIR / "SHA256SUMS").exists()

pytestmark = pytest.mark.skipif(not REAL, reason="models/ не найдены (ЛР2 п.10)")

API_TOKEN = "test-token"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(MODELS_DIR))
    monkeypatch.setenv("API_TOKEN", API_TOKEN)
    monkeypatch.setenv("RATE_LIMIT_MAX", "1000")
    from app import ratelimit
    ratelimit.reset()
    from app.app import app
    from fastapi.testclient import TestClient
    with TestClient(app, headers={"X-API-Key": API_TOKEN}) as c:
        yield c
    ratelimit.reset()


def test_health_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_version"]


def test_predict_ok_has_request_id_and_model_version(client):
    r = client.post("/predict", json={"spans": ["ivan.petrov@example.com"]})
    assert r.status_code == 200
    body = r.json()
    assert body.get("request_id")
    assert body["model_version"]
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert {"span", "label", "confidence"} <= set(p)
    assert 0.0 <= p["confidence"] <= 1.0


def test_predict_uses_x_request_id(client):
    rid = "test-" + uuid.uuid4().hex[:8]
    r = client.post("/predict", json={"spans": ["a@b.com"]},
                    headers={"X-Request-ID": rid})
    assert r.status_code == 200
    assert r.json()["request_id"] == rid
    assert r.headers.get("X-Request-ID") == rid


def test_predict_no_token_401(client):
    r = client.post("/predict", json={"spans": ["a@b.com"]},
                    headers={"X-API-Key": ""})
    assert r.status_code == 401


def test_predict_wrong_token_401(client):
    r = client.post("/predict", json={"spans": ["a@b.com"]},
                    headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_predict_empty_rejected(client):
    r = client.post("/predict", json={"spans": []})
    assert r.status_code == 422


def test_predict_too_many_rejected(client):
    r = client.post("/predict", json={"spans": ["x"] * 1001})
    assert r.status_code == 422


def test_health_unhealthy_no_internal_leak(client):
    from app import app as app_module
    saved = dict(app_module.STATE)
    app_module.STATE.update(ready=False, model_version=None,
                            reason="/app/models/x.joblib: SHA256 mismatch secret")
    try:
        r = client.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "unhealthy"
        text = str(body)
        assert "SHA256" not in text
        assert "/app" not in text
    finally:
        app_module.STATE.clear()
        app_module.STATE.update(saved)


def test_predict_wrong_content_type_415(client):
    r = client.post("/predict", data="x", headers={"content-type": "text/plain"})
    assert r.status_code == 415


def test_predict_malformed_json_400(client):
    r = client.post("/predict", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_predict_body_too_large_413(client):
    big = "x" * (2 * 1024 * 1024)
    r = client.post("/predict", content=big.encode(),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_predict_nan_rejected_422(client):
    r = client.post("/predict", json={"spans": [None]})
    assert r.status_code == 422


def test_rate_limit_429(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_MAX", "3")
    from app import ratelimit
    ratelimit.reset()
    for _ in range(3):
        r = client.post("/predict", json={"spans": ["a@b.com"]})
        assert r.status_code == 200
    r = client.post("/predict", json={"spans": ["a@b.com"]})
    assert r.status_code == 429
    assert r.headers.get("Retry-After")

def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/static/app.js" in r.text
    assert API_TOKEN not in r.text  # токен не встраивается в HTML


def test_static_assets_served(client):
    for path in ("/static/app.js", "/static/style.css"):
        r = client.get(path)
        assert r.status_code == 200, path


def test_static_js_does_not_leak_token(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert API_TOKEN not in r.text
    assert "localStorage" in r.text  # упомянут только в комментарии про «не хранить»

