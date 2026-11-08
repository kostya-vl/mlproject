import uuid
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REAL = (MODELS_DIR / "lab2_span_tfidf.joblib").exists() and (MODELS_DIR / "SHA256SUMS").exists()

pytestmark = pytest.mark.skipif(not REAL, reason="models/ не найдены (ЛР2 п.10)")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(MODELS_DIR))
    from app.app import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


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
    r = client.post("/predict",
                    json={"spans": ["a@b.com"]},
                    headers={"X-Request-ID": rid})
    assert r.status_code == 200
    assert r.json()["request_id"] == rid
    assert r.headers.get("X-Request-ID") == rid


def test_predict_empty_rejected(client):
    r = client.post("/predict", json={"spans": []})
    assert r.status_code == 422


def test_predict_too_many_rejected(client):
    r = client.post("/predict", json={"spans": ["x"] * 1001})
    assert r.status_code == 422


def test_health_unhealthy_no_internal_leak(client):
    """503 при not-ready, но без путей/причин в ответе."""
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
        assert "models" not in text
    finally:
        app_module.STATE.clear()
        app_module.STATE.update(saved)
