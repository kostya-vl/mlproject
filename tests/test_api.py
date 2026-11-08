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


def test_health_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"]


def test_predict_ok(client):
    r = client.post("/predict", json={"spans": ["ivan.petrov@example.com"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert {"span", "label", "confidence"} <= set(p)
    assert 0.0 <= p["confidence"] <= 1.0


def test_predict_empty_rejected(client):
    r = client.post("/predict", json={"spans": []})
    assert r.status_code == 422


def test_predict_too_many_rejected(client):
    r = client.post("/predict", json={"spans": ["x"] * 1001})
    assert r.status_code == 422
