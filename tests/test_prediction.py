from pathlib import Path

import pytest

import src.predictor as pred_mod
from src.predictor import PredictorError, predict


@pytest.mark.unit
def test_predict_returns_expected_shape(fake_artifacts):
    """Допустимый вход -> список dict с полями span/label/confidence."""
    out = predict(["a@b.com", "+7 900 111-22-33"])
    assert isinstance(out, list) and len(out) == 2
    for r in out:
        assert set(r) == {"span", "label", "confidence"}
        assert isinstance(r["label"], str)
        assert 0.0 <= r["confidence"] <= 1.0


@pytest.mark.unit
def test_predict_idempotent_with_seed(fake_artifacts):
    """Повторный вызов на том же входе даёт тот же результат."""
    spans = ["a@b.com", "+7 900 111-22-33"]
    r1 = predict(spans)
    r2 = predict(spans)
    assert [(x["label"], x["span"]) for x in r1] == \
           [(x["label"], x["span"]) for x in r2]


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REAL_ARTIFACTS = (MODELS_DIR / "lab2_span_tfidf.joblib").exists() and \
                 (MODELS_DIR / "lab2_span_logreg.joblib").exists()


@pytest.mark.integration
@pytest.mark.skipif(not REAL_ARTIFACTS, reason="models/*.joblib не найдены (ЛР2 п.10)")
def test_predict_with_saved_model_from_disk(monkeypatch):
    """Загрузка СОХРАНЁННОЙ модели из models/ и корректный ответ."""
    monkeypatch.setattr(pred_mod, "_DEFAULT_ART", MODELS_DIR)
    monkeypatch.setattr(pred_mod, "_VECTORIZER", None)
    monkeypatch.setattr(pred_mod, "_MODEL", None)
    monkeypatch.setattr(pred_mod, "_EXPECTED_N_FEATURES", None)

    out = predict(["ivan.petrov@example.com", "+7 900 123-45-67"])
    assert len(out) == 2
    labels = {r["label"] for r in out}
    assert labels <= set(pred_mod._MODEL.classes_)
    assert all(r["confidence"] > 0.3 for r in out)


@pytest.mark.integration
@pytest.mark.skipif(not REAL_ARTIFACTS, reason="models/*.joblib не найдены (ЛР2 п.10)")
def test_predict_with_saved_model_rejects_invalid(monkeypatch):
    """Та же модель, невалидный вход -> PredictorError, модель не перезагружается."""
    monkeypatch.setattr(pred_mod, "_DEFAULT_ART", MODELS_DIR)
    monkeypatch.setattr(pred_mod, "_VECTORIZER", None)
    monkeypatch.setattr(pred_mod, "_MODEL", None)
    monkeypatch.setattr(pred_mod, "_EXPECTED_N_FEATURES", None)
    pred_mod._ensure_model()
    before = id(pred_mod._MODEL)

    with pytest.raises(PredictorError):
        predict(42)

    assert id(pred_mod._MODEL) == before
