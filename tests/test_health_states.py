"""
/health при проблемах с моделью: не healthy при SHA256 mismatch и ошибке загрузки.
Изолированный TestClient + временный MODELS_DIR. Не трогает настоящие models/.
"""
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")  # в test-and-eval fastapi нет — пропускаем

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REAL = (MODELS_DIR / "lab2_span_tfidf.joblib").exists() and (MODELS_DIR / "SHA256SUMS").exists()
pytestmark = pytest.mark.skipif(not REAL, reason="models/ не найдены (ЛР2 п.10)")


@pytest.fixture
def fresh_client(monkeypatch, tmp_path):
    """Новый TestClient с указанным MODELS_DIR; после теста — восстановление."""
    from app import app as app_module

    import src.predictor as pred_mod

    saved = {
        "MODELS_DIR": app_module.MODELS_DIR,
        "CHECKSUMS": app_module.CHECKSUMS,
        "model": pred_mod._MODEL,
        "vectorizer": pred_mod._VECTORIZER,
        "n_features": pred_mod._EXPECTED_N_FEATURES,
    }

    def _build(models_dir: Path):
        monkeypatch.setattr(app_module, "MODELS_DIR", models_dir)
        monkeypatch.setattr(app_module, "CHECKSUMS", models_dir / "SHA256SUMS")
        monkeypatch.setattr(pred_mod, "_MODEL", None)
        monkeypatch.setattr(pred_mod, "_VECTORIZER", None)
        monkeypatch.setattr(pred_mod, "_EXPECTED_N_FEATURES", None)
        from fastapi.testclient import TestClient
        return TestClient(app_module.app)

    yield _build, tmp_path

    for k, v in saved.items():
        if k == "MODELS_DIR":
            app_module.MODELS_DIR = v
        elif k == "CHECKSUMS":
            app_module.CHECKSUMS = v
        elif k == "model":
            pred_mod._MODEL = v
        elif k == "vectorizer":
            pred_mod._VECTORIZER = v
        elif k == "n_features":
            pred_mod._EXPECTED_N_FEATURES = v


def _copy_models(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in MODELS_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def test_health_unhealthy_on_sha256_mismatch(fresh_client):
    build, tmp = fresh_client
    models = tmp / "models"
    _copy_models(models)
    # испортить эталон
    (models / "SHA256SUMS").write_text(
        "0" * 64 + "  lab2_span_logreg.joblib\n", encoding="utf-8")

    with build(models) as c:
        r = c.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "unhealthy"
        assert body["model_version"] is None
        # никаких путей/причин в ответе
        text = str(body)
        assert "SHA256" not in text
        assert str(models) not in text


def test_health_unhealthy_on_load_error(fresh_client):
    build, tmp = fresh_client
    models = tmp / "models"
    _copy_models(models)
    # удалить файл модели, но оставить запись в SHA256SUMS — load упадёт
    (models / "lab2_span_logreg.joblib").unlink()

    with build(models) as c:
        r = c.get("/health")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "unhealthy"
        assert body["model_version"] is None
        text = str(body)
        assert "FileNotFound" not in text
        assert ".joblib" not in text
