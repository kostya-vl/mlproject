import sys
from pathlib import Path

import joblib
import pytest

PROJECT = str(Path(__file__).resolve().parent.parent)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import src.predictor as pred_mod


@pytest.fixture
def fake_artifacts(tmp_path, monkeypatch):
    """Маленькие фейковые .joblib + подмена _DEFAULT_ART."""
    spans = ["a@b.com", "c@d.com", "+7 900 111-22-33", "+7 900 444-55-66",
             "123-456-789 00", "987-654-321 00"]
    labels = ["EMAIL", "EMAIL", "PHONE_NUMBER", "PHONE_NUMBER", "SNILS", "SNILS"]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(spans)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, labels)

    joblib.dump(vec, tmp_path / "lab2_span_tfidf.joblib")
    joblib.dump(clf, tmp_path / "lab2_span_logreg.joblib")

    monkeypatch.setattr(pred_mod, "_DEFAULT_ART", tmp_path)
    monkeypatch.setattr(pred_mod, "_VECTORIZER", None)
    monkeypatch.setattr(pred_mod, "_MODEL", None)
    monkeypatch.setattr(pred_mod, "_EXPECTED_N_FEATURES", None)
    yield tmp_path


class ModelSpy:
    def __init__(self):
        self.called = False
    def predict_proba(self, X):
        self.called = True
        raise AssertionError("model.predict_proba must NOT be called on invalid input")
    @property
    def classes_(self):
        return ["EMAIL", "PHONE_NUMBER", "SNILS"]
    n_features_in_ = None


@pytest.fixture
def model_spy(fake_artifacts, monkeypatch):
    pred_mod._ensure_model()
    spy = ModelSpy()
    spy.n_features_in_ = pred_mod._EXPECTED_N_FEATURES
    monkeypatch.setattr(pred_mod, "_MODEL", spy)
    yield spy
    assert not spy.called, "модель была вызвана при невалидном входе"


@pytest.fixture
def small_spans_df():
    """Фиксированный DataFrame: 4 документа, 3 спана, один документ без сущностей."""
    import pandas as pd
    return pd.DataFrame({
        "text": [
            "contact ivan@example.com",
            "phone +7 900 123-45-67",
            "no entities here",
            "snils 123-456-789 00",
        ],
        "entity_starts": [[8], [6], None, [6]],
        "entity_ends":   [[25], [23], None, [20]],
        "entity_labels": [["EMAIL"], ["PHONE_NUMBER"], None, ["SNILS"]],
    })


@pytest.fixture
def known_metrics():
    """Маленький пример с заранее известными метриками.

    y_true: [A, A, B, B, C]   y_pred: [A, B, B, B, C]
    accuracy = 4/5 = 0.8   (совпали позиции 1, 3, 4, 5; ошибка в позиции 2)
    macro_f1 ≈ 0.8222      (A: 0.6667, B: 0.8, C: 1.0 → среднее)
    """
    return {
        "y_true": ["A", "A", "B", "B", "C"],
        "y_pred": ["A", "B", "B", "B", "C"],
        "expected_accuracy": 0.8,
        "expected_macro_f1": 0.8222,
        "tolerance": 0.001,
    }
