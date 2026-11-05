import pytest

from src.metrics import compute_metrics


def test_metrics_known_values(known_metrics):
    """Известный пример: accuracy=0.6, macro_f1≈0.8222."""
    m = compute_metrics(known_metrics["y_true"], known_metrics["y_pred"])
    assert m["accuracy"] == pytest.approx(
        known_metrics["expected_accuracy"], abs=known_metrics["tolerance"]
    )
    assert m["macro_f1"] == pytest.approx(
        known_metrics["expected_macro_f1"], abs=known_metrics["tolerance"]
    )
    assert m["n_spans"] == 5


def test_metrics_perfect_prediction():
    """Все верно -> accuracy=1, macro_f1=1."""
    m = compute_metrics(["A", "B", "C"], ["A", "B", "C"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_metrics_all_wrong():
    """Все неверно (метки не пересекаются) -> accuracy=0, macro_f1=0."""
    m = compute_metrics(["A", "B"], ["X", "Y"])
    assert m["accuracy"] == 0.0
    assert m["macro_f1"] == 0.0


def test_metrics_empty_input():
    """Пустой вход -> 0.0 без исключений."""
    m = compute_metrics([], [])
    assert m == {"accuracy": 0.0, "macro_f1": 0.0, "n_spans": 0}
