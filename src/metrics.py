"""Метрики для span-type классификации."""
from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    """accuracy и macro_f1. Пустой вход → 0.0 без исключений."""
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n_spans": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_spans": len(y_true),
    }
