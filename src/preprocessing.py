"""Предобработка: нормализация текста и разворачивание документов в спаны."""
from __future__ import annotations

import unicodedata

import pandas as pd

SPAN_COLUMNS = ("span", "label")


def normalize_text(s: str) -> str:
    """NFKC + strip. Пустая строка после нормализации -> пустая строка."""
    return unicodedata.normalize("NFKC", s).strip()


def flatten_spans(df: pd.DataFrame) -> pd.DataFrame:
    """Разворачивает документы в отдельные спаны (span, label)."""
    rows = []
    for _, r in df.iterrows():
        text = r.get("text")
        if text is None:
            text = ""
        starts = r.get("entity_starts")
        if starts is None:
            continue
        try:
            if len(starts) == 0:
                continue
        except TypeError:
            continue
        ends = r.get("entity_ends")
        if ends is None:
            ends = []
        labels = r.get("entity_labels")
        if labels is None:
            labels = []
        for s, e, lbl in zip(starts, ends, labels):
            s = max(0, int(s))
            e = min(len(text), int(e))
            if s >= e:
                continue
            rows.append({"span": text[s:e], "label": lbl})
    return pd.DataFrame(rows, columns=list(SPAN_COLUMNS))
