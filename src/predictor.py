"""
Единая точка входа для инференса базовой span-type модели (ЛР2).

Контракт:
  predict(spans) -> list[dict]
  predict_from_file(path) -> list[dict]

Порядок: валидация -> предобработка (TF-IDF) -> проверка формы -> модель.
Любое отклонение входа -> PredictorError (наследники: ValidationError,
InputTooLargeError, FormatError, ModelError).
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ==== ОГРАНИЧЕНИЯ ====
MAX_SPANS_PER_REQUEST = 1_000
MAX_SPAN_LEN_CHARS    = 1_000
MAX_TOTAL_CHARS       = 1_000_000
MAX_FILE_BYTES        = 10 * 1024 * 1024

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BINARY_MAGIC = (
    b"%PDF", b"PK\x03\x04", b"\x89PNG",
    b"\xff\xd8\xff", b"GIF8", b"\x1f\x8b",
)
_DEFAULT_ART = Path(__file__).resolve().parent.parent / "artifacts"


class PredictorError(Exception):
    code = "predictor_error"

class ValidationError(PredictorError):
    code = "validation_error"

class InputTooLargeError(PredictorError):
    code = "input_too_large"

class FormatError(PredictorError):
    code = "format_error"

class ModelError(PredictorError):
    code = "model_error"


_VECTORIZER = None
_MODEL = None
_EXPECTED_N_FEATURES = None


def load_model(artifacts_dir=None):
    art = Path(artifacts_dir) if artifacts_dir is not None else _DEFAULT_ART
    vec_path = art / "lab2_span_tfidf.joblib"
    mdl_path = art / "lab2_span_logreg.joblib"
    if not vec_path.exists() or not mdl_path.exists():
        raise ModelError(
            f"Артефакты не найдены в {art}. "
            f"Запустите lab2_baseline_model.ipynb."
        )
    return joblib.load(vec_path), joblib.load(mdl_path)


def _ensure_model():
    global _VECTORIZER, _MODEL, _EXPECTED_N_FEATURES
    if _MODEL is not None:
        return
    _VECTORIZER, _MODEL = load_model()
    vec_n = _VECTORIZER.transform(["x"]).shape[1]
    model_n = getattr(_MODEL, "n_features_in_", None)
    if model_n is not None and model_n != vec_n:
        raise ModelError(
            f"Несовместимость артефактов: model n_features_in_={model_n}, "
            f"vectorizer n_features={vec_n}."
        )
    _EXPECTED_N_FEATURES = vec_n


def _check_string(s, field, max_len):
    if not isinstance(s, str):
        raise ValidationError(f"{field}: ожидается str, получено {type(s).__name__}")
    if s == "":
        raise ValidationError(f"{field}: пустая строка")
    if "\x00" in s:
        raise ValidationError(f"{field}: содержит NUL-байт")
    if _CTRL_RE.search(s):
        raise ValidationError(f"{field}: содержит управляющие символы")
    if len(s) > max_len:
        raise InputTooLargeError(f"{field}: длина {len(s)} > {max_len}")
    return s


def _normalize(s):
    return unicodedata.normalize("NFKC", s).strip()


def validate_input(spans):
    """Возвращает нормализованный list[str]. NaN/Inf/None -> ValidationError."""
    if isinstance(spans, pd.DataFrame):
        for col in ("span", "text"):
            if col in spans.columns:
                spans = spans[col].tolist()
                break
        else:
            raise ValidationError(
                f"DataFrame: нужна колонка 'span' или 'text', есть {list(spans.columns)}"
            )

    if spans is None:
        raise ValidationError("input: None не допускается")
    if isinstance(spans, str):
        spans = [spans]
    if not isinstance(spans, (list, tuple)):
        raise ValidationError(
            f"input: ожидается list[str] или str, получено {type(spans).__name__}"
        )
    if len(spans) == 0:
        raise ValidationError("input: пустой список")
    if len(spans) > MAX_SPANS_PER_REQUEST:
        raise InputTooLargeError(
            f"input: {len(spans)} спанов > лимита {MAX_SPANS_PER_REQUEST}"
        )

    cleaned = []
    total_chars = 0
    for i, s in enumerate(spans):
        if isinstance(s, float):
            if np.isnan(s):
                raise ValidationError(f"input[{i}]: NaN не допускается")
            if np.isinf(s):
                raise ValidationError(f"input[{i}]: Inf не допускается")
            raise ValidationError(f"input[{i}]: ожидается str, получено float")
        s = _check_string(s, f"input[{i}]", MAX_SPAN_LEN_CHARS)
        s = _normalize(s)
        if not s:
            raise ValidationError(f"input[{i}]: пустая строка после нормализации")
        total_chars += len(s)
        cleaned.append(s)

    if total_chars > MAX_TOTAL_CHARS:
        raise InputTooLargeError(
            f"input: суммарно {total_chars} символов > лимита {MAX_TOTAL_CHARS}"
        )
    return cleaned


def predict(spans, *, top_k=None):
    """Предсказать тип сущности для каждого спана."""
    _ensure_model()
    cleaned = validate_input(spans)

    try:
        X = _VECTORIZER.transform(cleaned)
    except Exception as e:
        raise PredictorError(f"preprocessing: {e}") from e

    if X.shape[0] != len(cleaned):
        raise ModelError(f"shape: {X.shape[0]} != {len(cleaned)}")
    if X.shape[1] != _EXPECTED_N_FEATURES:
        raise ModelError(
            f"shape: {X.shape[1]} признаков != ожидаемых {_EXPECTED_N_FEATURES}"
        )

    try:
        proba = _MODEL.predict_proba(X)
    except Exception as e:
        raise ModelError(f"inference: {e}") from e

    classes = _MODEL.classes_
    out = []
    for i, span in enumerate(cleaned):
        p = proba[i]
        k = len(classes) if top_k is None else min(top_k, len(classes))
        order = np.argsort(p)[::-1][:k]
        preds = [{"label": str(classes[j]), "confidence": float(p[j])} for j in order]
        item = {"span": span, "label": preds[0]["label"],
                "confidence": preds[0]["confidence"]}
        if top_k is not None:
            item["top_k"] = preds
        out.append(item)
    return out


_ALLOWED_EXT = {".txt"}


def predict_from_file(path, *, top_k=None):
    """Читает txt (один спан на строку), валидирует и предсказывает."""
    p = Path(path)

    if not p.exists():
        raise ValidationError(f"file: {p} не существует")
    if not p.is_file():
        raise ValidationError(f"file: {p} — не обычный файл")

    st = p.lstat()
    if st.st_size > MAX_FILE_BYTES:
        raise InputTooLargeError(
            f"file: {st.st_size} байт > лимита {MAX_FILE_BYTES}"
        )
    if p.suffix.lower() not in _ALLOWED_EXT:
        raise FormatError(
            f"file: расширение {p.suffix!r} не разрешено ({sorted(_ALLOWED_EXT)})"
        )

    with open(p, "rb") as f:
        head = f.read(8 * 1024)
        for magic in _BINARY_MAGIC:
            if head.startswith(magic):
                raise FormatError(
                    f"file: бинарная сигнатура {magic!r} — не текст"
                )
        if b"\x00" in head:
            raise FormatError("file: NUL-байт в начале — похоже на бинарник")
        f.seek(0)
        raw = f.read()

    if len(raw) > MAX_FILE_BYTES:
        raise InputTooLargeError(
            f"file: после чтения {len(raw)} байт > лимита {MAX_FILE_BYTES}"
        )

    text = None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise FormatError("file: не удалось декодировать как UTF-8")

    spans = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    return predict(spans, top_k=top_k)
