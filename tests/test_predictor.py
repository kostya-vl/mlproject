import os
import pytest

import src.predictor as pred_mod
from src.predictor import (
    predict, predict_from_file,
    PredictorError, ValidationError, InputTooLargeError, FormatError,
)


def test_positive_correct_input(fake_artifacts):
    result = predict(["a@b.com", "+7 900 111-22-33"])
    assert isinstance(result, list) and len(result) == 2
    for r in result:
        assert set(r) >= {"span", "label", "confidence"}
        assert isinstance(r["label"], str)
        assert 0.0 <= r["confidence"] <= 1.0


def test_negative_wrong_type(model_spy):
    with pytest.raises(ValidationError) as ei:
        predict(42)
    assert ei.value.code == "validation_error"
    assert not model_spy.called


def test_negative_none_and_empty(model_spy):
    with pytest.raises(ValidationError):
        predict(None)
    with pytest.raises(ValidationError):
        predict([])
    assert not model_spy.called


def test_negative_nan_inf(model_spy):
    with pytest.raises(ValidationError):
        predict([float("nan")])
    with pytest.raises(ValidationError):
        predict([float("inf")])
    assert not model_spy.called


def test_negative_too_many_spans(model_spy, monkeypatch):
    monkeypatch.setattr(pred_mod, "MAX_SPANS_PER_REQUEST", 3)
    with pytest.raises(InputTooLargeError) as ei:
        predict(["a", "b", "c", "d"])
    assert ei.value.code == "input_too_large"
    assert not model_spy.called


def test_negative_span_too_long(model_spy, monkeypatch):
    monkeypatch.setattr(pred_mod, "MAX_SPAN_LEN_CHARS", 10)
    with pytest.raises(InputTooLargeError):
        predict(["x" * 11])
    assert not model_spy.called


def test_negative_nul_and_ctrl(model_spy):
    with pytest.raises(ValidationError):
        predict(["ok", "\x00"])
    with pytest.raises(ValidationError):
        predict(["a\x01b"])
    assert not model_spy.called


def test_negative_file_too_large(model_spy, tmp_path, monkeypatch):
    monkeypatch.setattr(pred_mod, "MAX_FILE_BYTES", 100)
    p = tmp_path / "big.txt"
    p.write_bytes(b"a" * 200)
    with pytest.raises(InputTooLargeError) as ei:
        predict_from_file(p)
    assert ei.value.code == "input_too_large"
    assert not model_spy.called


def test_negative_file_binary_magic(model_spy, tmp_path):
    p = tmp_path / "fake.txt"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    with pytest.raises(FormatError) as ei:
        predict_from_file(p)
    assert ei.value.code == "format_error"
    assert not model_spy.called


def test_negative_file_bad_extension(model_spy, tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(FormatError):
        predict_from_file(p)
    assert not model_spy.called


def test_negative_file_bad_encoding(model_spy, tmp_path):
    p = tmp_path / "cp1251.txt"
    p.write_bytes("кириллица".encode("cp1251"))
    with pytest.raises(FormatError):
        predict_from_file(p)
    assert not model_spy.called


def test_negative_file_missing(model_spy, tmp_path):
    with pytest.raises(ValidationError):
        predict_from_file(tmp_path / "does_not_exist.txt")
    assert not model_spy.called
