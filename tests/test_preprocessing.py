import pandas as pd

from src.preprocessing import flatten_spans, normalize_text


def test_normalize_text_nfkc_and_strip():
    """NFKC: fullwidth 'A' -> 'A'; strip убирает пробелы по краям."""
    assert normalize_text("  Ａ@Ｂ.com  ") == "A@B.com"


def test_normalize_text_empty_stays_empty():
    assert normalize_text("   ") == ""
    assert normalize_text("") == ""


def test_flatten_spans_shape(small_spans_df):
    """Форма результата: ровно колонки span, label; 3 спана из 4 документов."""
    out = flatten_spans(small_spans_df)
    assert list(out.columns) == ["span", "label"]
    assert len(out) == 3
    assert out["span"].tolist() == [
        "ivan@example.com", "+7 900 123-45-67", "123-456-789 00",
    ]
    assert out["label"].tolist() == ["EMAIL", "PHONE_NUMBER", "SNILS"]


def test_flatten_spans_missing_values_skipped(small_spans_df):
    """None в entity_starts — валидное «нет сущностей», документ пропускается."""
    out = flatten_spans(small_spans_df)
    assert "no entities here" not in out["span"].tolist()


def test_flatten_spans_empty_df():
    """Пустой DataFrame -> пустой результат с правильными колонками."""
    empty = pd.DataFrame(
        columns=["text", "entity_starts", "entity_ends", "entity_labels"]
    )
    out = flatten_spans(empty)
    assert len(out) == 0
    assert list(out.columns) == ["span", "label"]


def test_flatten_spans_out_of_bounds_is_clipped():
    """end > len(text) обрезается, start<0 -> 0; спаны с s>=e отбрасываются."""
    df = pd.DataFrame({
        "text": ["abc"],
        "entity_starts": [[0, 5]],
        "entity_ends":   [[10, 2]],
        "entity_labels": [["X", "Y"]],
    })
    out = flatten_spans(df)
    assert out["span"].tolist() == ["abc"]
    assert out["label"].tolist() == ["X"]
