"""
Data gate: проверка данных до обучения/оценки.

Возвращает:
  0 — данные валидны;
  1 — нарушение политики (pipeline должен остановиться);
  2 — ошибка инфраструктуры (нет конфига/файла).
"""
from __future__ import annotations

import os

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> int:
    print(f"GATE FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_gate.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--skip-checksum", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: конфиг {cfg_path} не найден", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    data_path = Path(args.data or cfg["data"]["path"])
    if not data_path.exists():
        print(f"ERROR: файл данных {data_path} не найден", file=sys.stderr)
        return 2
    print(f"GATE: config={cfg_path}, data={data_path}")

    if not args.skip_checksum:
        chk_path = Path(cfg["data"]["checksums_file"])
        if not chk_path.exists():
            print(f"ERROR: {chk_path} не найден", file=sys.stderr)
            return 2
        expected = None
        for line in chk_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == data_path.name:
                expected = parts[0]
                break
        if expected is None:
            return fail(f"{data_path.name} не найден в {chk_path}")
        actual = sha256_of(data_path)
        if actual != expected:
            return fail(f"SHA-256 mismatch: ожидалось {expected[:16]}..., получено {actual[:16]}...")
        print(f"GATE OK: SHA-256 совпадает ({actual[:16]}...)")
    else:
        print("GATE: проверка SHA-256 пропущена (--skip-checksum)")

    df = pd.read_parquet(data_path)
    print(f"GATE: строк={len(df)}, колонок={len(df.columns)}")

    required = cfg["required_columns"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return fail(f"отсутствуют обязательные колонки: {missing}")

    type_spec = cfg["column_types"]
    for col, spec in type_spec.items():
        if col == "text":
            bad = df[~df["text"].apply(lambda x: isinstance(x, str))]
            if len(bad):
                return fail(f"колонка 'text': {len(bad)} строк не str (пример: {bad.index[0]})")
        elif spec in ("list[int]", "list[str]"):
            elem_type = (int, np.integer) if "int" in spec else str
            for idx, val in df[col].items():
                if val is None:
                    continue
                if not isinstance(val, (list, tuple, np.ndarray)):
                    return fail(f"колонка '{col}', строка {idx}: ожидался {spec}, получено {type(val).__name__}")
                for e in val:
                    if not isinstance(e, elem_type) or isinstance(e, bool):
                        et_name = (elem_type.__name__ if isinstance(elem_type, type)
                                   else "/".join(t.__name__ for t in elem_type))
                        return fail(f"колонка '{col}', строка {idx}: элемент {e!r} не {et_name}")

    c = cfg["constraints"]
    for col in c.get("forbid_none_in", []):
        n = df[col].isna().sum()
        if n:
            return fail(f"колонка '{col}': {n} None/NaN")
    for col in c.get("forbid_nan_in", []):
        n = df[col].apply(lambda v: isinstance(v, float) and math.isnan(v)).sum()
        if n:
            return fail(f"колонка '{col}': {n} NaN")
    for col in c.get("forbid_inf_in", []):
        n = df[col].apply(lambda v: isinstance(v, float) and math.isinf(v)).sum()
        if n:
            return fail(f"колонка '{col}': {n} Inf")

    spans = []
    for idx, row in df.iterrows():
        text = row["text"]
        if len(text) > c["max_text_len"]:
            return fail(f"строка {idx}: text длиной {len(text)} > {c['max_text_len']}")
        starts, ends, labels = row["entity_starts"], row["entity_ends"], row["entity_labels"]
        if starts is None:
            continue
        if not (len(starts) == len(ends) == len(labels)):
            return fail(f"строка {idx}: длины starts/ends/labels не совпадают ({len(starts)}/{len(ends)}/{len(labels)})")
        for s, e, lbl in zip(starts, ends, labels, strict=False):
            if s < 0 or e > len(text) or s >= e:
                return fail(f"строка {idx}: границы спана [{s},{e}] не влезают в text(len={len(text)})")
            if e - s > c["max_span_len"]:
                return fail(f"строка {idx}: спан длиной {e-s} > {c['max_span_len']}")
            spans.append(lbl)

    if len(df) < c["min_rows"]:
        return fail(f"строк {len(df)} < min_rows={c['min_rows']}")
    if len(spans) < c["min_spans"]:
        return fail(f"спанов {len(spans)} < min_spans={c['min_spans']}")

    counts = pd.Series(spans).value_counts()
    if counts.min() < c["min_per_class"]:
        bad = counts[counts < c["min_per_class"]].to_dict()
        return fail(f"классы с числом < min_per_class={c['min_per_class']}: {bad}")
    ratio = counts.max() / counts.min()
    if ratio > c["max_class_ratio"]:
        return fail(f"max/min по классам = {ratio:.1f} > max_class_ratio={c['max_class_ratio']} "
                    f"(max={counts.max()} '{counts.idxmax()}', min={counts.min()} '{counts.idxmin()}')")

    print(f"GATE OK: строк={len(df)}, спанов={len(spans)}, классов={len(counts)}, "
          f"ratio={ratio:.1f} (max/min), все правила соблюдены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
