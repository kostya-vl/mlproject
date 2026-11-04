"""Оценка модели на CI-выборке: загружает модель из artifacts, считает macro-F1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def flatten(df):
    rows = []
    for _, r in df.iterrows():
        t = r.get("text") or ""
        s = r.get("entity_starts")
        if s is None:
            continue
        try:
            if len(s) == 0:
                continue
        except TypeError:
            continue
        for st, en, lb in zip(r["entity_starts"], r["entity_ends"], r["entity_labels"]):
            rows.append({"span": t[st:en], "label": lb})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--data", default="ci_data/test.parquet")
    p.add_argument("--out", default="metrics/ci_metrics.json")
    args, _unknown = p.parse_known_args()

    art = Path(args.artifacts)
    vec_path = art / "lab2_span_tfidf.joblib"
    mdl_path = art / "lab2_span_logreg.joblib"
    if not vec_path.exists() or not mdl_path.exists():
        raise SystemExit(
            f"Нет артефактов в {art}. Сначала выполните:\n"
            f"  python -m src.train --artifacts {art} --seed 42"
        )

    vec = joblib.load(vec_path)
    model = joblib.load(mdl_path)

    sp = flatten(pd.read_parquet(args.data))
    if sp.empty:
        raise SystemExit(f"В {args.data} нет спанов — проверьте файл")

    X = vec.transform(sp["span"])
    pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(sp["label"], pred)),
        "macro_f1": float(f1_score(sp["label"], pred, average="macro", zero_division=0)),
        "n_spans": int(len(sp)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
