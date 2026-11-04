"""Оценка модели на CI-выборке: загрузка модели + compute_metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .metrics import compute_metrics
from .preprocessing import flatten_spans


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

    sp = flatten_spans(pd.read_parquet(args.data))
    if sp.empty:
        raise SystemExit(f"В {args.data} нет спанов — проверьте файл")

    X = vec.transform(sp["span"])
    pred = model.predict(X)
    metrics = compute_metrics(sp["label"], pred)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
