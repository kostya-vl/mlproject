"""
Обучение базовой span-type модели (TF-IDF + LogisticRegression).

CLI:
  python -m src.train --artifacts artifacts --seed 42
Читает artifacts/train.parquet, обучает, сохраняет:
  artifacts/lab2_span_tfidf.joblib
  artifacts/lab2_span_logreg.joblib
  artifacts/lab2_model_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .preprocessing import flatten_spans

DEFAULT_ART = Path(os.path.expanduser("~/mlproject/artifacts"))



def train(train_parquet, out_dir, seed=42):
    df = pd.read_parquet(train_parquet)
    sp = flatten_spans(df)
    if sp.empty:
        raise ValueError(f"Нет спанов в {train_parquet}")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=10_000,
        lowercase=True,
    )
    X = vectorizer.fit_transform(sp["span"])
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    t0 = time.time()
    model.fit(X, sp["label"])
    train_time = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "lab2_span_tfidf.joblib")
    joblib.dump(model, out_dir / "lab2_span_logreg.joblib")

    report = {
        "task": "span-type classification (baseline)",
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "train_time_sec": round(train_time, 3),
        "n_spans": int(len(sp)),
        "n_features": int(X.shape[1]),
        "n_classes": int(sp["label"].nunique()),
    }
    with open(out_dir / "lab2_model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main():
    p = argparse.ArgumentParser(description="Train baseline span-type model")
    p.add_argument("--artifacts", default=str(DEFAULT_ART), help="каталог artifacts/")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    art = Path(args.artifacts)
    report = train(art / "train.parquet", art, seed=args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
