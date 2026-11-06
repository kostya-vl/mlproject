"""
Оценка сохранённого артефакта (models/) на фиксированной контрольной выборке.
Та же предобработка, что в ЛР2: flatten_spans + TF-IDF + predict.
Та же метрика: compute_metrics (macro_f1).

Сохраняет metrics/candidate.json — результат ТЕКУЩЕГО запуска.
baseline.json не трогает.
"""
from __future__ import annotations

import os

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from src.metrics import compute_metrics
from src.preprocessing import flatten_spans


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="configs/regression_policy.yaml")
    ap.add_argument("--models", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml
    pol = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
    models_dir = Path(args.models or pol["data"]["models_dir"])
    data_path = Path(args.data or pol["data"]["test_parquet"])
    out_path = Path(args.out or pol["data"]["candidate_file"])
    metric_name = pol["metric"]["name"]

    vec_path = models_dir / "lab2_span_tfidf.joblib"
    mdl_path = models_dir / "lab2_span_logreg.joblib"
    if not vec_path.exists() or not mdl_path.exists():
        print(f"ERROR: нет артефактов в {models_dir}", file=sys.stderr)
        return 2
    if not data_path.exists():
        print(f"ERROR: нет {data_path}", file=sys.stderr)
        return 2

    vec = joblib.load(vec_path)
    model = joblib.load(mdl_path)
    sp = flatten_spans(pd.read_parquet(data_path))
    if sp.empty:
        print(f"ERROR: в {data_path} нет спанов", file=sys.stderr)
        return 2

    pred = model.predict(vec.transform(sp["span"]))
    metrics = compute_metrics(sp["label"], pred)

    if metric_name not in metrics:
        print(f"ERROR: compute_metrics не вернул '{metric_name}': {list(metrics)}", file=sys.stderr)
        return 2

    # data_version — из baseline.json (snapshot_hash) и из test.parquet (sha256)
    baseline = json.loads(Path(pol["data"]["baseline_file"]).read_text(encoding="utf-8"))
    bver = baseline.get("data_version", {})

    candidate = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {"type": "LogisticRegression", "source": str(models_dir)},
        "eval": {"data": str(data_path), "n_spans": len(sp)},
        "metric": {"name": metric_name, "value": float(metrics[metric_name])},
        "metrics_all": metrics,
        "data_version": {
            "snapshot_hash": bver.get("snapshot_hash"),
            "test_arrow_sha256": bver.get("test_arrow_sha256"),
            # фактический sha256 test.parquet — для контроля, что оценивалась та же выборка
            "test_parquet_sha256": sha256_of(data_path),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {out_path}")
    print(f"    {metric_name} = {candidate['metric']['value']:.4f}, n_spans = {candidate['eval']['n_spans']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
