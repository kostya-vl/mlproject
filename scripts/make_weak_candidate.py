"""
Слабый кандидат: КОНСТАНТНЫЙ предиктор (всегда самый частый класс из train).
Честная оценка на том же test.parquet — для демонстрации сбоя gate.
Сохраняет metrics/candidate_weak.json (baseline не трогает).
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

import pandas as pd
import yaml

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
    ap.add_argument("--train", default="artifacts/train.parquet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pol = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
    test_path = Path(pol["data"]["test_parquet"])
    out_path = Path(args.out or pol["data"]["candidate_weak_file"])

    tr = flatten_spans(pd.read_parquet(args.train))
    te = flatten_spans(pd.read_parquet(test_path))
    if tr.empty or te.empty:
        print("ERROR: пустой train/test", file=sys.stderr)
        return 2

    majority = tr["label"].value_counts().idxmax()
    pred = [majority] * len(te)
    metrics = compute_metrics(te["label"].tolist(), pred)

    baseline = json.loads(Path(pol["data"]["baseline_file"]).read_text(encoding="utf-8"))
    bver = baseline.get("data_version", {})

    cand = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {"type": "constant_predictor", "class": majority,
                  "note": "заведомо слабый кандидат для демонстрации сбоя gate"},
        "eval": {"data": str(test_path), "n_spans": len(te)},
        "metric": {"name": pol["metric"]["name"], "value": float(metrics[pol["metric"]["name"]])},
        "metrics_all": metrics,
        "data_version": {
            "snapshot_hash": bver.get("snapshot_hash"),
            "test_arrow_sha256": bver.get("test_arrow_sha256"),
            "test_parquet_sha256": sha256_of(test_path),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cand, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {out_path}")
    print(f"    constant='{majority}', {pol['metric']['name']} = {cand['metric']['value']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
