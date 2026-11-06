"""
Сравнение кандидата с baseline по политике (ЛР3, задание 6).

Коды возврата:
  0 — candidate прошёл порог;
  1 — candidate отклонён (метрика хуже, отсутствует, нечисловая, несовместимые данные);
  2 — инфраструктурная ошибка (нет baseline/candidate/policy).

baseline только читается. Результат — metrics/comparison.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


def reject(msg: str, comparison: dict, out: Path) -> int:
    comparison["pass"] = False
    comparison["reason"] = msg
    comparison["generated_at_utc"] = datetime.now(UTC).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"REJECT: {msg}", file=sys.stderr)
    print(f"saved: {out}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="configs/regression_policy.yaml")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pol_path = Path(args.policy)
    if not pol_path.exists():
        print(f"ERROR: нет {pol_path}", file=sys.stderr)
        return 2
    pol = yaml.safe_load(pol_path.read_text(encoding="utf-8"))

    baseline_path = Path(args.baseline or pol["data"]["baseline_file"])
    candidate_path = Path(args.candidate or pol["data"]["candidate_file"])
    out = Path(args.out or pol["data"]["comparison_file"])

    if not baseline_path.exists():
        print(f"ERROR: нет baseline {baseline_path}", file=sys.stderr)
        return 2
    if not candidate_path.exists():
        print(f"ERROR: нет candidate {candidate_path}", file=sys.stderr)
        return 2

    metric = pol["metric"]
    name = metric["name"]
    drop = float(metric["allowed_relative_drop"])
    abs_min = float(metric.get("absolute_min", 0.0))

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    comparison = {
        "metric": name,
        "direction": metric["direction"],
        "allowed_relative_drop": drop,
        "baseline_file": str(baseline_path),
        "candidate_file": str(candidate_path),
    }

    # --- baseline ---
    bm = baseline.get("main_metric") or {}
    if bm.get("name") != name:
        return reject(f"baseline.main_metric.name='{bm.get('name')}' != policy '{name}'",
                      comparison, out)
    bval = bm.get("value")
    if not isinstance(bval, (int, float)) or isinstance(bval, bool) or math.isnan(bval) or math.isinf(bval):
        return reject(f"baseline.main_metric.value не число: {bval!r}", comparison, out)
    comparison["baseline_value"] = float(bval)

    # --- candidate ---
    cm = candidate.get("metric") or {}
    if cm.get("name") != name:
        return reject(f"candidate.metric.name='{cm.get('name')}' != policy '{name}'",
                      comparison, out)
    cval = cm.get("value")
    if not isinstance(cval, (int, float)) or isinstance(cval, bool) or math.isnan(cval) or math.isinf(cval):
        return reject(f"candidate.metric.value не число: {cval!r}", comparison, out)
    comparison["candidate_value"] = float(cval)

    # --- data_version ---
    required = pol.get("data_version", {}).get("required_fields", [])
    bver = baseline.get("data_version") or {}
    cver = candidate.get("data_version") or {}
    for f in required:
        if bver.get(f) != cver.get(f):
            return reject(f"data_version.{f}: baseline={bver.get(f)!r} != candidate={cver.get(f)!r}",
                          comparison, out)
    comparison["data_version"] = {f: bver.get(f) for f in required}

    # --- threshold и решение ---
    if bval == 0:
        threshold = abs_min
        comparison["threshold_mode"] = "absolute_min (baseline == 0)"
    else:
        threshold = bval * (1.0 - drop)
        comparison["threshold_mode"] = f"baseline * (1 - {drop})"
    comparison["threshold"] = float(threshold)
    comparison["pass"] = bool(cval >= threshold)
    comparison["reason"] = (
        f"candidate {cval:.4f} {'>=' if comparison['pass'] else '<'} threshold {threshold:.4f}"
    )
    comparison["generated_at_utc"] = datetime.now(UTC).isoformat()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"baseline {name} = {bval:.4f}")
    print(f"candidate {name} = {cval:.4f}")
    print(f"threshold = {threshold:.4f} ({comparison['threshold_mode']})")
    print(f"{'PASS' if comparison['pass'] else 'FAIL'}: {comparison['reason']}")
    print(f"saved: {out}")
    return 0 if comparison["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
