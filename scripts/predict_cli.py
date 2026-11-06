"""
CLI для инференса: python scripts/predict_cli.py "span1" "span2" ...
В контейнере: ENTRYPOINT ["python", "scripts/predict_cli.py"], модели в /app/models.
"""
from __future__ import annotations

import os
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import sys
from pathlib import Path

_here = Path(__file__).resolve()
_root = _here.parent.parent if _here.parent.name == "scripts" else _here.parent
sys.path.insert(0, str(_root))

import argparse
import json

import src.predictor as pred_mod
from src.predictor import PredictorError, predict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spans", nargs="*", help="текстовые спаны")
    ap.add_argument("--models", default="/app/models")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    if not args.spans:
        print("ERROR: передайте хотя бы один спан", file=sys.stderr)
        return 2

    pred_mod._DEFAULT_ART = Path(args.models)
    try:
        out = predict(args.spans, top_k=args.top_k)
    except PredictorError as e:
        print(f"ERROR [{e.code}]: {e}", file=sys.stderr)
        return 1

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
