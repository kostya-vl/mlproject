"""
Создаёт ci_data/train_broken.parquet — КОПИЮ train.parquet с намеренными нарушениями.
Исходный train.parquet не изменяется.
"""
from __future__ import annotations
import os
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

from pathlib import Path
import pandas as pd

SRC = Path("ci_data/train.parquet")
DST = Path("ci_data/train_broken.parquet")

df = pd.read_parquet(SRC).copy()

# Нарушение 1: None в text (policy ЛР2: forbid_none_in)
df.loc[df.index[0], "text"] = None

# Нарушение 2: границы спана за пределами текста
row1 = df.index[1]
if df.loc[row1, "entity_ends"] is None or len(df.loc[row1, "entity_ends"]) == 0:
    df.loc[row1, "entity_starts"] = [[0]]
    df.loc[row1, "entity_ends"] = [[99999]]
else:
    ends = list(df.loc[row1, "entity_ends"])
    ends[0] = 99999
    df.at[row1, "entity_ends"] = ends

# Нарушение 3: почти все спаны заменены на один класс → min_per_class и/или ratio
# Берём половину строк и обнуляем в них entity_starts/ends/labels
half = df.index[len(df) // 2:]
for idx in half:
    df.at[idx, "entity_starts"] = None
    df.at[idx, "entity_ends"] = None
    df.at[idx, "entity_labels"] = None

df.to_parquet(DST, index=False)
print(f"OK: {DST} ({len(df)} строк) — содержит 3 нарушения")
