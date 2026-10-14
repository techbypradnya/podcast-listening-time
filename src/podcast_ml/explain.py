"""Model-agnostic explanations: permutation importance on raw input columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import RAW_COLS
from .model import predict_scaled

GROUPS = {c: [c] for c in RAW_COLS if c != "Episode_Title"}
GROUPS["Episode_Number"] = ["Episode_Title"]


def permutation_importance(model, prep, df: pd.DataFrame, y, n_repeats: int = 3, max_rows: int = 15000,
                           seed: int = 0) -> dict:
    """Increase in RMSE (minutes) when a raw column is shuffled. Larger = more important."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype="float64")
    idx = rng.choice(len(df), min(len(df), max_rows), replace=False)
    df, y = df.iloc[idx].reset_index(drop=True), y[idx]

    def rmse(d):
        p = prep.unscale_y(predict_scaled(model, prep.transform(d[RAW_COLS])))
        return float(np.sqrt(np.mean((p - y) ** 2)))

    base = rmse(df)
    out = {}
    for name, cols in GROUPS.items():
        deltas = []
        for _ in range(n_repeats):
            d = df.copy()
            perm = rng.permutation(len(d))
            for c in cols:
                d[c] = d[c].to_numpy()[perm]
            deltas.append(rmse(d) - base)
        out[name] = float(np.mean(deltas))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
