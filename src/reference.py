import json
import numpy as np
import pandas as pd

def compute_reference_stats(X: pd.DataFrame, n_bins: int = 10) -> dict:
    stats = {"columns": {}, "n_rows": int(len(X))}
    for col in X.columns:
        if X[col].dtype == object:
            vc = X[col].fillna("UNKNOWN").value_counts(normalize=True).to_dict()
            stats["columns"][col] = {"type": "categorical", "freq": vc}
        else:
            s = pd.to_numeric(X[col], errors="coerce")
            qs = np.nanquantile(s, np.linspace(0, 1, n_bins + 1))
            # ensure strictly increasing bins
            qs = np.unique(qs)
            stats["columns"][col] = {"type": "numeric", "quantiles": qs.tolist()}
    return stats

def save_reference_stats(stats: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def load_reference_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
