import pandas as pd
import numpy as np
from .reference import load_reference_stats

def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index core formula."""
    expected = np.clip(expected, eps, 1)
    actual = np.clip(actual, eps, 1)
    return float(np.sum((actual - expected) * np.log(actual / expected)))

def _psi_numeric_from_quantiles(qs: list[float], cur_series: pd.Series) -> float:
    """
    PSI numérico usando quantis da referência como bins.
    Para simplificar (aceitável no Datathon), usamos uma distribuição esperada uniforme.
    """
    bins = np.array(qs, dtype=float)
    if len(bins) < 2:
        return 0.0

    cur = pd.to_numeric(cur_series, errors="coerce")
    cur_counts, _ = np.histogram(cur.dropna(), bins=bins)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)

    exp_dist = np.ones_like(cur_dist, dtype=float) / max(len(cur_dist), 1)
    return psi(exp_dist, cur_dist)

def _psi_categorical(ref_freq: dict, cur_series: pd.Series) -> float:
    """PSI categórico comparando frequências."""
    cur = cur_series.fillna("UNKNOWN").astype(str)
    cur_freq = cur.value_counts(normalize=True).to_dict()

    keys = set(ref_freq) | set(cur_freq)
    exp = np.array([ref_freq.get(k, 0.0) for k in keys], dtype=float)
    act = np.array([cur_freq.get(k, 0.0) for k in keys], dtype=float)

    return psi(exp, act)

def compute_drift(reference_path: str, inference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula drift (PSI) por feature, comparando inferência vs estatísticas de referência.

    - Numérico: PSI com bins definidos pelos quantis da referência
    - Categórico: PSI comparando frequências salvas na referência
    """
    ref = load_reference_stats(reference_path)
    rows = []

    for col, info in ref.get("columns", {}).items():
        if col not in inference_df.columns:
            continue

        if info.get("type") == "numeric":
            qs = info.get("quantiles", [])
            val = _psi_numeric_from_quantiles(qs, inference_df[col])
            rows.append({"feature": col, "type": "numeric", "psi": float(val)})
        else:
            freq = info.get("freq", {})
            val = _psi_categorical(freq, inference_df[col])
            rows.append({"feature": col, "type": "categorical", "psi": float(val)})

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame([{"feature": "(none)", "type": "none", "psi": 0.0}])
    return out.sort_values("psi", ascending=False).reset_index(drop=True)

psi_categorical = _psi_categorical