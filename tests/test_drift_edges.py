import json

import pandas as pd

from src.drift import compute_drift


def test_compute_drift_returns_none_row_when_no_common_columns(tmp_path):
    ref = {"columns": {"x": {"type": "numeric", "quantiles": [0, 1, 2]}}}
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref), encoding="utf-8")

    inference = pd.DataFrame({"other": [1, 2, 3]})
    out = compute_drift(str(ref_path), inference)
    assert out.iloc[0]["feature"] == "(none)"
    assert out.iloc[0]["psi"] == 0.0


def test_compute_drift_numeric_with_single_quantile_bin_returns_zero(tmp_path):
    ref = {"columns": {"x": {"type": "numeric", "quantiles": [1.0]}}}
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref), encoding="utf-8")

    inference = pd.DataFrame({"x": [1, 2, 3]})
    out = compute_drift(str(ref_path), inference)
    assert out.iloc[0]["feature"] == "x"
    assert float(out.iloc[0]["psi"]) == 0.0
