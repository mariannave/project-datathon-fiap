import pandas as pd
import json
from src.drift import compute_drift

def test_compute_drift(tmp_path):
    ref = {
        "columns": {
            "num_col": {
                "type": "numeric",
                "quantiles": [0, 1, 2, 3]
            }
        }
    }

    ref_path = tmp_path / "ref.json"
    with open(ref_path, "w") as f:
        json.dump(ref, f)

    df = pd.DataFrame({"num_col": [1, 2]})

    result = compute_drift(str(ref_path), df)

    assert "psi" in result.columns
