import pandas as pd
from src.reference import compute_reference_stats, save_reference_stats, load_reference_stats

def test_reference_compute_save_and_load(tmp_path):
    df = pd.DataFrame({
        "num_col": [1, 2, 3, 4, 5],
        "cat_col": ["A", "B", "A", "B", "A"]
    })

    stats = compute_reference_stats(df, n_bins=5)

    out_path = tmp_path / "ref.json"
    save_reference_stats(stats, str(out_path))

    loaded = load_reference_stats(str(out_path))

    assert "columns" in loaded
    assert "num_col" in loaded["columns"]
    assert "cat_col" in loaded["columns"]
    assert loaded["columns"]["num_col"]["type"] == "numeric"
    assert loaded["columns"]["cat_col"]["type"] == "categorical"
