import json

import pandas as pd

from src import drift_cli


def test_drift_cli_main_generates_html_report(tmp_path, monkeypatch):
    reference = {
        "columns": {
            "num_col": {"type": "numeric", "quantiles": [0, 1, 2, 3]},
            "cat_col": {"type": "categorical", "freq": {"A": 0.5, "B": 0.5}},
        }
    }
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(reference), encoding="utf-8")

    inf_path = tmp_path / "inf.csv"
    pd.DataFrame({"num_col": [1, 2], "cat_col": ["A", "A"]}).to_csv(inf_path, index=False)

    out_path = tmp_path / "report.html"
    monkeypatch.setattr(
        "sys.argv",
        [
            "drift_cli",
            "--reference",
            str(ref_path),
            "--inference",
            str(inf_path),
            "--out",
            str(out_path),
        ],
    )

    drift_cli.main()

    html = out_path.read_text(encoding="utf-8")
    assert "Drift Report (PSI)" in html
    assert "Tabela completa" in html


def test_drift_cli_main_skips_malformed_inference_rows(tmp_path, monkeypatch, capsys):
    reference = {
        "columns": {
            "num_col": {"type": "numeric", "quantiles": [0, 1, 2, 3]},
            "cat_col": {"type": "categorical", "freq": {"A": 0.5, "B": 0.5}},
        }
    }
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(reference), encoding="utf-8")

    inf_path = tmp_path / "inf_bad.csv"
    inf_path.write_text(
        "num_col,cat_col\n1,A\n2,A,EXTRA\n3,B\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "report.html"
    monkeypatch.setattr(
        "sys.argv",
        [
            "drift_cli",
            "--reference",
            str(ref_path),
            "--inference",
            str(inf_path),
            "--out",
            str(out_path),
        ],
    )

    drift_cli.main()
    captured = capsys.readouterr()
    assert "Warning: malformed inference log detected." in captured.out
    assert out_path.exists()
