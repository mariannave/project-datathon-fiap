import pandas as pd
import pytest

pytest.importorskip("openpyxl")

from src.data_io import load_excel


def test_load_excel_returns_only_expected_sheets(tmp_path):
    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"RA": [1], "Defasagem": [0]}).to_excel(writer, sheet_name="PEDE2023", index=False)
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="OUTRA", index=False)

    out = load_excel(str(path))
    assert list(out.keys()) == ["PEDE2023"]
    assert "RA" in out["PEDE2023"].columns


def test_load_excel_raises_when_no_expected_sheet(tmp_path):
    path = tmp_path / "invalid.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="SHEET1", index=False)

    with pytest.raises(ValueError, match="No expected sheets found"):
        load_excel(str(path))
