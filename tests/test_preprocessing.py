import numpy as np
import pandas as pd

from src.preprocessing import basic_clean, coerce_numeric, normalize_columns


def test_normalize_columns_replaces_spaces_and_symbols():
    df = pd.DataFrame(columns=[" A / B ", "2º Ano", "Turma 1ª"])
    out = normalize_columns(df)
    assert out.columns.tolist() == ["A___B", "2o_Ano", "Turma_1a"]


def test_coerce_numeric_converts_invalid_to_nan():
    df = pd.DataFrame({"x": ["1", "abc", None], "y": ["2.5", "3", "4"]})
    out = coerce_numeric(df, ["x", "y"])
    assert out["x"].tolist()[0] == 1
    assert np.isnan(out["x"].tolist()[1])
    assert out["y"].tolist() == [2.5, 3.0, 4.0]


def test_basic_clean_trims_and_nullifies_string_nans():
    df = pd.DataFrame({" nome ": ["  Ana  ", "None", "nan", "NaT", "  Bia "]})
    out = basic_clean(df)
    assert out.columns.tolist() == ["nome"]
    assert out["nome"].tolist()[0] == "Ana"
    assert pd.isna(out["nome"].tolist()[1])
    assert pd.isna(out["nome"].tolist()[2])
    assert pd.isna(out["nome"].tolist()[3])
    assert out["nome"].tolist()[4] == "Bia"
