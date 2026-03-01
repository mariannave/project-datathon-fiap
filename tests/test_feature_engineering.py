import pandas as pd
import pytest
from src.feature_engineering import add_derived_features, pick_target
from src.feature_engineering import build_training_pairs, select_features

def test_pick_target_binary():
    df = pd.DataFrame({"Defasagem":[-1,0,1,2,None]})
    y = pick_target(df, target_col="Defasagem")
    assert y.tolist() == [0,0,1,1,0]

def test_add_derived_features_tempo_pm():
    df = pd.DataFrame({"Ano_ingresso":[2020,2022,None], "Pedra_22":["Quartzo","Ametista",None], "Pedra_21":["Ágata","Quartzo","Topázio"]})
    out = add_derived_features(df, year=2022)
    assert "tempo_PM" in out.columns
    assert out.loc[0,"tempo_PM"] == 2
    assert "Pedra_22_ord" in out.columns
    assert "delta_Pedra_22" in out.columns


def test_select_features_converts_excel_like_date_artifact_to_numeric():
    df = pd.DataFrame(
        {
            "Fase": ["ALFA"],
            "Turma": ["1A"],
            "Idade": ["1900-01-07 00:00:00"],
            "Genero": ["M"],
            "Ano_ingresso": ["1900-01-07 00:00:00"],
            "Instituicao_de_ensino": ["Escola X"],
            "IAA": [7.0],
            "IEG": [6.0],
            "IPS": [8.0],
            "IDA": [6.0],
            "IPV": [7.0],
            "IAN": [-1],
            "INDE_23": [6.5],
            "Pedra_23": ["Ametista"],
            "Pedra_22": ["Agata"],
            "Atingiu_PV": ["SIM"],
            "Indicado": ["NAO"],
        }
    )

    out = select_features(df, year=2023, mode="2023")
    assert float(out.loc[0, "Idade"]) == 7.0
    assert pd.isna(out.loc[0, "Ano_ingresso"])


def test_select_features_raises_for_invalid_mode():
    with pytest.raises(ValueError, match="mode must be '2022' or '2023'"):
        select_features(pd.DataFrame({"Fase": ["ALFA"]}), year=2023, mode="2024")


def test_build_training_pairs_returns_expected_rows():
    d22 = pd.DataFrame({"RA": [1, 2], "A": [10, 20]})
    d23 = pd.DataFrame({"RA": [1, 2], "Defasagem": [0, 1], "B": [30, 40]})
    d24 = pd.DataFrame({"RA": [1, 2], "Defasagem": [1, 0], "C": [50, 60]})

    m_train, m_valid = build_training_pairs(d22, d23, d24)
    assert list(m_train["RA"]) == [1, 2]
    assert list(m_valid["RA"]) == [1, 2]
    assert "Defasagem" in m_train.columns
    assert "Defasagem_y" in m_valid.columns


def test_build_training_pairs_requires_ra_in_all_inputs():
    d22 = pd.DataFrame({"A": [10]})
    d23 = pd.DataFrame({"RA": [1], "Defasagem": [0]})
    d24 = pd.DataFrame({"RA": [1], "Defasagem": [1]})

    with pytest.raises(ValueError, match="RA column must exist in all sheets"):
        build_training_pairs(d22, d23, d24)
