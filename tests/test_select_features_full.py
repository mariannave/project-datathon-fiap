import pandas as pd
from src.feature_engineering import add_derived_features, select_features

def test_select_features_full_including_derived():
    df = pd.DataFrame({
        "Fase": ["ALFA"],
        "Turma": ["1A"],
        "Idade": [14],
        "Genero": ["M"],
        "Ano_ingresso": [2021],
        "Instituicao_de_ensino": ["Escola X"],
        "IAA": [7.0],
        "IEG": [6.0],
        "IPS": [8.0],
        "IPP": [7.0],
        "IDA": [6.0],
        "IPV": [7.0],
        "IAN": [-1],
        "INDE": [6.5],
        "Pedra_atual": ["Ametista"],
        "Pedra_anterior": ["Agata"],
        "Atingiu_PV": ["SIM"],
        "Indicado": ["NAO"]
    })

    # Derived features happen here
    df2 = add_derived_features(df, year=2023)

    result = select_features(df2, year=2023, mode="2023")

    assert "Fase" in result.columns
    assert "tempo_PM" in result.columns
