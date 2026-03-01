import pandas as pd
import numpy as np
from .config import STONE_ORDER, CATEGORICAL_DEFAULT


NUMERIC_FEATURES = {
    "Idade", "Ano_ingresso", "IAA", "IEG", "IPS", "IPP", "IDA", "IPV", "IAN", "INDE", "tempo_PM"
}


def _coerce_numeric_feature(series: pd.Series, column: str) -> pd.Series:
    """Coerce numeric-like values and handle Excel date artifacts.

    Some numeric fields may arrive as date-like strings such as
    '1900-01-07 00:00:00' due to spreadsheet formatting. For non-year
    features, these values are interpreted as day-of-month (e.g. 7).
    """
    out = pd.to_numeric(series, errors="coerce")
    needs_fix = out.isna() & series.notna()
    if not needs_fix.any():
        return out

    dt = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    raw = series.loc[needs_fix].astype(str).str.strip()
    date_like = raw.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$", na=False)
    if date_like.any():
        dt.loc[date_like.index[date_like]] = pd.to_datetime(
            raw.loc[date_like], errors="coerce", format="mixed"
        )

    if column == "Ano_ingresso":
        years = dt.dt.year.astype(float)
        min_year = 1990
        max_year = pd.Timestamp.now().year + 1
        years = years.where((years >= min_year) & (years <= max_year), np.nan)
        out.loc[needs_fix] = years.loc[needs_fix]
        return out

    excel_like = dt.notna() & (dt.dt.year == 1900) & (dt.dt.month == 1)
    out.loc[needs_fix & excel_like] = dt.loc[needs_fix & excel_like].dt.day.astype(float)
    return out


def stone_to_ord(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return STONE_ORDER.get(s, np.nan)

def add_derived_features(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add a small set of safe derived features (no leakage).

    Expected columns (if available):
    - Ano_ingresso
    - INDE_22 / INDE_23 in some sheets already
    - Pedra_20, Pedra_21, Pedra_22, Pedra_23
    """
    out = df.copy()

    # time in program
    if "Ano_ingresso" in out.columns:
        out["Ano_ingresso"] = _coerce_numeric_feature(out["Ano_ingresso"], "Ano_ingresso")
        out["tempo_PM"] = year - out["Ano_ingresso"]

    # ordinal stones
    for col in [c for c in out.columns if c.startswith("Pedra_")]:
        out[col + "_ord"] = out[col].map(stone_to_ord)

    # delta stones (most recent - previous if present)
    # try known sequences
    seqs = [("Pedra_22_ord", "Pedra_21_ord"), ("Pedra_23_ord", "Pedra_22_ord")]
    for newer, older in seqs:
        if newer in out.columns and older in out.columns:
            out[f"delta_{newer.replace('_ord','')}"] = out[newer] - out[older]

    # delta INDE (when both years are present in same row, e.g., 2023/2024 sheets)
    if "INDE_23" in out.columns and "INDE_22" in out.columns:
        out["delta_INDE_23_22"] = pd.to_numeric(out["INDE_23"], errors="coerce") - pd.to_numeric(out["INDE_22"], errors="coerce")

    return out

def build_training_pairs(pede_2022: pd.DataFrame, pede_2023: pd.DataFrame, pede_2024: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two datasets:
    - train: features from 2022, target from 2023
    - valid: features from 2023, target from 2024
    Requires the key column RA in all sheets.
    """
    if "RA" not in pede_2022.columns or "RA" not in pede_2023.columns or "RA" not in pede_2024.columns:
        raise ValueError("RA column must exist in all sheets.")

    # Merge to align IDs
    m_train = pede_2022.merge(
        pede_2023[["RA", "Defasagem"]],
        on="RA",
        how="inner",
        suffixes=("", "_y"),
    )
    # PEDE2022 uses 'Defas' name, PEDE2023 uses 'Defasagem'
    # After merge we have target in Defasagem (from 2023)

    m_valid = pede_2023.merge(pede_2024[["RA","Defasagem"]], on="RA", how="inner", suffixes=("", "_y"))

    return m_train, m_valid

def pick_target(df: pd.DataFrame, target_col: str = "Defasagem") -> pd.Series:
    """Binary risk label: 1 if defasagem > 0 else 0."""
    y = pd.to_numeric(df[target_col], errors="coerce")
    return (y > 0).astype(int)

def select_features(df: pd.DataFrame, year: int, mode: str) -> pd.DataFrame:
    """Select a stable set of features.

    mode:
      - '2022': features from PEDE2022 sheet
      - '2023': features from PEDE2023 sheet
    """

    common = [
        "Fase","Turma","Idade","Genero","Ano_ingresso","Instituicao_de_ensino",
        "IAA","IEG","IPS","IPP","IDA","IPV","IAN",
        "INDE",
        "Pedra_atual","Pedra_anterior",
        "Atingiu_PV","Indicado",
        "tempo_PM"
    ]

    out = df.copy()
    out = out.rename(
        columns={
            "Idade_22": "Idade",
            "Gênero": "Genero",
            "Instituição_de_ensino": "Instituicao_de_ensino",
            "Atingiu_PV_": "Atingiu_PV",
            "Atingiu_PV.": "Atingiu_PV",
        }
    )

    # INDE and stone columns differ by sheet/year
    if mode == "2022":
        if "INDE_22" in out.columns:
            out["INDE"] = out["INDE_22"]
        if "Pedra_22" in out.columns:
            out["Pedra_atual"] = out["Pedra_22"]
        if "Pedra_21" in out.columns:
            out["Pedra_anterior"] = out["Pedra_21"]
    elif mode == "2023":
        if "INDE_23" in out.columns:
            out["INDE"] = out["INDE_23"]
        if "Pedra_23" in out.columns:
            out["Pedra_atual"] = out["Pedra_23"]
        if "Pedra_22" in out.columns:
            out["Pedra_anterior"] = out["Pedra_22"]
    else:
        raise ValueError("mode must be '2022' or '2023'")

    if "Fase" in out.columns:
        out["Fase"] = out["Fase"].fillna(CATEGORICAL_DEFAULT).astype(str)

    # Enforce numeric dtypes in stable numeric features
    for col in [c for c in NUMERIC_FEATURES if c in out.columns]:
        out[col] = _coerce_numeric_feature(out[col], col)

    # Fill missing categoricals with default
    for cat in ["Turma","Genero","Instituicao_de_ensino","Pedra_atual","Pedra_anterior"]:
        if cat in out.columns:
            out[cat] = out[cat].fillna(CATEGORICAL_DEFAULT)

    # Some columns may not exist (e.g., IPP in 2022)
    feats = [c for c in common if c in out.columns]
    return out[feats]
