import pandas as pd
import numpy as np

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a consistent snake-like style."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("/", "_", regex=False)
        .str.replace("º", "o", regex=False)
        .str.replace("ª", "a", regex=False)
    )
    return out

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: normalize columns, trim strings."""
    out = normalize_columns(df)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip()
            out.loc[out[c].isin(["nan", "None", "NaT"]), c] = np.nan
    return out
