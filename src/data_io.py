import pandas as pd

SHEETS = ["PEDE2022", "PEDE2023", "PEDE2024"]

def load_excel(path: str) -> dict[str, pd.DataFrame]:
    """Load PEDE Excel and return dict of DataFrames by sheet name."""
    xls = pd.ExcelFile(path)
    available = [s for s in SHEETS if s in xls.sheet_names]
    if not available:
        raise ValueError(f"No expected sheets found in Excel. Expected one of: {SHEETS}")
    return {s: pd.read_excel(path, sheet_name=s) for s in available}
