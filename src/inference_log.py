import csv
import os
import time
from typing import Any

import pandas as pd


INFERENCE_LOG_FIELDS = [
    "year",
    "RA",
    "Fase",
    "Turma",
    "Idade",
    "Genero",
    "Ano_ingresso",
    "Instituicao_de_ensino",
    "IAA",
    "IEG",
    "IPS",
    "IPP",
    "IDA",
    "IPV",
    "IAN",
    "INDE",
    "Pedra_atual",
    "Pedra_anterior",
    "Atingiu_PV",
    "Indicado",
    "tempo_PM",
    "pred_proba",
    "pred_label",
    "pred_band",
    "risk_threshold",
    "review_threshold",
    "ts",
]


def _ensure_log_schema(path: str, timestamp: int) -> None:
    """Rotate legacy logs with incompatible CSV header."""
    if not os.path.exists(path):
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header == INFERENCE_LOG_FIELDS:
        return

    backup = f"{path}.legacy.{timestamp}"
    os.replace(path, backup)


def append_inference_log(
    path: str,
    payload: dict[str, Any],
    proba: float,
    label: int,
    band: str,
    risk_threshold: float,
    review_threshold: float,
    now_ts: int | None = None,
) -> None:
    timestamp = int(now_ts if now_ts is not None else time.time())
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    _ensure_log_schema(path, timestamp)

    row = {k: payload.get(k) for k in INFERENCE_LOG_FIELDS}
    row["pred_proba"] = proba
    row["pred_label"] = label
    row["pred_band"] = band
    row["risk_threshold"] = risk_threshold
    row["review_threshold"] = review_threshold
    row["ts"] = timestamp
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=INFERENCE_LOG_FIELDS, extrasaction="ignore"
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_inference_log(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            total_data_rows = max(sum(1 for _ in f) - 1, 0)
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        skipped = max(total_data_rows - len(df), 0)
        print(
            f"Warning: malformed inference log detected. "
            f"Skipped {skipped} invalid row(s) from {path}."
        )
        return df
