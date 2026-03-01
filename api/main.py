import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, AsyncIterator

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.inference_log import INFERENCE_LOG_FIELDS, append_inference_log

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
INFERENCE_LOG = os.getenv("INFERENCE_LOG", "data/processed/inference_log.csv")
DEFAULT_RISK_THRESHOLD = 0.8
DEFAULT_REVIEW_THRESHOLD = 0.5

model = None
risk_threshold = DEFAULT_RISK_THRESHOLD
review_threshold = DEFAULT_REVIEW_THRESHOLD
model_load_error: Optional[str] = None


def load_decision_policy() -> None:
    """Load and validate thresholds used to convert probability into labels."""
    global risk_threshold, review_threshold

    risk_threshold = float(os.getenv("RISK_THRESHOLD", str(DEFAULT_RISK_THRESHOLD)))
    review_threshold = float(
        os.getenv("REVIEW_THRESHOLD", str(DEFAULT_REVIEW_THRESHOLD))
    )

    if not (0.0 <= review_threshold <= risk_threshold <= 1.0):
        raise RuntimeError(
            "Invalid threshold policy. Expected 0 <= REVIEW_THRESHOLD <= RISK_THRESHOLD <= 1."
        )


def load_model() -> None:
    global model, model_load_error
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train first.")
    try:
        model = joblib.load(MODEL_PATH)
        model_load_error = None
    except Exception as exc:
        model = None
        model_load_error = (
            f"Failed to load model at {MODEL_PATH}. This usually means a "
            f"scikit-learn compatibility issue between training and serving "
            f"environments. Current scikit-learn version: {sklearn.__version__}. "
            "Use the project virtual environment or retrain the model in the "
            "same environment used by the API."
        )
        raise RuntimeError(model_load_error) from exc


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_decision_policy()
    if model is None:
        try:
            load_model()
        except RuntimeError as exc:
            # Keep API process alive for diagnostics (/health) even if model fails to load.
            print(f"Warning: {exc}")
    yield


app = FastAPI(title="PEDE Defasagem Risk API", lifespan=lifespan)


class PredictRequest(BaseModel):
    year: int = Field(
        ..., description="Ano de referência das features (ex.: 2023 para prever 2024)"
    )
    RA: Optional[int] = None
    Fase: Optional[Any] = None
    Turma: Optional[str] = None
    Idade: Optional[float] = None
    Genero: Optional[str] = None
    Ano_ingresso: Optional[int] = None
    Instituicao_de_ensino: Optional[str] = None

    IAA: Optional[float] = None
    IEG: Optional[float] = None
    IPS: Optional[float] = None
    IPP: Optional[float] = None
    IDA: Optional[float] = None
    IPV: Optional[float] = None
    IAN: Optional[float] = None
    INDE: Optional[float] = None

    Pedra_atual: Optional[str] = None
    Pedra_anterior: Optional[str] = None

    Atingiu_PV: Optional[Any] = None
    Indicado: Optional[Any] = None
    tempo_PM: Optional[float] = None


def _append_inference_log(payload: Dict[str, Any], proba: float, label: int, band: str):
    append_inference_log(
        path=INFERENCE_LOG,
        payload=payload,
        proba=proba,
        label=label,
        band=band,
        risk_threshold=risk_threshold,
        review_threshold=review_threshold,
        now_ts=int(time.time()),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        msg = "Model is not loaded."
        if model_load_error:
            msg = f"{msg} {model_load_error}"
        raise RuntimeError(msg)

    payload = req.model_dump()
    df = pd.DataFrame([payload])

    # Drop 'year' from features by default; keep only columns the pipeline expects
    if "year" in df.columns:
        df_feat = df.drop(columns=["year"])
    else:
        df_feat = df

    proba = float(model.predict_proba(df_feat)[:, 1][0])
    label = int(proba >= risk_threshold)
    if proba >= risk_threshold:
        band = "high_risk"
    elif proba >= review_threshold:
        band = "review"
    else:
        band = "low_risk"

    _append_inference_log(payload, proba, label, band)
    return {
        "risk_label": label,
        "risk_probability": proba,
        "risk_band": band,
        "risk_threshold": risk_threshold,
        "review_threshold": review_threshold,
    }
