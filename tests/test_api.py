import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from api import main as api_main


class DummyModel:
    def __init__(self):
        self.last_columns = None

    def predict_proba(self, X):
        self.last_columns = list(X.columns)
        return np.array([[0.2, 0.8] for _ in range(len(X))], dtype=float)


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api_main, "model", DummyModel())
    with TestClient(api_main.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint_returns_probability_and_logs(monkeypatch):
    model = DummyModel()
    logged = {}

    def fake_append(payload, proba, label, band):
        logged["payload"] = payload
        logged["proba"] = proba
        logged["label"] = label
        logged["band"] = band

    monkeypatch.setattr(api_main, "model", model)
    monkeypatch.setattr(api_main, "_append_inference_log", fake_append)

    payload = {
        "year": 2023,
        "Fase": "ALFA",
        "Turma": "1A",
        "Genero": "M",
        "Instituicao_de_ensino": "Escola X",
        "IAA": 7.2,
    }

    with TestClient(api_main.app) as client:
        resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_label"] == 1
    assert body["risk_probability"] == 0.8
    assert body["risk_band"] == "high_risk"
    assert body["risk_threshold"] == 0.8
    assert body["review_threshold"] == 0.5
    assert "year" not in model.last_columns
    assert logged["payload"]["year"] == 2023
    assert logged["label"] == 1
    assert logged["band"] == "high_risk"


def test_predict_endpoint_applies_custom_threshold_policy(monkeypatch):
    class MediumRiskModel(DummyModel):
        def predict_proba(self, X):
            self.last_columns = list(X.columns)
            return np.array([[0.2, 0.75] for _ in range(len(X))], dtype=float)

    monkeypatch.setenv("RISK_THRESHOLD", "0.9")
    monkeypatch.setenv("REVIEW_THRESHOLD", "0.7")
    monkeypatch.setattr(api_main, "model", MediumRiskModel())

    payload = {
        "year": 2023,
        "Fase": "ALFA",
    }

    with TestClient(api_main.app) as client:
        resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_label"] == 0
    assert body["risk_band"] == "review"
    assert body["risk_threshold"] == 0.9
    assert body["review_threshold"] == 0.7


def test_api_startup_fails_for_invalid_threshold_policy(monkeypatch):
    monkeypatch.setenv("RISK_THRESHOLD", "0.4")
    monkeypatch.setenv("REVIEW_THRESHOLD", "0.7")
    monkeypatch.setattr(api_main, "model", DummyModel())

    with pytest.raises(RuntimeError, match="Invalid threshold policy"):
        with TestClient(api_main.app):
            pass


def test_predict_raises_if_model_not_loaded(monkeypatch):
    monkeypatch.setattr(api_main, "model", None)
    req = api_main.PredictRequest(year=2023)
    with pytest.raises(RuntimeError, match="Model is not loaded"):
        api_main.predict(req)


def test_api_startup_keeps_running_when_model_load_fails(monkeypatch):
    monkeypatch.delenv("RISK_THRESHOLD", raising=False)
    monkeypatch.delenv("REVIEW_THRESHOLD", raising=False)
    monkeypatch.setattr(api_main, "model", None)

    def _broken_load():
        raise RuntimeError("failed to load model artifact")

    monkeypatch.setattr(api_main, "load_model", _broken_load)

    with TestClient(api_main.app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
