import numpy as np
import pytest
import pandas as pd

from api import main as api_main


class _DummyModel:
    def __init__(self, p1):
        self.p1 = p1

    def predict_proba(self, X):
        return np.array([[1 - self.p1, self.p1]], dtype=float)


def test_load_decision_policy_defaults(monkeypatch):
    monkeypatch.delenv("RISK_THRESHOLD", raising=False)
    monkeypatch.delenv("REVIEW_THRESHOLD", raising=False)
    api_main.load_decision_policy()
    assert api_main.risk_threshold == api_main.DEFAULT_RISK_THRESHOLD
    assert api_main.review_threshold == api_main.DEFAULT_REVIEW_THRESHOLD


def test_load_decision_policy_raises_for_invalid_order(monkeypatch):
    monkeypatch.setenv("RISK_THRESHOLD", "0.4")
    monkeypatch.setenv("REVIEW_THRESHOLD", "0.7")
    with pytest.raises(RuntimeError, match="Invalid threshold policy"):
        api_main.load_decision_policy()


def test_load_model_raises_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(RuntimeError, match="Model file not found"):
        api_main.load_model()


def test_load_model_raises_clear_message_for_incompatible_artifact(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(api_main, "MODEL_PATH", str(model_path))

    def _raise_incompat(_):
        raise AttributeError(
            "Can't get attribute '_RemainderColsList' on sklearn.compose._column_transformer"
        )

    monkeypatch.setattr(api_main.joblib, "load", _raise_incompat)

    with pytest.raises(RuntimeError, match="scikit-learn compatibility issue"):
        api_main.load_model()

    assert api_main.model is None
    assert api_main.model_load_error is not None
    assert "Current scikit-learn version" in api_main.model_load_error


@pytest.mark.parametrize(
    ("proba", "expected_label", "expected_band"),
    [
        (0.2, 0, "low_risk"),
        (0.6, 0, "review"),
        (0.95, 1, "high_risk"),
    ],
)
def test_predict_assigns_expected_bands(monkeypatch, proba, expected_label, expected_band):
    monkeypatch.setattr(api_main, "model", _DummyModel(proba))
    monkeypatch.setattr(api_main, "risk_threshold", 0.8)
    monkeypatch.setattr(api_main, "review_threshold", 0.5)
    monkeypatch.setattr(api_main, "_append_inference_log", lambda *args, **kwargs: None)

    req = api_main.PredictRequest(year=2023, Fase="ALFA")
    out = api_main.predict(req)
    assert out["risk_label"] == expected_label
    assert out["risk_band"] == expected_band


def test_append_inference_log_rotates_legacy_schema(monkeypatch, tmp_path):
    log_path = tmp_path / "inference.csv"
    log_path.write_text("year,RA,pred_proba,pred_label,ts\n2023,1,0.3,0,1\n", encoding="utf-8")

    monkeypatch.setattr(api_main, "INFERENCE_LOG", str(log_path))
    monkeypatch.setattr(api_main, "risk_threshold", 0.8)
    monkeypatch.setattr(api_main, "review_threshold", 0.5)
    monkeypatch.setattr(api_main.time, "time", lambda: 1700000000)

    api_main._append_inference_log({"year": 2023, "Fase": "ALFA"}, 0.75, 0, "review")

    backup = tmp_path / "inference.csv.legacy.1700000000"
    assert backup.exists()

    out = pd.read_csv(log_path)
    assert list(out.columns) == api_main.INFERENCE_LOG_FIELDS
    assert out.loc[0, "pred_band"] == "review"
    assert out.loc[0, "risk_threshold"] == 0.8
    assert out.loc[0, "review_threshold"] == 0.5
