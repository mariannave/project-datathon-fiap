import sys

import numpy as np
import pandas as pd

from src import train as train_mod


class _PipeWithProba:
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        self.fit_columns = list(X.columns)
        self.fit_y = list(y)
        return self

    def predict(self, X):
        return np.array([0, 1], dtype=int)

    def predict_proba(self, X):
        return np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)


class _PipeNoProba:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([1, 0], dtype=int)


def _base_data():
    d22 = pd.DataFrame({"RA": [1, 2], "A": [10, 20]})
    d23 = pd.DataFrame({"RA": [1, 2], "Defasagem": [0, 1], "B": [30, 40]})
    d24 = pd.DataFrame({"RA": [1, 2], "Defasagem": [1, 0], "C": [50, 60]})
    return {"PEDE2022": d22, "PEDE2023": d23, "PEDE2024": d24}


def _patch_common(monkeypatch):
    monkeypatch.setattr(train_mod, "load_excel", lambda _: _base_data())
    monkeypatch.setattr(train_mod, "basic_clean", lambda df: df.copy())
    monkeypatch.setattr(train_mod, "add_derived_features", lambda df, year: df.copy())

    def fake_select_features(df, year, mode):
        if mode == "2022":
            return pd.DataFrame({"common": [1.0, 2.0], "train_only": [9.0, 9.0]})
        return pd.DataFrame({"common": [1.5, 2.5], "valid_only": [8.0, 8.0]})

    monkeypatch.setattr(train_mod, "select_features", fake_select_features)
    monkeypatch.setattr(train_mod, "pick_target", lambda df, target_col: pd.Series([0, 1], dtype=int))


def test_train_main_runs_end_to_end_with_predict_proba(monkeypatch, tmp_path):
    _patch_common(monkeypatch)

    dump_calls = {}
    metric_calls = {}
    reference_calls = {}

    pipe = _PipeWithProba()
    monkeypatch.setattr(train_mod, "build_pipeline", lambda X: pipe)
    monkeypatch.setattr(train_mod.joblib, "dump", lambda obj, path: dump_calls.update({"obj": obj, "path": path}))

    def fake_save_metrics(y_true, y_pred, y_proba, out_dir):
        metric_calls["y_true"] = list(y_true)
        metric_calls["y_pred"] = list(y_pred)
        metric_calls["y_proba"] = list(y_proba)
        metric_calls["out_dir"] = out_dir
        return {"ok": True}

    monkeypatch.setattr(train_mod, "save_metrics", fake_save_metrics)
    monkeypatch.setattr(train_mod, "save_confusion_matrix", lambda y_true, y_pred, out_dir: None)
    monkeypatch.setattr(train_mod, "compute_reference_stats", lambda X: {"columns": {}, "n_rows": len(X)})
    monkeypatch.setattr(
        train_mod,
        "save_reference_stats",
        lambda stats, path: reference_calls.update({"stats": stats, "path": path}),
    )

    out_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--excel",
            str(tmp_path / "fake.xlsx"),
            "--out",
            str(out_dir),
            "--reports",
            str(reports_dir),
        ],
    )

    train_mod.main()

    assert pipe.fitted is True
    assert pipe.fit_columns == ["common"]
    assert dump_calls["obj"] is pipe
    assert dump_calls["path"].endswith("model.joblib")
    assert metric_calls["y_true"] == [0, 1]
    assert metric_calls["y_pred"] == [0, 1]
    assert metric_calls["y_proba"] == [0.2, 0.7]
    assert reference_calls["path"].endswith("reference_stats.json")


def test_train_main_falls_back_when_predict_proba_missing(monkeypatch, tmp_path):
    _patch_common(monkeypatch)

    metric_calls = {}
    monkeypatch.setattr(train_mod, "build_pipeline", lambda X: _PipeNoProba())
    monkeypatch.setattr(train_mod.joblib, "dump", lambda obj, path: None)
    monkeypatch.setattr(train_mod, "save_confusion_matrix", lambda y_true, y_pred, out_dir: None)
    monkeypatch.setattr(train_mod, "compute_reference_stats", lambda X: {"columns": {}, "n_rows": len(X)})
    monkeypatch.setattr(train_mod, "save_reference_stats", lambda stats, path: None)

    def fake_save_metrics(y_true, y_pred, y_proba, out_dir):
        metric_calls["y_pred"] = list(y_pred)
        metric_calls["y_proba"] = list(y_proba)
        return {"ok": True}

    monkeypatch.setattr(train_mod, "save_metrics", fake_save_metrics)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--excel",
            str(tmp_path / "fake.xlsx"),
            "--out",
            str(tmp_path / "artifacts"),
            "--reports",
            str(tmp_path / "reports"),
        ],
    )

    train_mod.main()
    assert metric_calls["y_pred"] == [1, 0]
    assert metric_calls["y_proba"] == [1.0, 0.0]
