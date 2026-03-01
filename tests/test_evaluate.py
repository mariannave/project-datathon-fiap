import json
import warnings
from pathlib import Path

from sklearn.exceptions import UndefinedMetricWarning

from src.evaluate import save_confusion_matrix, save_metrics


def test_save_metrics_writes_json_with_auc(tmp_path):
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.4, 0.2]

    metrics = save_metrics(y_true, y_pred, y_proba, str(tmp_path))
    assert metrics["roc_auc"] is not None

    path = tmp_path / "metrics.json"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert "classification_report" in loaded
    assert "roc_auc" in loaded


def test_save_metrics_sets_auc_none_when_not_computable(tmp_path):
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    y_proba = [0.1, 0.2, 0.3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        metrics = save_metrics(y_true, y_pred, y_proba, str(tmp_path))
    assert metrics["roc_auc"] is None


def test_save_confusion_matrix_creates_png(tmp_path):
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    out = save_confusion_matrix(y_true, y_pred, str(tmp_path))
    assert Path(out).exists()
    assert out.endswith("confusion_matrix.png")
