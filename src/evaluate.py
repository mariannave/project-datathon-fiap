import json
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

def save_metrics(y_true, y_pred, y_proba, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    auc = None
    try:
        auc_val = roc_auc_score(y_true, y_proba)
        auc = float(auc_val) if np.isfinite(auc_val) else None
    except Exception:
        auc = None
    metrics = {
        "classification_report": report,
        "roc_auc": auc,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics

def save_confusion_matrix(y_true, y_pred, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return path
