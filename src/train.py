import argparse
import os
import joblib
import pandas as pd

from .data_io import load_excel
from .preprocessing import basic_clean
from .feature_engineering import add_derived_features, select_features, pick_target
from .model import build_pipeline
from .evaluate import save_metrics, save_confusion_matrix
from .reference import compute_reference_stats, save_reference_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path to PEDE Excel file")
    parser.add_argument("--out", default="artifacts", help="Output directory for artifacts")
    parser.add_argument("--reports", default="reports", help="Output directory for reports")
    args = parser.parse_args()

    data = load_excel(args.excel)
    d22 = basic_clean(data["PEDE2022"])
    d23 = basic_clean(data["PEDE2023"])
    d24 = basic_clean(data["PEDE2024"])

    # Build training dataset: 2022 -> target 2023
    merged_train = d22.merge(d23[["RA","Defasagem"]], on="RA", how="inner")
    X_train_raw = add_derived_features(merged_train, year=2022)
    X_train = select_features(X_train_raw, year=2022, mode="2022")
    y_train = pick_target(merged_train, target_col="Defasagem")

    # Validation dataset: 2023 -> target 2024
    merged_valid = d23.merge(d24[["RA","Defasagem"]], on="RA", how="inner", suffixes=("","_target"))
    X_valid_raw = add_derived_features(merged_valid, year=2023)
    X_valid = select_features(X_valid_raw, year=2023, mode="2023")
    y_valid = pick_target(merged_valid, target_col="Defasagem_target")

    # Align columns: keep intersection
    common_cols = [c for c in X_train.columns if c in X_valid.columns]
    X_train = X_train[common_cols]
    X_valid = X_valid[common_cols]

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_valid)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_valid)[:,1]
    else:
        y_proba = y_pred.astype(float)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)

    joblib.dump(pipe, os.path.join(args.out, "model.joblib"))

    metrics = save_metrics(y_valid, y_pred, y_proba, args.reports)
    save_confusion_matrix(y_valid, y_pred, args.reports)

    ref = compute_reference_stats(X_train)
    save_reference_stats(ref, os.path.join(args.out, "reference_stats.json"))

    print("Training complete.")
    print(f"Saved model to {os.path.join(args.out,'model.joblib')}")
    print(f"Saved reference stats to {os.path.join(args.out,'reference_stats.json')}")
    print(f"Saved metrics to {os.path.join(args.reports,'metrics.json')}")

if __name__ == "__main__":
    main()
