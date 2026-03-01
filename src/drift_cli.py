import argparse
import os
from .drift import compute_drift
from .inference_log import read_inference_log
from .report import build_html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to reference_stats.json")
    parser.add_argument("--inference", required=True, help="Path to inference_log.csv")
    parser.add_argument("--out", required=True, help="Output HTML report path")
    args = parser.parse_args()

    df = read_inference_log(args.inference)
    drift_df = compute_drift(args.reference, df)
    html = build_html(drift_df)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved drift report to {args.out}")

if __name__ == "__main__":
    main()
