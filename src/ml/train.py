"""Train local models from a labeled feature CSV and persist inference artifacts."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.ml.artifacts import save_model
from src.ml.train_distress import train_logistic_baseline, train_xgboost, compare_models
from src.ml.neural_network import train_neural_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--track", choices=["distress", "manipulation"], required=True)
    parser.add_argument("--model", choices=["logistic", "xgboost", "neural_network"], default="neural_network")
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--label-available-col", required=True)
    parser.add_argument("--company-col", default="cik")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--company-folds", type=int, default=3)
    parser.add_argument("--test-companies", nargs="+")
    parser.add_argument("--period", choices=["Q", "Y"], default="Q")
    parser.add_argument("--min-train-periods", type=int, default=4)
    parser.add_argument("--output-dir", default="src/ml/models")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, dtype={args.company_col: str})
    if "valid_for_training" in frame:
        eligible = frame["valid_for_training"].astype(str).str.lower().isin(["true", "1"])
        frame = frame[eligible].copy()
    if "label_review_status" in frame and frame["label_review_status"].eq("coverage_review_required").any():
        raise ValueError("Unreviewed event coverage cannot be used for production training")
    cutoff = pd.Timestamp(args.cutoff)
    dates = pd.to_datetime(frame["filing_date"], errors="raise")
    available = pd.to_datetime(frame[args.label_available_col], errors="raise")
    train = frame[(dates <= cutoff) & (available <= cutoff)].copy()
    if args.compare:
        report = compare_models(train, args.features, args.target, company_col=args.company_col,
                                label_available_col=args.label_available_col,
                                n_company_folds=args.company_folds, test_companies=args.test_companies,
                                period_freq=args.period, min_train_periods=args.min_train_periods)
        report_path = Path(args.output_dir) / f"{args.track}_comparison.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Evaluation: {report_path}")
        if any(value["validation_status"] != "evaluated" for value in report.values()):
            raise ValueError("Evaluation was inconclusive; no production artifact will be saved")
    trainer = {"logistic": train_logistic_baseline, "xgboost": train_xgboost,
               "neural_network": train_neural_network}[args.model]
    model = trainer(train, args.features, args.target)
    path = Path(args.output_dir) / f"{args.track}.joblib"
    save_model(model, args.features, path, trained_through=args.cutoff, target=args.target)
    print(f"Saved {path} from {len(train)} eligible rows")


if __name__ == "__main__":
    main()
