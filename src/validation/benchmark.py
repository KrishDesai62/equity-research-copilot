"""Diagnostic real-data comparison, NOT a point-in-time US-market validation.

UCI's 5year.arff is the fifth forecasting case (one-year bankruptcy horizon).
It does not expose company IDs or filing dates. Do not invent them or combine
the five files and pretend that random rows are independent company histories.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import warnings
import zipfile

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.ml.evaluate import evaluate_classifier, compare_to_naive_baseline
from src.ml.neural_network import train_neural_network
from src.ml.train_distress import parse_arff_to_dataframe, train_logistic_baseline, train_xgboost
from src.validation.download import fetch, UCI_URL


def run_benchmark(archive="data/raw/validation/polish_bankruptcy.zip",
                  output="data/processed/validation/benchmark"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = fetch(UCI_URL, archive)
    with zipfile.ZipFile(archive) as data:
        with data.open("5year.arff") as source:
            frame = parse_arff_to_dataframe(source)
    features = [col for col in frame if col != "class"]
    raw_count = len(frame)
    # Remove all feature duplicates, including conflicting-label duplicates.
    duplicate = frame.duplicated(features, keep=False)
    frame = frame.loc[~duplicate].reset_index(drop=True)
    frame["class"] = frame["class"].astype(int)
    train, remaining = train_test_split(frame.index, test_size=.4, stratify=frame["class"], random_state=42)
    validation, test = train_test_split(remaining, test_size=.5, stratify=frame.loc[remaining, "class"], random_state=43)
    splits = {"train": train, "validation": validation, "test": test}
    split_manifest = pd.concat([frame.loc[idx].assign(split=name, source_row=idx) for name, idx in splits.items()])
    split_manifest.to_csv(output / "features_and_splits.csv", index=False)
    report = {"dataset": "UCI Polish Companies Bankruptcy", "source": provenance,
              "citation": "Tomczak (2016), https://doi.org/10.24432/C5F600, CC BY 4.0",
              "file": "5year.arff", "prediction_horizon_years": 1,
              "raw_rows": raw_count, "duplicate_feature_rows_removed": int(duplicate.sum()),
              "feature_count": len(features),
              "protocol": "Fixed stratified 60/20/20 row split; model selected on validation PR-AUC only",
              "limitations": ["No company identifiers or filing timestamps; company and temporal separation cannot be verified.",
                              "Diagnostic classifier benchmark only; not validation of SEC extraction, US-company performance or calibration."],
              "split_counts": {name: {"rows": len(idx), "positives": int(frame.loc[idx, "class"].sum())}
                               for name, idx in splits.items()}, "models": {}}
    trainers = {"logistic": train_logistic_baseline, "xgboost": train_xgboost,
                "neural_network": partial(train_neural_network, max_iter=500)}
    models = {}
    with threadpool_limits(limits=2):
        for name, trainer in trainers.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = trainer(frame.loc[train], features, "class")
            models[name] = model
            scores = model.predict_proba(frame.loc[validation, features])[:, 1]
            report["models"][name] = {
                "validation": {**evaluate_classifier(frame.loc[validation, "class"], scores),
                               **compare_to_naive_baseline(frame.loc[validation, "class"], scores)},
                "training_warnings": sorted({str(w.message) for w in caught})}
            joblib.dump({"model": model, "feature_cols": features, "purpose": "UCI diagnostic benchmark only"},
                        output / f"{name}.joblib")
            print(name, "validation PR-AUC:", round(report["models"][name]["validation"]["pr_auc"], 4), flush=True)
        winner = max(report["models"], key=lambda name: report["models"][name]["validation"]["pr_auc"])
        test_scores = models[winner].predict_proba(frame.loc[test, features])[:, 1]
        report["selected_model"] = winner
        report["selected_model_test"] = {
            **evaluate_classifier(frame.loc[test, "class"], test_scores),
            **compare_to_naive_baseline(frame.loc[test, "class"], test_scores)}
    pd.DataFrame({"source_row": test, "target": frame.loc[test, "class"].to_numpy(),
                  "score": test_scores}).to_csv(output / "test_predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("Selected:", winner, "test PR-AUC:", round(report["selected_model_test"]["pr_auc"], 4), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/raw/validation/polish_bankruptcy.zip")
    parser.add_argument("--output", default="data/processed/validation/benchmark")
    args = parser.parse_args()
    run_benchmark(args.archive, args.output)
