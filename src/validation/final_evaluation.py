"""Freeze development choices, then evaluate the reserved cohort at most once."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.ml.financial_history import BASE_FEATURES, EXTRA_FEATURES

ROOT = Path("data/processed/validation/reserved_20261017")
TRAIN = ROOT / "development_features.csv"
REPORT = ROOT / "development_comparison.json"
FEATURES = BASE_FEATURES + EXTRA_FEATURES


def prepare():
    # Freeze copies so a later data refresh cannot mutate this experiment.
    for source, destination in ((Path("data/processed/validation/model_readiness/reviewed_features.csv"), TRAIN),
                                (Path("data/processed/validation/model_readiness/report.json"), REPORT)):
        if not destination.exists():
            shutil.copyfile(source, destination)
    development = json.loads(REPORT.read_text())
    cohort = json.loads(Path("config/reserved_cohort_20261017.json").read_text())
    sealed = json.loads((ROOT / "sealed_manifest.json").read_text())
    if sealed["features_sha256"] != sha256((ROOT / "test_features.csv").read_bytes()).hexdigest():
        raise ValueError("Reserved test features changed after preparation")
    candidates = {k: v for k, v in development["development_comparison"].items()
                  if v["validation_status"] == "evaluated" and v["mean_pr_auc"] is not None}
    selected = max(candidates, key=lambda k: candidates[k]["mean_pr_auc"])
    choice = {"model": selected, "features": FEATURES, "threshold": .5,
        "selection": "Highest development mean fold average precision; no final-test predictions used",
        "train_before": cohort["train_labels_available_before"],
        "development_data_sha256": sha256(TRAIN.read_bytes()).hexdigest(),
        "development_report_sha256": sha256(REPORT.read_bytes()).hexdigest(),
        "test_features_sha256": sealed["features_sha256"],
        "cohort_manifest_sha256": sealed["cohort_manifest_sha256"]}
    path = ROOT / "frozen_selection.json"
    if path.exists() and json.loads(path.read_text()) != choice:
        raise ValueError("Frozen selection differs; do not adapt choices against this test")
    write_json(path, choice)
    return choice


def evaluate_once():
    choice = prepare()
    cohort = json.loads(Path("config/reserved_cohort_20261017.json").read_text())
    if sha256(Path("config/reserved_cohort_20261017.json").read_bytes()).hexdigest() != choice["cohort_manifest_sha256"]:
        raise ValueError("Reserved cohort definition changed")
    reserved = {c["cik"] for c in cohort["companies"]}
    frame = pd.read_csv(TRAIN, dtype={"cik": str})
    cutoff = pd.Timestamp(choice["train_before"])
    train = frame.loc[frame.valid_for_training & frame.historical_coverage_eligible &
        ~frame.cik.isin(reserved) & pd.to_datetime(frame.filing_date).lt(cutoff) &
        pd.to_datetime(frame.label_available_date_1yr).lt(cutoff)].copy()
    train = train.sort_values("filing_date").drop_duplicates(["cik", "fiscal_period_end"])
    test = pd.read_csv(ROOT / "test_features.csv", dtype={"cik": str})
    if set(train.cik) & set(test.cik) or not set(test.cik) <= reserved:
        raise ValueError("Company leakage detected")
    if pd.to_datetime(test.filing_date).min() < cutoff:
        raise ValueError("Test period precedes the training cutoff")
    if train.target_distressed_1yr.nunique() != 2 or test.target_distressed_1yr.nunique() != 2:
        raise ValueError("Evaluation requires both classes; never reshuffle the fixed cohort")
    # Exclusive creation prevents routine reruns from repeatedly exposing metrics.
    with (ROOT / "evaluation_started.json").open("x") as marker:
        json.dump({"choice": choice, "status": "evaluation_started"}, marker)
    from src.ml.train_distress import train_logistic_baseline, train_xgboost
    from src.ml.neural_network import train_neural_network
    from src.ml.evaluate import evaluate_classifier
    trainers = {"logistic": train_logistic_baseline, "xgboost": train_xgboost, "neural_network": train_neural_network}
    model = trainers[choice["model"]](train, FEATURES, "target_distressed_1yr")
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    metrics = evaluate_classifier(test.target_distressed_1yr, probabilities, choice["threshold"])
    report = {"selected_model": choice["model"], "test_metrics": metrics,
        "accuracy": float(((probabilities >= choice["threshold"]) == test.target_distressed_1yr).mean()),
        "always_no_bankruptcy_accuracy": float(test.target_distressed_1yr.eq(0).mean()),
        "train_rows": len(train), "test_rows": len(test), "test_companies": test.cik.nunique(),
        "evaluated_once": True, "preprocessing_fit_on": "training rows only", "production_ready": False,
        "independent_test": True, "representative_population": False, "validated_calibration": False,
        "independent_positive_events": int(test.loc[test.target_distressed_1yr.eq(1), "cik"].nunique()),
        "limitations": "Small case-control sample with BRD-scoped historical labels; not market prevalence or calibrated production risk."}
    write_json(ROOT / "final_result.json", report)
    test[["cik", "filing_date", "target_distressed_1yr"]].assign(score=probabilities).to_csv(ROOT / "final_predictions.csv", index=False)
    sealed = json.loads((ROOT / "sealed_manifest.json").read_text())
    sealed["scored"] = True
    write_json(ROOT / "sealed_manifest.json", sealed)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-once", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate_once() if args.evaluate_once else prepare(), indent=2))
