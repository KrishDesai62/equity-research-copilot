"""Tune research models on new development data and compare once on new firms."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import load_model, save_model
from src.ml.calibration import select_threshold
from src.ml.evaluate import evaluate_classifier
from src.ml.financial_history import BASE_FEATURES, EXTRA_FEATURES
from src.ml.robust_models import InterceptCalibrator, train_candidate
from src.validation.improvement_data import ROOT, CONFIG, PLAN, commit, digest, prepare
from src.validation.local_workflow import local_only
from src.validation.population_calibration import calibration_metrics

FEATURES = BASE_FEATURES + EXTRA_FEATURES
TARGET = "target_distressed_1yr"
AVAILABLE = "label_available_date_1yr"
OLD = Path("data/processed/validation/calibration_expanded")


def eligible(frame):
    return (frame.valid_for_training.astype(str).str.lower().eq("true") &
            frame.historical_coverage_eligible.astype(str).str.lower().eq("true") & frame[TARGET].notna())


def make_parts(natural, original, config):
    from hashlib import sha256
    natural = natural.copy()
    natural["cik"] = natural.cik.astype(str).str.zfill(10)
    roles = natural.cik.map(lambda c: int(sha256(f"{config['seed']}:{c}".encode()).hexdigest(), 16) % 4)
    dates = pd.to_datetime(natural.filing_date)
    train = natural.loc[roles.eq(0) & dates.lt(config["training_filing_before"]) & eligible(natural)]
    # Original positive-enriched development data may support classifier fitting
    # only. Natural calibration, selection and final firms are entirely separate.
    original = original.loc[eligible(original) & pd.to_datetime(original.filing_date).lt(config["training_filing_before"])].copy()
    train = pd.concat([train, original], ignore_index=True)
    train = train.loc[pd.to_datetime(train[AVAILABLE]).lt(config["training_outcomes_before"])]
    train = train.sort_values(["filing_date", "accession_number"]).drop_duplicates(["cik", "fiscal_period_end"])
    parts = {"training": train}
    for name, role, key in (("calibration", 1, "calibration_year"), ("selection", 2, "selection_year"), ("final", 3, "final_year")):
        # Choose the first annual anchor before looking at coverage or labels.
        rows = natural.loc[roles.eq(role) & dates.dt.year.eq(config[key])].sort_values(["filing_date", "accession_number"]).drop_duplicates("cik")
        parts[name] = rows.loc[eligible(rows)].copy()
    return parts


def validate_stages(parts, *, include_final=False):
    names = ["training", "calibration", "selection"] + (["final"] if include_final else [])
    seen, previous = set(), None
    for name in names:
        frame = parts[name]
        if frame.empty or frame[["cik", "filing_date", AVAILABLE, "latest_input_filing_date"]].isna().any().any():
            raise ValueError(f"Insufficient identified, dated {name} observations")
        companies = set(frame.cik)
        if companies & seen:
            raise ValueError("Companies overlap across experimental stages")
        seen |= companies
        dates = pd.to_datetime(frame.filing_date)
        available = pd.to_datetime(frame[AVAILABLE])
        if (available < dates).any() or (pd.to_datetime(frame.latest_input_filing_date) > dates).any():
            raise ValueError("Invalid public feature/outcome dates")
        if (available.loc[frame[TARGET].eq(0)] < dates.loc[frame[TARGET].eq(0)] + pd.DateOffset(years=1)).any():
            raise ValueError("Negative labels lack full one-year follow-up")
        if previous is not None and pd.to_datetime(previous[AVAILABLE]).max() >= dates.min():
            raise ValueError("Earlier-stage outcomes were unavailable before the next stage")
        previous = frame


def clean(frame):
    return frame[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)


def describe(frame):
    return {"rows": len(frame), "companies": frame.cik.nunique(), "positive_rows": int(frame[TARGET].eq(1).sum()),
            "positive_companies": frame.loc[frame[TARGET].eq(1), "cik"].nunique()}


@contextmanager
def experiment_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / ".experiment.lock"
    with lock.open("x") as handle:
        handle.write("V2 experiment in progress\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def read_parts():
    config = json.loads(CONFIG.read_text())
    data = prepare()  # Verified checkpoints; sealed data are reused without requests.
    natural = pd.read_csv(ROOT / "natural_features.csv", dtype={"cik": str})
    originals, hashes = [], {}
    heldout = set(json.loads(Path("data/processed/validation/expanded/committed_holdout.json").read_text())["holdout_companies"])
    for path in (Path("data/processed/validation/reserved_20261017/development_features.csv"),
                 Path("data/processed/validation/development_expansion/reviewed_features.csv")):
        frame = pd.read_csv(path, dtype={"cik": str})
        originals.append(frame.loc[~frame.cik.isin(heldout)])
        hashes[str(path)] = digest(path)
    parts = make_parts(natural, pd.concat(originals, ignore_index=True), config)
    return config, data, hashes, parts


def fit():
    with local_only(), experiment_lock():
        config, data, sources, parts = read_parts()
        frozen_path = ROOT / "candidate_selection.json"
        if frozen_path.exists():
            choice = json.loads(frozen_path.read_text())
            if (choice["config_sha256"] != digest(CONFIG) or choice["data_sha256"] != data["features_sha256"] or
                    choice["training_source_hashes"] != sources or choice["artifact_sha256"] != digest(ROOT / "candidate.joblib")):
                raise ValueError("Frozen v2 candidate changed; never retune after final evaluation")
            return json.loads((ROOT / "development_report.json").read_text())
        validate_stages(parts)
        train, calibration, selection = (parts[key] for key in ("training", "calibration", "selection"))
        if train[TARGET].nunique() != 2:
            raise ValueError("V2 training needs both classes; no label-driven reshuffling is allowed")
        results, candidates, failures = [], {}, []
        for name in config["models"]:
            print(f"Training V2 candidate {name}", flush=True)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                base = train_candidate(train, FEATURES, TARGET, name)
            for method in config["calibration_methods"]:
                key = f"{name}:{method}"
                candidate = base if method == "none" else InterceptCalibrator(base, config["intercept_pseudocount_per_class"]).fit(clean(calibration), calibration[TARGET])
                scores = candidate.predict_proba(clean(selection))[:, list(candidate.classes_).index(1)]
                metrics = calibration_metrics(selection[TARGET], scores)
                threshold = (select_threshold(selection[TARGET], scores,
                    min_precision=config["threshold_min_precision"], min_recall=config["threshold_min_recall"])
                    if selection[TARGET].nunique() == 2 else {"status": "blocked_single_class_selection", "threshold": None})
                results.append({"candidate": key, "selection_metrics": metrics, "threshold_selection": threshold,
                    "warnings": sorted({str(w.message) for w in captured})})
                candidates[key] = candidate
        selected = min(results, key=lambda r: (r["selection_metrics"]["log_loss"], r["selection_metrics"]["brier_loss"], r["candidate"]))
        model = candidates[selected["candidate"]]
        artifact = ROOT / "candidate.joblib"
        save_model(model, FEATURES, artifact, trained_through=f"{config['final_year']}-01-01", target=TARGET)
        # Seal final features now; they have not been predicted or used to choose
        # models, calibrators or thresholds. Do not print their outcome counts.
        parts["final"].to_csv(ROOT / "final_features.csv", index=False)
        choice = {"selected_candidate": selected["candidate"], "threshold": selected["threshold_selection"]["threshold"],
            "threshold_status": selected["threshold_selection"]["status"], "artifact_sha256": digest(artifact),
            "features": FEATURES, "config_sha256": digest(CONFIG), "data_sha256": data["features_sha256"],
            "training_source_hashes": sources, "final_features_sha256": digest(ROOT / "final_features.csv"),
            "development_companies": sorted(set(train.cik) | set(calibration.cik) | set(selection.cik)),
            "baseline_sha256": json.loads(PLAN.read_text())["baseline_sha256"],
            "constant_calibration_prevalence": float((calibration[TARGET].sum() + 1) / (len(calibration) + 2)),
            "production_ready": False, "validated_calibration": False}
        report = {"status": "frozen_research_candidate_only", "partitions": {key: describe(parts[key]) for key in ("training", "calibration", "selection")},
            "comparisons": results, "selected": selected, "candidate": choice, "final_test_scored": False,
            "production_ready": False, "limits": config["limits"], "acquisition": data,
            "note": "Selection loss is a development result. Small/single-class natural samples cannot validate bankruptcy detection."}
        write_json(ROOT / "development_report.json", report)
        with frozen_path.open("x") as handle:
            json.dump(choice, handle, indent=2)
        return report


def evaluate_once():
    with local_only(), experiment_lock():
        choice = json.loads((ROOT / "candidate_selection.json").read_text())
        if digest(CONFIG) != choice["config_sha256"] or digest(ROOT / "final_features.csv") != choice["final_features_sha256"]:
            raise ValueError("Frozen final protocol changed")
        config, data, sources, parts = read_parts()
        if data["features_sha256"] != choice["data_sha256"] or sources != choice["training_source_hashes"]:
            raise ValueError("Frozen development provenance changed")
        validate_stages(parts, include_final=True)
        final = pd.read_csv(ROOT / "final_features.csv", dtype={"cik": str})
        if set(final.cik) & set(choice["development_companies"]) or final.cik.duplicated().any():
            raise ValueError("Final company leakage or repeated observations")
        model = load_model(ROOT / "candidate.joblib", expected_sha256=choice["artifact_sha256"])["model"]
        baseline = load_model(OLD / "candidate.joblib", expected_sha256=choice["baseline_sha256"])["model"]
        with (ROOT / "evaluation_started.json").open("x") as marker:
            json.dump({"candidate": choice["artifact_sha256"], "final_features": choice["final_features_sha256"]}, marker)
        probabilities = {}
        results = {}
        for name, estimator in (("v2", model), ("previous_candidate", baseline)):
            scores = estimator.predict_proba(clean(final))[:, list(estimator.classes_).index(1)]
            probabilities[name] = scores
            results[name] = calibration_metrics(final[TARGET], scores)
        constant = np.full(len(final), choice["constant_calibration_prevalence"])
        results["constant_calibration_rate"] = calibration_metrics(final[TARGET], constant)
        threshold = choice["threshold"]
        classification = (evaluate_classifier(final[TARGET], probabilities["v2"], threshold)
                          if threshold is not None and final[TARGET].nunique() == 2 else None)
        report = {"evaluated_once": True, "candidate": choice, "final_cohort": describe(final),
            "probability_metrics": results, "classification_metrics": classification,
            "classification_status": "measured" if classification is not None else "inconclusive_insufficient_classes_or_no_feasible_threshold",
            "production_ready": False, "representative_population": False, "validated_calibration": False,
            "limits": config["limits"],
            "note": "Same new companies and frozen comparators. Lower probability loss does not establish useful bankruptcy recall or market calibration."}
        final[["cik", "filing_date", TARGET]].assign(v2_score=probabilities["v2"], previous_score=probabilities["previous_candidate"]).to_csv(ROOT / "final_predictions.csv", index=False)
        write_json(ROOT / "final_result.json", report)
        return report


def assess_result(result, development):
    candidate = result["probability_metrics"]["v2"]
    previous = result["probability_metrics"]["previous_candidate"]
    constant = result["probability_metrics"]["constant_calibration_rate"]
    failures = []
    if development["candidate"]["threshold"] is None:
        failures.append("No defensible classification threshold was selected")
    if development["partitions"]["calibration"]["positive_companies"] < 3:
        failures.append("Too few independent positive companies for calibration")
    if result["final_cohort"]["positive_companies"] < 20:
        failures.append("Too few independent positive final events to establish detection performance")
    if candidate["brier_loss"] >= constant["brier_loss"] or candidate["log_loss"] >= constant["log_loss"]:
        failures.append("The selected candidate did not beat the constant-rate baseline on both probability losses")
    if not result["representative_population"] or not result["validated_calibration"]:
        failures.append("Representative-population calibration is unvalidated")
    return {"production_ready": False, "research_status": "not_validated", "failures": failures,
        "brier_improved_vs_previous": candidate["brier_loss"] < previous["brier_loss"],
        "log_loss_improved_vs_previous": candidate["log_loss"] < previous["log_loss"],
        "note": "Lower loss than a weak previous candidate is insufficient when a simple baseline is better."}


def summary():
    result = json.loads((ROOT / "final_result.json").read_text())
    development = json.loads((ROOT / "development_report.json").read_text())
    if digest(ROOT / "candidate.joblib") != result["candidate"]["artifact_sha256"]:
        raise ValueError("V2 summary artifact integrity check failed")
    assessment = assess_result(result, development)
    output = {"data": development["acquisition"], "development_partitions": development["partitions"],
        "selected_candidate": result["candidate"]["selected_candidate"], "final_cohort": result["final_cohort"],
        "probability_metrics": result["probability_metrics"], "assessment": assessment,
        "old_final_tests_rescored": False, "aws_deployed": False}
    write_json(ROOT / "summary.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--fit", action="store_true")
    group.add_argument("--evaluate-once", action="store_true")
    group.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    with local_only():
        if args.prepare:
            from src.data_layer.xbrl_bulk import download_quarterly_zip
            from src.validation.improvement_data import RAW
            config = json.loads(CONFIG.read_text())
            for year in config["vintages"]:
                download_quarterly_zip(year, config["quarter"], RAW / "bulk", cache=False)
            result = prepare()
        else:
            result = summary() if args.summary else evaluate_once() if args.evaluate_once else fit() if args.fit else commit()
    print(json.dumps(result, indent=2))
