"""Run development-only calibration; insufficient outcomes block rather than leak."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.ml.calibration import fit_candidate
from src.ml.artifacts import save_model
from src.ml.financial_history import BASE_FEATURES, EXTRA_FEATURES


def partition(frame, policy):
    frame = frame.copy()
    frame["cik"] = frame.cik.astype(str).str.zfill(10)
    roles = frame.cik.map(lambda c: int(sha256(f"{policy['seed']}:{c}".encode()).hexdigest(), 16) % 4)
    dates = pd.to_datetime(frame.filing_date, errors="raise")
    available = pd.to_datetime(frame.label_available_date_1yr, errors="raise")
    first, second, last = [pd.Timestamp(policy[key]) for key in ("training_end", "calibration_end", "tuning_end")]
    if not first < second < last:
        raise ValueError("Calibration windows must be chronological")
    return {
        "training": frame.loc[roles.isin([0, 1]) & dates.lt(first) & available.lt(first)].copy(),
        "calibration": frame.loc[roles.eq(2) & dates.ge(first) & dates.lt(second) & available.lt(second)].copy(),
        "threshold_tuning": frame.loc[roles.eq(3) & dates.ge(second) & dates.lt(last) & available.lt(last)].copy()}


def _run(output=None, include_expansion=False):
    output = output or ("data/processed/validation/calibration_expanded" if include_expansion
                        else "data/processed/validation/calibration_development")
    output = Path(output)
    policy_path = Path("config/calibration_development.json")
    policy = json.loads(policy_path.read_text())
    # Frozen development inputs are separate from the exposed final features/scores.
    source = Path("data/processed/validation/reserved_20261017/development_features.csv")
    selection = json.loads(Path("data/processed/validation/reserved_20261017/frozen_selection.json").read_text())
    if sha256(source.read_bytes()).hexdigest() != selection["development_data_sha256"]:
        raise ValueError("Frozen development inputs changed")
    frame = pd.read_csv(source, dtype={"cik": str})
    expansion_hash = None
    if include_expansion:
        extra = Path("data/processed/validation/development_expansion/reviewed_features.csv")
        acquisition = json.loads(extra.with_name("report.json").read_text())
        expansion_hash = sha256(extra.read_bytes()).hexdigest()
        if acquisition.get("features_sha256") != expansion_hash:
            raise ValueError("Expanded development inputs changed after acquisition")
        additional = pd.read_csv(extra, dtype={"cik": str})
        if set(additional.cik) & set(frame.cik):
            raise ValueError("Expanded development companies overlap original development inputs")
        frame = pd.concat([frame, additional], ignore_index=True)
    excluded = set(json.loads(Path("data/processed/validation/expanded/committed_holdout.json").read_text())["holdout_companies"])
    excluded |= {c["cik"] for c in json.loads(Path("config/reserved_cohort_20261017.json").read_text())["companies"]}
    frame = frame.loc[frame.valid_for_training & frame.historical_coverage_eligible & ~frame.cik.isin(excluded)]
    frame = frame.sort_values("filing_date").drop_duplicates(["cik", "fiscal_period_end"])
    parts = partition(frame, policy)
    counts, blockers = {}, []
    for name, rows in parts.items():
        positives = rows.loc[rows.target_distressed_1yr.eq(1), "cik"].nunique()
        negatives = rows.loc[rows.target_distressed_1yr.eq(0), "cik"].nunique()
        counts[name] = {"rows": len(rows), "companies": rows.cik.nunique(),
            "positive_rows": int(rows.target_distressed_1yr.eq(1).sum()),
            "positive_companies": positives, "negative_companies": negatives}
        if positives < policy["min_positive_companies_per_stage"]:
            blockers.append(f"{name}: {positives} positive companies; need at least {policy['min_positive_companies_per_stage']}")
        if negatives < policy["min_negative_companies_per_stage"]:
            blockers.append(f"{name}: {negatives} negative companies; need at least {policy['min_negative_companies_per_stage']}")
    report = {"status": "blocked_insufficient_independent_outcomes", "partitions": counts,
        "blockers": blockers, "input_sha256": sha256(source.read_bytes()).hexdigest(),
        "expansion_sha256": expansion_hash,
        "policy_sha256": sha256(policy_path.read_bytes()).hexdigest(), "policy": policy,
        "final_test_read": False, "final_test_rescored": False, "production_ready": False,
        "selected_model": selection["model"], "validated_calibration": False}
    frozen_path = output / "candidate_selection.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        for key in ("input_sha256", "expansion_sha256", "policy_sha256", "selected_model"):
            if frozen[key] != report[key]:
                raise ValueError("Frozen research candidate inputs changed; start a separate development experiment")
        if sha256((output / "candidate.joblib").read_bytes()).hexdigest() != frozen["artifact_sha256"]:
            raise ValueError("Frozen research candidate artifact changed")
        saved = json.loads((output / "report.json").read_text())
        print(json.dumps(saved, indent=2))
        return saved
    if not blockers:
        from src.ml.train_distress import train_logistic_baseline, train_xgboost
        from src.ml.neural_network import train_neural_network
        trainer = {"logistic": train_logistic_baseline, "xgboost": train_xgboost,
                   "neural_network": train_neural_network}[selection["model"]]
        model, diagnostics = fit_candidate(parts, BASE_FEATURES + EXTRA_FEATURES,
            "target_distressed_1yr", trainer,
            min_precision=policy["min_precision"], min_recall=policy["min_recall"])
        report.update(diagnostics)
        report["status"] = diagnostics["threshold_selection"]["status"]
        if diagnostics["threshold_selection"]["threshold"] is not None:
            artifact = output / "candidate.joblib"
            # Never write to the serving model registry or create a release record.
            save_model(model, BASE_FEATURES + EXTRA_FEATURES, artifact,
                trained_through=policy["tuning_end"], target="target_distressed_1yr")
            frozen = {key: report[key] for key in ("input_sha256", "expansion_sha256", "policy_sha256", "selected_model")}
            frozen.update({"artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
                "threshold": diagnostics["threshold_selection"]["threshold"],
                "status": "research_candidate_only", "production_ready": False,
                "features": BASE_FEATURES + EXTRA_FEATURES,
                "note": "No approved release. Requires a fresh independent evaluation and representative calibration evidence."})
            report["candidate"] = frozen
            write_json(output / "report.json", report)
            with frozen_path.open("x") as handle:
                json.dump(frozen, handle, indent=2)
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


def run(output=None, include_expansion=False):
    destination = Path(output or ("data/processed/validation/calibration_expanded" if include_expansion
                                 else "data/processed/validation/calibration_development"))
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".candidate_run.lock"
    # Never overwrite a candidate concurrently or remove another run's lock.
    with lock.open("x") as handle:
        handle.write("development candidate fitting\n")
    try:
        return _run(str(destination), include_expansion)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--include-expansion", action="store_true")
    result = run(**vars(parser.parse_args()))
    if result["status"].startswith("blocked_"):
        raise SystemExit(2)
