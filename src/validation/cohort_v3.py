"""Frozen, design-weighted bankruptcy experiment. Research artifacts only."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from importlib.metadata import version
import platform
import warnings

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.ml.artifacts import save_model, load_model
from src.ml.design_calibration import WeightedSigmoid, metrics, choose_threshold, reliability
from src.ml.financial_history import BASE_FEATURES, EXTRA_FEATURES
from src.ml.robust_models import train_candidate
from src.validation.cohort_v3_data import ROOT, CONFIG, PLAN, TARGET, digest, prepare, ensure_archives
from src.validation.local_workflow import local_only

FEATURES = BASE_FEATURES + EXTRA_FEATURES + ["log_total_assets", "feature_age_days"]
AVAILABLE = "label_available_date_1yr"


@contextmanager
def lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / ".experiment.lock"
    with path.open("x") as handle:
        handle.write("V3 experiment in progress")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def fold_for(cik, seed, folds):
    return int(sha256(f"{seed}:calibration-fold:{cik}".encode()).hexdigest(), 16) % folds


def validate_parts(parts):
    seen, previous = set(), None
    for role in ("training", "calibration", "final"):
        frame = parts[role]
        required = ["cik", "prediction_date", "feature_filing_date", "latest_input_filing_date", AVAILABLE, "source_url"]
        if frame.empty or frame[required].isna().any().any() or frame.unit_id.duplicated().any():
            raise ValueError(f"Missing identities, public sources or dates in {role}")
        companies = set(frame.cik)
        if seen & companies:
            raise ValueError("Company leakage between stages")
        seen |= companies
        prediction, feature, available = [pd.to_datetime(frame[c]) for c in ("prediction_date", "feature_filing_date", AVAILABLE)]
        if ((feature >= prediction).any() or (pd.to_datetime(frame.latest_input_filing_date) > feature).any()
                or (available <= prediction).any() or (available > prediction + pd.DateOffset(years=1)).any()):
            raise ValueError("Invalid point-in-time financial inputs or outcome availability")
        if (available[frame[TARGET].eq(0)] != (prediction + pd.DateOffset(years=1))[frame[TARGET].eq(0)]).any():
            raise ValueError("A negative lacks full calendar-year follow-up")
        if previous is not None and pd.to_datetime(previous[AVAILABLE]).max() >= prediction.min():
            raise ValueError("Earlier-stage outcomes overlap the next prediction period")
        w, pi = frame.design_weight.to_numpy(float), frame.inclusion_probability.to_numpy(float)
        if not np.isfinite(w).all() or not np.isfinite(pi).all() or (pi <= 0).any() or (pi > 1).any() or not np.allclose(w * pi, 1):
            raise ValueError("Invalid company inclusion weights")
        previous = frame


def read_parts():
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    if (manifest["dataset_sha256"] != digest(ROOT / "dataset.csv") or
            manifest["config_sha256"] != digest(CONFIG) or manifest["plan_sha256"] != digest(PLAN)):
        raise ValueError("Sealed dataset or sampling plan changed")
    frame = pd.read_csv(ROOT / "dataset.csv", dtype={"cik": str})
    usable = frame.eligible.astype(str).str.lower().eq("true") & frame.valid_for_training.astype(str).str.lower().eq("true") & frame[TARGET].notna()
    parts = {role: frame.loc[usable & frame.role.eq(role)].copy() for role in ("training", "calibration", "final")}
    validate_parts(parts)
    return json.loads(CONFIG.read_text()), manifest, parts


def inputs(frame):
    return frame[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)


def describe(frame):
    return {"rows": len(frame), "companies": int(frame.cik.nunique()), "positive_rows": int(frame[TARGET].sum()),
            "positive_companies": int(frame.loc[frame[TARGET].eq(1), "cik"].nunique())}


def verify_selection():
    frozen = json.loads((ROOT / "candidate_selection.json").read_text())
    for key, path in (("config_sha256", CONFIG), ("dataset_sha256", ROOT / "dataset.csv"),
                      ("plan_sha256", PLAN), ("artifact_sha256", ROOT / "candidate.joblib"),
                      ("source_seal_sha256", ROOT / "source_seal.json"),
                      ("data_quality_report_sha256", ROOT / "data_quality_report.json"),
                      ("calibration_predictions_sha256", ROOT / "calibration_predictions.csv"),
                      ("development_report_sha256", ROOT / "development_report.json")):
        if frozen[key] != digest(path):
            raise ValueError(f"Frozen V3 selection changed: {key}")
    return frozen


def fit():
    with lock():
        if (ROOT / "candidate_selection.json").exists():
            verify_selection()
            return json.loads((ROOT / "development_report.json").read_text())
        if (ROOT / "evaluation_started.json").exists():
            raise ValueError("Final evaluation has started; selection cannot be changed")
        config, manifest, parts = read_parts()
        from src.validation.cohort_v3_audit import audit
        audit()
        train, cal = parts["training"], parts["calibration"]
        folds = cal.cik.map(lambda c: fold_for(c, config["seed"], config["calibration_folds"])).to_numpy()
        candidates, failures, fitted = [], [], {}
        oof = cal[["cik", "unit_id", TARGET, "design_weight", AVAILABLE]].copy()
        oof["company_fold"] = folds
        for name in config["models"]:
            print(f"V3 training {name}", flush=True)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = train_candidate(train, FEATURES, TARGET, name, sample_weight=train.design_weight)
                raw = model.predict_proba(inputs(cal))[:, list(model.classes_).index(1)]
                crossfit = np.full(len(cal), np.nan)
                for fold in range(config["calibration_folds"]):
                    held = folds == fold
                    if not held.any():
                        continue
                    calibrator = WeightedSigmoid(model, config["calibration_regularization"]).fit_scores(
                        raw[~held], cal[TARGET].to_numpy()[~held], cal.design_weight.to_numpy()[~held])
                    crossfit[held] = calibrator.predict_scores(raw[held])
                measured = metrics(cal[TARGET], crossfit, cal.design_weight)
                oof[name] = crossfit
                threshold = choose_threshold(cal[TARGET], crossfit, cal.design_weight, config["min_precision"], config["min_recall"])
                candidates.append({"name": name, "out_of_company_fold_metrics": measured, "threshold": threshold,
                    "warnings": sorted(set(str(w.message) for w in caught))})
                fitted[name] = model
            except (ValueError, RuntimeError) as exc:
                failures.append({"name": name, "reason": str(exc)})
        if not candidates:
            write_json(ROOT / "development_failure.json", {"failures": failures, "production_ready": False})
            raise ValueError("No valid cross-fitted candidate; see development_failure.json")
        selected = min(candidates, key=lambda r: (r["out_of_company_fold_metrics"]["log_loss"], r["out_of_company_fold_metrics"]["brier"], r["name"]))
        model = fitted[selected["name"]]
        wrapper = WeightedSigmoid(model, config["calibration_regularization"]).fit_scores(
            model.predict_proba(inputs(cal))[:, list(model.classes_).index(1)], cal[TARGET], cal.design_weight)
        baseline = float(np.average(cal[TARGET], weights=cal.design_weight))
        save_model(wrapper, FEATURES, ROOT / "candidate.joblib", trained_through="2018-12-30", target=TARGET)
        oof.to_csv(ROOT / "calibration_predictions.csv", index=False)
        report = {"training": describe(train), "calibration": describe(cal), "candidates": candidates, "failures": failures,
                  "selected": selected, "weighted_calibration_prevalence": baseline,
                  "calibration_positive_company_gate": describe(cal)["positive_companies"] >= config["min_calibration_positive_companies"],
                  "final_used_for_selection": False, "production_ready": False,
                  "calibration_slope": float(wrapper.calibrator_.coef_[0, 0]),
                  "calibration_intercept": float(wrapper.calibrator_.intercept_[0]),
                  "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")}},
                  "implementation_sha256": {path: digest(path) for path in ("src/validation/cohort_v3.py", "src/validation/cohort_v3_data.py", "src/validation/cohort_v3_audit.py", "src/validation/historical_coverage.py", "src/data_layer/research_dataset.py", "src/data_layer/bankruptcy_labels.py", "src/ml/design_calibration.py", "src/ml/robust_models.py", "src/ml/financial_history.py", "requirements-local.lock")},
                  "warning": "Cross-fitting controls calibration reuse, but selecting among candidates still makes development estimates optimistic."}
        write_json(ROOT / "development_report.json", report)
        write_json(ROOT / "candidate_selection.json", {"config_sha256": digest(CONFIG), "dataset_sha256": manifest["dataset_sha256"],
            "plan_sha256": digest(PLAN), "artifact_sha256": digest(ROOT / "candidate.joblib"),
            "source_seal_sha256": digest(ROOT / "source_seal.json"),
            "data_quality_report_sha256": digest(ROOT / "data_quality_report.json"),
            "calibration_predictions_sha256": digest(ROOT / "calibration_predictions.csv"),
            "development_report_sha256": digest(ROOT / "development_report.json"), "selected": selected["name"],
            "threshold": selected["threshold"], "constant_baseline": baseline, "final_used_for_selection": False})
        return report


def cluster_intervals(frame, probabilities, threshold, baseline, repetitions=500):
    """Resample companies, retaining their snapshots and original design weights."""
    groups = list(frame.groupby("cik", sort=True).indices.values())
    rng = np.random.default_rng(20261017)
    values = {key: [] for key in ("brier", "log_loss", "average_precision", "roc_auc", "brier_improvement_over_constant")}
    if threshold is not None:
        values.update(precision=[], recall=[])
    y, w = frame[TARGET].to_numpy(), frame.design_weight.to_numpy()
    for _ in range(repetitions):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        measured = metrics(y[idx], probabilities[idx], w[idx], threshold)
        constant = metrics(y[idx], np.full(len(idx), baseline), w[idx])
        measured["brier_improvement_over_constant"] = constant["brier"] - measured["brier"]
        for key in values:
            if measured[key] is not None:
                values[key].append(measured[key])
    return {key: {"lower": float(np.quantile(v, .025)), "upper": float(np.quantile(v, .975)), "valid_repetitions": len(v)}
            for key, v in values.items() if v}


def evaluate_once():
    with lock():
        frozen = verify_selection()
        config, _, parts = read_parts()
        from src.validation.cohort_v3_audit import audit
        audit()
        verify_selection()
        test = parts["final"].reset_index(drop=True)
        artifact = load_model(ROOT / "candidate.joblib", expected_sha256=frozen["artifact_sha256"])
        if artifact["feature_cols"] != FEATURES or artifact["target"] != TARGET:
            raise ValueError("Unexpected final artifact schema")
        if pd.Timestamp(artifact["trained_through"]) >= pd.to_datetime(test.prediction_date).min():
            raise ValueError("Artifact calibration cutoff overlaps the final period")
        # Consume before any final prediction; failures cannot silently reset the test.
        with (ROOT / "evaluation_started.json").open("x") as handle:
            json.dump({"artifact_sha256": frozen["artifact_sha256"], "selection_sha256": digest(ROOT / "candidate_selection.json"),
                       "dataset_sha256": frozen["dataset_sha256"], "final_test_consumed": True}, handle)
        p = artifact["model"].predict_proba(inputs(test))[:, 1]
        weighted = metrics(test[TARGET], p, test.design_weight, frozen["threshold"])
        constant = metrics(test[TARGET], np.full(len(test), frozen["constant_baseline"]), test.design_weight)
        intervals = cluster_intervals(test, p, frozen["threshold"], frozen["constant_baseline"])
        development = json.loads((ROOT / "development_report.json").read_text())
        gates = {"minimum_final_companies": test.cik.nunique() >= config["min_final_companies"],
                 "minimum_final_positive_companies": describe(test)["positive_companies"] >= config["min_final_positive_companies"],
                 "adequate_calibration_positive_companies": development["calibration_positive_company_gate"],
                 "threshold_exists": frozen["threshold"] is not None,
                 "precision_floor": weighted.get("precision", 0) >= config["min_precision"],
                 "recall_floor": weighted.get("recall", 0) >= config["min_recall"],
                 "precision_interval_lower_floor": intervals.get("precision", {}).get("lower", 0) >= config["min_precision"],
                 "recall_interval_lower_floor": intervals.get("recall", {}).get("lower", 0) >= config["min_recall"],
                 "brier_beats_constant": weighted["brier"] < constant["brier"],
                 "log_loss_beats_constant": weighted["log_loss"] < constant["log_loss"],
                 "independent_company_and_time_test": True,
                 "current_population_source_completeness_reviewed": False,
                 "external_probability_calibration_validated": False}
        report = {"final": describe(test), "design_weighted": weighted,
                  "fixed_bin_calibration_diagnostics": reliability(test[TARGET], p, test.design_weight),
                  "unweighted_audit": metrics(test[TARGET], p, np.ones(len(test)), frozen["threshold"]),
                  "constant_baseline": constant, "company_cluster_95_percent_intervals": intervals,
                  "gates": {k: bool(v) for k, v in gates.items()}, "production_ready": bool(all(gates.values())),
                  "limitations": ["Historical eligible population through 2022; no 2026 outcomes.",
                    "Inverse sampling weights do not correct missing sources or imperfect legal-entity registry coverage.",
                    "Cluster intervals are conditional on the observed sample, not a source completeness certificate.",
                    "No agent trading-return validation or production approval is implied."],
                  "artifact_sha256": frozen["artifact_sha256"], "dataset_sha256": frozen["dataset_sha256"]}
        test["predicted_probability"] = p
        test.to_csv(ROOT / "final_predictions.csv", index=False)
        write_json(ROOT / "final_evaluation.json", report)
        write_json(ROOT / "evaluation_receipt.json", {"selection_sha256": digest(ROOT / "candidate_selection.json"),
            "report_sha256": digest(ROOT / "final_evaluation.json"), "predictions_sha256": digest(ROOT / "final_predictions.csv"),
            "started_sha256": digest(ROOT / "evaluation_started.json")})
        return report


def summary():
    verify_selection()
    receipt = json.loads((ROOT / "evaluation_receipt.json").read_text())
    for key, name in (("selection_sha256", "candidate_selection.json"), ("report_sha256", "final_evaluation.json"),
                      ("predictions_sha256", "final_predictions.csv"), ("started_sha256", "evaluation_started.json")):
        if receipt[key] != digest(ROOT / name):
            raise ValueError("Saved final evidence changed")
    return json.loads((ROOT / "final_evaluation.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "fit", "evaluate-once", "summary"])
    args = parser.parse_args()
    with local_only():
        if args.action == "prepare":
            ensure_archives()
            result = prepare()
        elif args.action == "fit":
            result = fit()
        elif args.action == "evaluate-once":
            result = evaluate_once()
        else:
            result = summary()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
