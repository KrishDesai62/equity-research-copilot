"""Prespecify a new historical benchmark and score the frozen candidate once."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import brier_score_loss, log_loss

from src.data_layer.research_dataset import financial_labels, load_submissions, write_json
from src.ml.artifacts import load_model
from src.ml.evaluate import evaluate_classifier
from src.ml.financial_history import extract_history
from src.validation.download import companyfacts
from src.validation.historical_coverage import apply_historical_coverage, load_reviewed_registry

ROOT = Path("data/processed/validation/independent_candidate_20261017")
RAW = Path("data/raw/independent_20261017")
PLAN = Path("config/independent_candidate_20261017.json")
CANDIDATE = Path("data/processed/validation/calibration_expanded")
SEED = "independent-candidate-20261017-v1"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def commit():
    choice_path = CANDIDATE / "candidate_selection.json"
    choice = json.loads(choice_path.read_text())
    if digest(CANDIDATE / "candidate.joblib") != choice["artifact_sha256"]:
        raise ValueError("Candidate changed before cohort commitment")
    excluded = set()
    for path in ("data/processed/validation/reserved_20261017/development_features.csv",
                 "data/processed/validation/expanded/features.csv",
                 "data/processed/validation/development_expansion/reviewed_features.csv"):
        excluded |= set(pd.read_csv(path, dtype={"cik": str}).cik)
    prior = json.loads(Path("config/reserved_cohort_20261017.json").read_text())
    excluded |= {c["cik"] for c in prior["companies"]}
    cases = load_reviewed_registry()
    first = cases.dropna(subset=["cik"]).sort_values("filing_date").drop_duplicates("cik")
    eligible = first.loc[first.Chapter.eq("11") & first.filing_date.between("2020-01-01", "2021-12-31") & ~first.cik.isin(excluded)].copy()
    rank = lambda c: sha256(f"{SEED}:{c}".encode()).hexdigest()
    eligible["rank"] = eligible.cik.map(rank)
    positives = [{"cik": r.cik, "ticker": f"CIK{r.cik}", "company_name": r.NameCorp,
        "event_date": r.filing_date, "event_source": "https://lopucki.law.ufl.edu/download_cases_table.php",
        "sampling_role": "case"} for r in eligible.sort_values("rank").head(40).itertuples()]
    archive = RAW / "bulk/2019q4.zip"
    with zipfile.ZipFile(archive) as z:
        historical = pd.read_csv(z.open("sub.txt"), sep="\t", dtype={"cik": str}, low_memory=False)
    historical["cik"] = historical.cik.str.zfill(10)
    sic = pd.to_numeric(historical.sic, errors="coerce")
    # Nonfinancial historical issuers, selected without inspecting model scores.
    historical = historical.loc[historical.form.eq("10-K") & ~sic.between(6000, 6999) & sic.notna()]
    registry_ciKs = set(cases.cik.dropna())
    historical = historical.loc[~historical.cik.isin(excluded | registry_ciKs | {c['cik'] for c in positives})].drop_duplicates("cik").copy()
    historical["rank"] = historical.cik.map(rank)
    controls = [{"cik": r.cik, "ticker": f"CIK{r.cik}", "company_name": r.name,
        "event_date": None, "sampling_role": "control"}
        for r in historical.sort_values("rank").head(40).itertuples(index=False)]
    if len(positives) < 20 or len(controls) != 40:
        raise ValueError("Insufficient prespecified cohort candidates; do not substitute after scoring")
    manifest = {"seed": SEED, "companies": positives + controls, "excluded_companies": sorted(excluded),
        "candidate_selection_sha256": digest(choice_path), "artifact_sha256": choice["artifact_sha256"],
        "threshold": choice["threshold"], "registry_sha256": digest("data/raw/lopucki/cases.download"),
        "historical_universe_sha256": digest(archive), "test_start": "2020-01-01", "test_end": "2021-12-31",
        "observation_end": "2022-12-31", "representative_population": False,
        "sampling": "Prespecified historical case-control benchmark: up to 40 first BRD cases and 40 nonfinancial issuers from SEC 2019 Q4 original annual filings. Not natural market prevalence.",
        "attrition_policy": "No substitutions. Report ineligible/missing-feature firms and permanent 404s. Transient source failures block scoring.",
        "limitations": "Historical scope only; case-control precision and probability calibration do not transfer to current market prevalence."}
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != manifest:
            raise ValueError("Committed independent cohort or candidate changed")
    else:
        with PLAN.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


def materialize():
    manifest = commit()
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "sealed_manifest.json").exists():
        return json.loads((ROOT / "sealed_manifest.json").read_text())
    frames, filings, sources, attrition, errors = [], [], [], [], []
    for company in manifest["companies"]:
        print(f"Preparing independent company {company['cik']}", flush=True)
        try:
            payload, source = companyfacts(company["cik"], directory=RAW / "companyfacts")
            sources.append(source)
            if str(payload["cik"]).zfill(10) != company["cik"]:
                raise ValueError("Companyfacts identity mismatch")
            rows, _ = load_submissions(company, "2010-01-01", manifest["observation_end"], RAW, sources)
            frame, _, _ = extract_history(payload, company["ticker"], public_filings=rows)
            frames.append(frame)
            filings.extend(rows)
        except requests.HTTPError as exc:
            item = {"cik": company["cik"], "reason": str(exc)}
            (attrition if exc.response is not None and exc.response.status_code == 404 else errors).append(item)
        except ValueError as exc:
            destination = attrition if str(exc).startswith("No original 10-K/10-Q USD Assets facts") else errors
            destination.append({"cik": company["cik"], "reason": str(exc)})
        except Exception as exc:
            errors.append({"cik": company["cik"], "reason": str(exc)})
    write_json(ROOT / "acquisition_report.json", {"sources": sources, "attrition": attrition, "errors": errors})
    if errors or not frames:
        raise ValueError("Incomplete source acquisition; retry the same cohort before scoring")
    frame = financial_labels(pd.concat(frames, ignore_index=True), manifest["companies"], manifest["observation_end"])
    frame, evidence = apply_historical_coverage(frame, load_reviewed_registry(), pd.DataFrame(filings), observation_end=manifest["observation_end"])
    frame = frame.loc[frame.historical_coverage_eligible & frame.valid_for_training &
        frame.filing_date.between(manifest["test_start"], manifest["test_end"])].copy()
    frame = frame.sort_values("filing_date").drop_duplicates(["cik", "fiscal_period_end"])
    frame.to_csv(ROOT / "test_features.csv", index=False)
    evidence.to_csv(ROOT / "coverage_evidence.csv", index=False)
    missing = set(c["cik"] for c in manifest["companies"]) - set(frame.cik)
    sealed = {"plan_sha256": digest(PLAN), "features_sha256": digest(ROOT / "test_features.csv"),
        "rows": len(frame), "companies": frame.cik.nunique(), "requested_companies": len(manifest["companies"]),
        "companies_without_eligible_rows": sorted(missing), "source_attrition": attrition,
        "scored": False, "preprocessing_fitted": False}
    write_json(ROOT / "sealed_manifest.json", sealed)
    return sealed


def validate_test(frame, manifest, trained_through):
    if frame.empty or frame.target_distressed_1yr.isna().any() or set(frame.target_distressed_1yr.unique()) != {0, 1}:
        raise ValueError("Independent evaluation needs both classes; never reshuffle to obtain them")
    companies = set(frame.cik.astype(str).str.zfill(10))
    planned = {c["cik"] for c in manifest["companies"]}
    if companies & set(manifest["excluded_companies"]) or not companies <= planned:
        raise ValueError("Company leakage or unplanned test companies")
    dates = pd.to_datetime(frame.filing_date, errors="raise")
    available = pd.to_datetime(frame.label_available_date_1yr, errors="raise")
    inputs = pd.to_datetime(frame.latest_input_filing_date, errors="raise")
    if dates.isna().any() or available.isna().any() or inputs.isna().any():
        raise ValueError("Independent evaluation dates cannot be missing")
    if (dates < pd.Timestamp(trained_through)).any() or (inputs > dates).any():
        raise ValueError("Future training or feature inputs")
    if not dates.between(manifest["test_start"], manifest["test_end"]).all() or (available > pd.Timestamp(manifest["observation_end"])).any():
        raise ValueError("Test dates or outcome availability outside frozen protocol")
    if (available < dates).any() or frame.duplicated(["cik", "fiscal_period_end"]).any():
        raise ValueError("Invalid outcome dates or duplicate company periods")
    negative = frame.target_distressed_1yr.eq(0)
    if (available.loc[negative] < (dates.loc[negative] + pd.DateOffset(years=1))).any():
        raise ValueError("Negative outcomes require a complete one-year horizon")


def evaluate_once():
    manifest = commit()
    sealed = json.loads((ROOT / "sealed_manifest.json").read_text())
    if digest(PLAN) != sealed["plan_sha256"] or digest(ROOT / "test_features.csv") != sealed["features_sha256"]:
        raise ValueError("Sealed independent inputs changed")
    artifact = load_model(CANDIDATE / "candidate.joblib", expected_sha256=manifest["artifact_sha256"])
    frame = pd.read_csv(ROOT / "test_features.csv", dtype={"cik": str})
    validate_test(frame, manifest, artifact["trained_through"])
    # Claim before generating ANY predictions. An interrupted evaluation needs
    # investigation, never deletion of this marker to try different settings.
    with (ROOT / "evaluation_started.json").open("x") as handle:
        json.dump({"plan_sha256": sealed["plan_sha256"], "artifact_sha256": manifest["artifact_sha256"]}, handle)
    features = frame[artifact["feature_cols"]].astype(float).replace([np.inf, -np.inf], np.nan)
    model = artifact["model"]
    probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
    target = frame.target_distressed_1yr.astype(int)
    report = {"test_metrics": evaluate_classifier(target, probabilities, manifest["threshold"]),
        "threshold": manifest["threshold"], "test_rows": len(frame), "test_companies": frame.cik.nunique(),
        "independent_positive_events": frame.loc[target.eq(1), "cik"].nunique(),
        "accuracy": float(((probabilities >= manifest["threshold"]) == target).mean()),
        "always_no_bankruptcy_accuracy": float(target.eq(0).mean()),
        "brier_loss": float(brier_score_loss(target, probabilities)),
        "constant_test_prevalence_brier": float(target.mean() * (1 - target.mean())),
        "naive_average_precision": float(target.mean()),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "independent_test": True, "representative_population": False, "validated_calibration": False,
        "preprocessing_fit_on": "training rows only", "evaluated_once": True, "production_ready": False,
        "candidate_artifact_sha256": manifest["artifact_sha256"], "plan_sha256": sealed["plan_sha256"],
        "attrition": sealed, "limitations": manifest["limitations"]}
    positive = frame.loc[target.eq(1)].assign(detected=probabilities[target.eq(1)] >= manifest["threshold"])
    event_detection = positive.groupby("cik").detected.any()
    report["positive_companies_detected"] = int(event_detection.sum())
    report["event_detection_fraction"] = float(event_detection.mean())
    report["event_detection_note"] = "Any alert on a positive row within the one-year event window; not the same measure as row-level recall."
    report["probability_bins"] = []
    for lower, upper in zip([0, .1, .2, .4, .6, .8], [.1, .2, .4, .6, .8, 1.000001]):
        mask = (probabilities >= lower) & (probabilities < upper)
        report["probability_bins"].append({"lower": lower, "upper": min(upper, 1.), "rows": int(mask.sum()),
            "mean_score": float(probabilities[mask].mean()) if mask.any() else None,
            "observed_fraction": float(target.loc[mask].mean()) if mask.any() else None})
    report["calibration_note"] = "Bins and Brier loss describe this case-control sample only, with repeated company rows; not validated market calibration. Constant prevalence baseline uses test labels for description only."
    frame[["cik", "filing_date", "target_distressed_1yr"]].assign(score=probabilities).to_csv(ROOT / "predictions.csv", index=False)
    write_json(ROOT / "final_result.json", report)
    sealed["scored"] = True
    write_json(ROOT / "sealed_manifest.json", sealed)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate-once", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(materialize(), indent=2))
    elif args.evaluate_once:
        evaluate_once()
    else:
        print(json.dumps(commit(), indent=2))
