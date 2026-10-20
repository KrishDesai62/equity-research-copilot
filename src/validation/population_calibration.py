"""Local natural-sample calibration audit; never certify broad market coverage."""
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
from src.ml.financial_history import extract_history
from src.validation.download import companyfacts
from src.validation.historical_coverage import apply_historical_coverage, load_reviewed_registry

ROOT = Path("data/processed/validation/population_calibration_20261017")
RAW = Path("data/raw/population_calibration_20261017")
PLAN = Path("config/population_calibration_20261017.json")
CANDIDATE = Path("data/processed/validation/calibration_expanded")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def select_sample(universe, seed, size):
    """Sample issuer IDs without consulting outcomes, scores or present-day status."""
    frame = universe.copy()
    frame["cik"] = frame.cik.astype(str).str.zfill(10)
    sic = pd.to_numeric(frame.sic, errors="coerce")
    frame = frame.loc[frame.form.eq("10-K") & sic.notna() & ~sic.between(6000, 6999)].drop_duplicates("cik").copy()
    frame["rank"] = frame.cik.map(lambda c: sha256(f"{seed}:{c}".encode()).hexdigest())
    return frame.sort_values("rank").head(size)


def commit():
    archive = RAW / "bulk/2019q1.zip"
    with zipfile.ZipFile(archive) as source:
        universe = pd.read_csv(source.open("sub.txt"), sep="\t", dtype={"cik": str}, low_memory=False)
    sample = select_sample(universe, "natural-sample-calibration-v1-20261017", 120)
    excluded = set()
    for path in ("data/processed/validation/reserved_20261017/development_features.csv",
                 "data/processed/validation/expanded/features.csv",
                 "data/processed/validation/development_expansion/reviewed_features.csv"):
        excluded |= set(pd.read_csv(path, dtype={"cik": str}).cik)
    for path in ("config/reserved_cohort_20261017.json", "config/independent_candidate_20261017.json"):
        excluded |= {c["cik"] for c in json.loads(Path(path).read_text())["companies"]}
    choice = json.loads((CANDIDATE / "candidate_selection.json").read_text())
    if digest(CANDIDATE / "candidate.joblib") != choice["artifact_sha256"]:
        raise ValueError("Frozen model artifact changed")
    manifest = {"version": 1, "seed": "natural-sample-calibration-v1-20261017",
        "companies": [{"cik": r.cik, "ticker": f"CIK{r.cik}", "company_name": r.name}
                      for r in sample.itertuples(index=False)],
        "excluded_exposed_companies": sorted(excluded), "sample_size": 120,
        "universe_sha256": digest(archive), "registry_sha256": digest("data/raw/lopucki/cases.download"),
        "candidate_artifact_sha256": choice["artifact_sha256"], "threshold": choice["threshold"],
        "selection": "Uniform identifier-hash sample from SEC 2019 Q1 nonfinancial original annual issuers, before outcome lookup. No case/control balancing.",
        "anchor": "First extracted original public financial filing during 2020, selected before coverage/label filtering; one observation per company.",
        "overlap_policy": "Record sampled exposed companies as exclusions, without scoring or replacement. This exclusion and coverage attrition can bias the result.",
        "observation_end": "2022-12-31", "production_ready": False,
        "limitations": "Historical one-quarter annual-issuer frame only, BRD large-public-company eligibility, complete-source/coverage attrition and prior-exposure exclusion. Not all issuers or current market prevalence."}
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != manifest:
            raise ValueError("Committed natural-sample experiment changed")
    else:
        with PLAN.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


def prepare():
    plan = commit()
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "sealed_manifest.json").exists():
        return json.loads((ROOT / "sealed_manifest.json").read_text())
    cases = load_reviewed_registry()
    if digest("data/raw/lopucki/cases.download") != plan["registry_sha256"]:
        raise ValueError("Outcome registry changed")
    events = cases.loc[cases.Chapter.eq("11")].sort_values("filing_date").drop_duplicates("cik").set_index("cik")
    frames, filings, companies, sources, attrition, errors, extraction_issues = [], [], [], [], [], [], []
    for original in plan["companies"]:
        company = dict(original)
        cik = company["cik"]
        if cik in plan["excluded_exposed_companies"]:
            attrition.append({"cik": cik, "reason": "previously_exposed_company"})
            continue
        print(f"Preparing population audit company {cik}", flush=True)
        if cik in events.index:
            company.update(event_date=events.loc[cik, "filing_date"],
                           event_source="https://lopucki.law.ufl.edu/download_cases_table.php")
        companies.append(company)
        try:
            payload, provenance = companyfacts(cik, RAW / "companyfacts")
            sources.append(provenance)
            if str(payload["cik"]).zfill(10) != cik:
                raise ValueError("Companyfacts identity mismatch")
            rows, _ = load_submissions(company, "2010-01-01", plan["observation_end"], RAW, sources)
            frame, _, issues = extract_history(payload, company["ticker"], public_filings=rows)
            extraction_issues.extend({"cik": cik, **issue} for issue in issues)
            frames.append(frame)
            filings.extend(rows)
        except requests.HTTPError as exc:
            item = {"cik": cik, "reason": str(exc)}
            (attrition if exc.response is not None and exc.response.status_code == 404 else errors).append(item)
        except ValueError as exc:
            destination = attrition if str(exc).startswith("No original 10-K/10-Q USD Assets facts") else errors
            destination.append({"cik": cik, "reason": str(exc)})
        except Exception as exc:
            errors.append({"cik": cik, "reason": str(exc)})
    write_json(ROOT / "acquisition_report.json", {"sources": sources, "attrition": attrition,
        "errors": errors, "extraction_issues": extraction_issues})
    if errors or not frames:
        raise ValueError("Source acquisition incomplete; retry the same sample without substitutions")
    all_rows = pd.concat(frames, ignore_index=True)
    anchors = all_rows.loc[all_rows.filing_date.between("2020-01-01", "2020-12-31")].sort_values(["filing_date", "accession_number"]).drop_duplicates("cik")
    # Select anchors BEFORE reviewing their labels or their eligibility flags.
    keys = set(zip(anchors.cik, anchors.accession_number))
    labeled = financial_labels(all_rows, companies, plan["observation_end"])
    labeled, evidence = apply_historical_coverage(labeled, cases, pd.DataFrame(filings), observation_end=plan["observation_end"])
    selected = labeled.loc[[pair in keys for pair in zip(labeled.cik, labeled.accession_number)]].copy()
    eligible = selected.loc[selected.historical_coverage_eligible & selected.valid_for_training].copy()
    eligible.to_csv(ROOT / "test_features.csv", index=False)
    selected.to_csv(ROOT / "all_anchors.csv", index=False)
    evidence.to_csv(ROOT / "coverage_evidence.csv", index=False)
    missing = set(c["cik"] for c in plan["companies"]) - set(eligible.cik)
    sealed = {"plan_sha256": digest(PLAN), "features_sha256": digest(ROOT / "test_features.csv"),
        "sampled_companies": len(plan["companies"]), "eligible_companies": eligible.cik.nunique(),
        "companies_without_eligible_anchor": sorted(missing), "source_attrition": attrition,
        "positive_companies": int(eligible.target_distressed_1yr.eq(1).sum()), "scored": False}
    write_json(ROOT / "sealed_manifest.json", sealed)
    return sealed


def calibration_metrics(target, scores):
    y, p = np.asarray(target), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Calibration requires equally sized nonempty one-dimensional arrays")
    if not set(y) <= {0, 1} or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid calibration targets or scores")
    y = y.astype(int)
    bins = []
    for lo, hi in zip([0, .05, .1, .2, .4, .6, .8], [.05, .1, .2, .4, .6, .8, 1.000001]):
        mask = (p >= lo) & (p < hi)
        bins.append({"lower": lo, "upper": min(hi, 1.), "companies": int(mask.sum()),
            "mean_score": float(p[mask].mean()) if mask.any() else None,
            "event_fraction": float(y[mask].mean()) if mask.any() else None})
    return {"companies": len(y), "observed_events": int(y.sum()), "observed_fraction": float(y.mean()),
        "mean_score": float(p.mean()), "expected_events_from_scores": float(p.sum()),
        "brier_loss": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "constant_sample_prevalence_brier": float(y.mean() * (1 - y.mean())), "bins": bins}


def evaluate_once():
    plan = commit()
    sealed = json.loads((ROOT / "sealed_manifest.json").read_text())
    if digest(PLAN) != sealed["plan_sha256"] or digest(ROOT / "test_features.csv") != sealed["features_sha256"]:
        raise ValueError("Sealed calibration inputs changed")
    artifact = load_model(CANDIDATE / "candidate.joblib", expected_sha256=plan["candidate_artifact_sha256"])
    frame = pd.read_csv(ROOT / "test_features.csv", dtype={"cik": str})
    if frame.empty or frame.cik.duplicated().any() or set(frame.cik) & set(plan["excluded_exposed_companies"]):
        raise ValueError("Empty, repeated or previously exposed population anchors")
    if not set(frame.cik) <= {c["cik"] for c in plan["companies"]}:
        raise ValueError("Unplanned population test company")
    dates = pd.to_datetime(frame.filing_date)
    available = pd.to_datetime(frame.label_available_date_1yr)
    input_dates = pd.to_datetime(frame.latest_input_filing_date)
    if dates.isna().any() or available.isna().any() or input_dates.isna().any():
        raise ValueError("Population anchor dates cannot be missing")
    if not dates.between("2020-01-01", "2020-12-31").all() or (input_dates > dates).any():
        raise ValueError("Population anchor dates or inputs violate the frozen protocol")
    if (available < dates).any() or (available > pd.Timestamp(plan["observation_end"])).any():
        raise ValueError("Population outcomes fall outside the observation horizon")
    negative = frame.target_distressed_1yr.eq(0)
    if (available.loc[negative] < dates.loc[negative] + pd.DateOffset(years=1)).any():
        raise ValueError("Population negatives need completed one-year outcomes")
    if dates.min() < pd.Timestamp(artifact["trained_through"]):
        raise ValueError("Candidate training postdates population anchors")
    with (ROOT / "evaluation_started.json").open("x") as marker:
        json.dump({"artifact_sha256": plan["candidate_artifact_sha256"], "plan_sha256": sealed["plan_sha256"]}, marker)
    model = artifact["model"]
    scores = model.predict_proba(frame[artifact["feature_cols"]].astype(float).replace([np.inf, -np.inf], np.nan))[:, list(model.classes_).index(1)]
    metrics = calibration_metrics(frame.target_distressed_1yr, scores)
    report = {"metrics": metrics, "attrition": sealed, "candidate_artifact_sha256": plan["candidate_artifact_sha256"],
        "plan_sha256": sealed["plan_sha256"], "evaluated_once": True, "model_refitted": False,
        "threshold_changed": False, "validated_calibration": False, "representative_population": False,
        "production_ready": False, "limitations": plan["limitations"],
        "decision": "Descriptive historical natural-sample audit only; limited outcomes and selection/coverage exclusions prevent population-wide calibration certification."}
    frame[["cik", "filing_date", "target_distressed_1yr"]].assign(score=scores).to_csv(ROOT / "predictions.csv", index=False)
    write_json(ROOT / "report.json", report)
    write_json(ROOT / "sealed_manifest.json", {**sealed, "scored": True})
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate-once", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
    elif args.evaluate_once:
        evaluate_once()
    else:
        print(json.dumps(commit(), indent=2))
