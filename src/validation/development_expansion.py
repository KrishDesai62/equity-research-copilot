"""Acquire prespecified historical development cases, never a new final test."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import financial_labels, load_submissions, write_json
from src.ml.financial_history import extract_history
from src.validation.download import companyfacts
from src.validation.historical_coverage import apply_historical_coverage, load_reviewed_registry

ROOT = Path("data/processed/validation/development_expansion")
RAW = Path("data/raw/development_expansion")


def commit_plan():
    policy = json.loads(Path("config/calibration_development.json").read_text())
    cases = load_reviewed_registry()
    base = pd.read_csv("data/processed/validation/reserved_20261017/development_features.csv", dtype={"cik": str})
    excluded = set(base.cik)
    excluded |= {c["cik"] for c in json.loads(Path("config/reserved_cohort_20261017.json").read_text())["companies"]}
    old = json.loads(Path("data/processed/validation/expanded/committed_holdout.json").read_text())
    excluded |= set(old["holdout_companies"]) | set(old["development_companies"])
    # The earliest registry event must itself be Chapter 11; no pre-existing case
    # is silently discarded in order to find a convenient later event.
    candidates = cases.dropna(subset=["cik"]).sort_values("filing_date").drop_duplicates("cik")
    candidates = candidates.loc[candidates.Chapter.eq("11") & ~candidates.cik.isin(excluded)].copy()
    candidates["role"] = candidates.cik.map(lambda c: int(sha256(f"{policy['seed']}:{c}".encode()).hexdigest(), 16) % 4)
    candidates["rank"] = candidates.cik.map(lambda c: sha256(f"expanded-development-v1:{c}".encode()).hexdigest())
    companies = []
    for stage, roles, start, end in (("training", [0, 1], "2014-01-01", "2015-12-31"),
            ("calibration", [2], "2017-01-01", "2017-12-31"),
            ("threshold_tuning", [3], "2019-01-01", "2019-12-31")):
        selected = candidates.loc[candidates.role.isin(roles) & candidates.filing_date.between(start, end)].sort_values("rank").head(6)
        for row in selected.itertuples():
            companies.append({"cik": row.cik, "ticker": f"CIK{row.cik}", "company_name": row.NameCorp,
                "event_date": row.filing_date, "event_source": "https://lopucki.law.ufl.edu/download_cases_table.php",
                "planned_stage": stage})
    manifest = {"version": 1, "purpose": "Positive-enriched historical DEVELOPMENT data; not representative and not a final test",
        "policy_sha256": sha256(Path("config/calibration_development.json").read_bytes()).hexdigest(),
        "registry_sha256": sha256(Path("data/raw/lopucki/cases.download").read_bytes()).hexdigest(),
        "companies": companies, "excluded_companies": sorted(excluded),
        "missing_source_policy": "Retain failures explicitly; never substitute cases based on scores or observed class counts."}
    ROOT.mkdir(parents=True, exist_ok=True)
    committed = Path("config/development_acquisition_20261017.json")
    if committed.exists():
        if json.loads(committed.read_text()) != manifest:
            raise ValueError("Committed development acquisition manifest changed")
    else:
        with committed.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    path = ROOT / "acquisition_plan.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Development acquisition plan changed; preserve the previous experiment")
    else:
        with path.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


def run():
    manifest = commit_plan()  # Fixed before obtaining new feature values.
    existing_report = ROOT / "report.json"
    if existing_report.exists():
        saved = json.loads(existing_report.read_text())
        features = ROOT / "reviewed_features.csv"
        if not saved.get("failures") and features.exists():
            if saved.get("features_sha256") != sha256(features.read_bytes()).hexdigest():
                raise ValueError("Frozen development acquisition features changed")
            return saved  # Preserve inputs already used by archived experiments.
    cases = load_reviewed_registry()
    frames, filings, provenance, failures = [], [], [], []
    for company in manifest["companies"]:
        print(f"Acquiring development case {company['cik']} ({company['planned_stage']})", flush=True)
        try:
            payload, source = companyfacts(company["cik"], directory=RAW / "companyfacts")
            provenance.append(source)
            if str(payload["cik"]).zfill(10) != company["cik"]:
                raise ValueError("Companyfacts CIK mismatch")
            rows, _ = load_submissions(company, "2010-01-01", "2019-12-31", RAW, provenance)
            frame, _, _ = extract_history(payload, company["ticker"], public_filings=rows)
            frames.append(frame)
            filings.extend(rows)
        except Exception as exc:
            failures.append({"cik": company["cik"], "reason": str(exc)})
    report = {"companies_requested": len(manifest["companies"]), "companies_downloaded": len(frames),
        "failures": failures, "source_manifest": provenance, "production_ready": False,
        "purpose": manifest["purpose"], "final_test_read": False, "final_test_rescored": False}
    if frames:
        combined = financial_labels(pd.concat(frames, ignore_index=True), manifest["companies"], "2019-12-31")
        labeled, evidence = apply_historical_coverage(combined, cases, pd.DataFrame(filings), observation_end="2019-12-31")
        labeled.to_csv(ROOT / "reviewed_features.csv", index=False)
        evidence.to_csv(ROOT / "coverage_evidence.csv", index=False)
        report.update({"rows": len(labeled), "eligible_rows": int((labeled.valid_for_training & labeled.historical_coverage_eligible).sum()),
            "positive_companies": int(labeled.loc[labeled.valid_for_training & labeled.historical_coverage_eligible & labeled.target_distressed_1yr.eq(1), "cik"].nunique()),
            "features_sha256": sha256((ROOT / "reviewed_features.csv").read_bytes()).hexdigest()})
    write_json(ROOT / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_manifest"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = run() if args.download else commit_plan()
    if not args.download:
        print(json.dumps(result, indent=2))
    if args.download and result["failures"]:
        raise SystemExit(2)
