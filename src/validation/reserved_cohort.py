"""Commit and materialize an unscored, company-disjoint final evaluation cohort."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data_layer.lopucki import parse_archive
from src.data_layer.research_dataset import load_submissions, financial_labels, write_json
from src.ml.financial_history import extract_history
from src.validation.download import companyfacts
from src.validation.historical_coverage import apply_historical_coverage

SEED = "new-holdout-20261017"
CONTROLS = [("NKE", "0000320187"), ("PEP", "0000077476"), ("KO", "0000021344"),
    ("PG", "0000080424"), ("CVS", "0000064803"), ("UPS", "0001090727"),
    ("FDX", "0001048911"), ("CAT", "0000018230")]


def commit(destination="config/reserved_cohort_20261017.json"):
    cases, _ = parse_archive("data/raw/lopucki/cases.download")
    exposed = {c["cik"] for c in json.loads(Path("config/research_recent.json").read_text())["companies"]}
    original = pd.read_csv("data/processed/validation/expanded/features.csv", dtype={"cik": str})
    exposed |= set(original.cik)
    candidates = cases.loc[cases.Chapter.eq("11") & cases.cik.notna()].sort_values("filing_date").drop_duplicates("cik")
    candidates = candidates.loc[candidates.filing_date.between("2020-01-01", "2022-12-31") & ~candidates.cik.isin(exposed)].copy()
    candidates["rank"] = candidates.cik.map(lambda c: sha256(f"{SEED}:{c}".encode()).hexdigest())
    failures = [{"cik": r.cik, "ticker": f"CIK{r.cik}", "company_name": r.NameCorp,
        "event_date": r.filing_date, "event_source": "https://lopucki.law.ufl.edu/download_cases_table.php"}
        for r in candidates.sort_values("rank").head(8).itertuples()]
    controls = [{"ticker": t, "cik": c, "event_date": None} for t, c in CONTROLS]
    identifiers = {c["cik"] for c in failures + controls}
    if len(failures) != 8 or len(identifiers) != 16 or identifiers & exposed:
        raise ValueError("New cohort must contain 16 unique, previously unexposed companies")
    manifest = {"seed": SEED, "created_on": "2026-10-17", "train_labels_available_before": "2020-01-01",
        "test_filing_start": "2020-01-01", "test_filing_end": "2021-12-31", "observation_end": "2022-12-31",
        "companies": failures + controls, "excluded_development_companies": sorted(exposed),
        "sampling": "Prespecified case-control benchmark: 8 first BRD Chapter 11 cases selected by seeded CIK hash, 8 named public-company controls. Not market-prevalence representative.",
        "policy": "Never replace unavailable or ineligible firms based on scores/classes. Test features never enter fitting, preprocessing, model selection or threshold choice."}
    path = Path(destination)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("The reserved cohort is immutable; do not reshuffle an exposed test")
    write_json(path, manifest)
    return manifest


def materialize():
    manifest = commit()  # Written BEFORE downloading or inspecting any feature values.
    root = Path("data/processed/validation/reserved_20261017")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "sealed_manifest.json").exists():
        return json.loads((root / "sealed_manifest.json").read_text())
    cases, _ = parse_archive("data/raw/lopucki/cases.download")
    frames, filings, failures, provenance = [], [], [], []
    raw = Path("data/raw/reserved_20261017")
    for company in manifest["companies"]:
        print("Preparing reserved company", company["cik"], flush=True)
        try:
            payload, source = companyfacts(company["cik"], directory=raw / "companyfacts")
            rows, _ = load_submissions(company, "2010-01-01", manifest["observation_end"], raw, provenance)
            frame, _, _ = extract_history(payload, company["ticker"], public_filings=rows)
            frames.append(frame)
            filings.extend(rows)
            provenance.append(source)
        except Exception as exc:
            failures.append({"cik": company["cik"], "error": str(exc)})
    if failures:
        write_json(root / "preparation_failures.json", failures)
        raise ValueError("Reserved source failures; retry without substituting companies")
    frame = financial_labels(pd.concat(frames, ignore_index=True), manifest["companies"], manifest["observation_end"])
    frame, evidence = apply_historical_coverage(frame, cases, pd.DataFrame(filings), observation_end=manifest["observation_end"])
    frame = frame.loc[frame.historical_coverage_eligible & frame.valid_for_training &
        frame.filing_date.between(manifest["test_filing_start"], manifest["test_filing_end"])].copy()
    frame = frame.sort_values("filing_date").drop_duplicates(["cik", "fiscal_period_end"])
    frame.to_csv(root / "test_features.csv", index=False)
    evidence.to_csv(root / "coverage_evidence.csv", index=False)
    report = {"cohort_manifest_sha256": sha256(Path("config/reserved_cohort_20261017.json").read_bytes()).hexdigest(),
        "features_sha256": sha256((root / "test_features.csv").read_bytes()).hexdigest(),
        "rows": len(frame), "companies_with_eligible_rows": frame.cik.nunique(), "requested_companies": 16,
        "scored": False, "preprocessing_fitted": False, "final_metrics": None,
        "purpose": "Sealed unscored final cohort; fit and choose models on development only before evaluating once.",
        "limitations": ["Case-control sample; calibration and precision do not represent natural market prevalence.",
                        "Ineligible/missing-history companies are not replaced; cohort eligibility losses remain explicit."],
        "source_manifest": provenance}
    write_json(root / "sealed_manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_manifest"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        materialize()
    else:
        print(json.dumps(commit(), indent=2))
