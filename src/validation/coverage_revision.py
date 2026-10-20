"""Export corrected development eligibility without rescoring archived models."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.historical_coverage import apply_historical_coverage, load_reviewed_registry

ROOT = Path("data/processed/validation/coverage_revision_v2")


def run():
    cases = load_reviewed_registry()
    excluded = set(json.loads(Path("data/processed/validation/expanded/committed_holdout.json").read_text())["holdout_companies"])
    for name in ("reserved_cohort_20261017", "independent_candidate_20261017", "population_calibration_20261017"):
        excluded |= {c["cik"] for c in json.loads(Path(f"config/{name}.json").read_text())["companies"]}
    config = json.loads(Path("config/improvement_v2.json").read_text())
    natural = pd.read_csv("data/processed/validation/improvement_v2/natural_features.csv", dtype={"cik": str})
    # Remove every final-role company, including all its earlier history.
    final_role = natural.cik.map(lambda c: int(sha256(f"{config['seed']}:{c}".encode()).hexdigest(), 16) % 4 == 3)
    excluded |= set(natural.loc[final_role, "cik"])
    sources = [Path("data/processed/research/financial_features.csv"),
               Path("data/processed/validation/improvement_v2/natural_features.csv")]
    current = pd.read_csv(sources[0], dtype={"cik": str})
    filings = pd.DataFrame([json.loads(line) for line in Path("data/processed/research/filings.jsonl").read_text().splitlines()])
    more = []
    for path in Path("data/processed/validation/improvement_v2/company_checkpoints").glob("*.json"):
        more.extend(json.loads(path.read_text()).get("filings", []))
    all_filings = pd.concat([filings, pd.DataFrame(more)], ignore_index=True).drop_duplicates(["cik", "accession_number"])
    frame = pd.concat([current, natural], ignore_index=True)
    frame = frame.loc[~frame.cik.isin(excluded) & pd.to_datetime(frame.filing_date).lt("2020-01-01")].copy()
    original, _ = apply_historical_coverage(frame, cases, all_filings, observation_end="2022-12-31", positive_followup="full_horizon")
    revised, evidence = apply_historical_coverage(frame, cases, all_filings, observation_end="2022-12-31")
    changed = original.historical_coverage_eligible.ne(revised.historical_coverage_eligible)
    changes = revised.loc[changed, ["cik", "accession_number", "filing_date", "target_distressed_1yr"]].copy()
    changes["previous_eligible"] = original.loc[changed, "historical_coverage_eligible"]
    changes["revised_eligible"] = revised.loc[changed, "historical_coverage_eligible"]
    ROOT.mkdir(parents=True, exist_ok=True)
    changes.to_csv(ROOT / "eligibility_changes.csv", index=False)
    evidence.to_csv(ROOT / "coverage_evidence.csv", index=False)
    development = revised.loc[revised.historical_coverage_eligible & revised.target_distressed_1yr.notna() &
        pd.to_datetime(revised.label_available_date_1yr).lt("2020-01-01")].copy()
    development.to_csv(ROOT / "reviewed_development.csv", index=False)
    report = {"policy_version": 2, "input_rows": len(frame), "changed_eligibility_rows": int(changed.sum()),
        "development_rows": len(development), "development_companies": development.cik.nunique(),
        "positive_rows": int(development.target_distressed_1yr.eq(1).sum()),
        "positive_companies": development.loc[development.target_distressed_1yr.eq(1), "cik"].nunique(),
        "source_hashes": {str(p): sha256(p.read_bytes()).hexdigest() for p in sources},
        "excluded_company_count": len(excluded), "final_tests_rescored": False, "models_retrained": False,
        "note": "Future development inputs only. Confirmed positives stop follow-up at the registry event; negatives retain full one-year follow-up. Archived experiment cohorts are preserved."}
    write_json(ROOT / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
