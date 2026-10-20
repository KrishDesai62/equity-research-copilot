"""Versioned natural development panels; prior evaluated companies stay excluded."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pandas as pd
import requests

from src.data_layer.research_dataset import financial_labels, load_submissions, write_json
from src.ml.financial_history import extract_history
from src.validation.download import companyfacts
from src.validation.historical_coverage import apply_historical_coverage, load_reviewed_registry

ROOT = Path("data/processed/validation/improvement_v2")
RAW = Path("data/raw/improvement_v2")
CONFIG = Path("config/improvement_v2.json")
PLAN = Path("config/improvement_v2_acquisition.json")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def exposed_companies():
    excluded = set()
    for path in ("config/reserved_cohort_20261017.json", "config/independent_candidate_20261017.json",
                 "config/population_calibration_20261017.json", "config/development_acquisition_20261017.json"):
        excluded |= {c["cik"] for c in json.loads(Path(path).read_text())["companies"]}
    for path in ("data/processed/validation/reserved_20261017/development_features.csv",
                 "data/processed/validation/expanded/features.csv"):
        excluded |= set(pd.read_csv(path, usecols=["cik"], dtype={"cik": str}).cik)
    return excluded


def sample_vintage(frame, seed, year, size, excluded):
    frame = frame.copy()
    frame["cik"] = frame.cik.astype(str).str.zfill(10)
    sic = pd.to_numeric(frame.sic, errors="coerce")
    frame = frame.loc[frame.form.eq("10-K") & sic.notna() & ~sic.between(6000, 6999) & ~frame.cik.isin(excluded)].copy()
    frame["filed"] = pd.to_datetime(frame.filed.astype(str), format="%Y%m%d", errors="raise")
    frame = frame.sort_values(["filed", "cik"]).drop_duplicates("cik")
    frame["rank"] = frame.cik.map(lambda c: sha256(f"{seed}:{year}:{c}".encode()).hexdigest())
    return frame.sort_values("rank").head(size)


def commit():
    config = json.loads(CONFIG.read_text())
    excluded = exposed_companies()
    companies, universes = {}, []
    for year in config["vintages"]:
        archive = RAW / "bulk" / f"{year}q{config['quarter']}.zip"
        with zipfile.ZipFile(archive) as source:
            frame = pd.read_csv(source.open("sub.txt"), sep="\t", dtype={"cik": str}, low_memory=False)
        selected = sample_vintage(frame, config["seed"], year, config["issuers_per_vintage"], excluded)
        if len(selected) != config["issuers_per_vintage"]:
            raise ValueError("Historical sampling frame too small; do not substitute after inspecting outcomes")
        universes.append({"year": year, "sha256": digest(archive)})
        for row in selected.itertuples(index=False):
            entry = row.filed.date().isoformat()
            if row.cik not in companies or entry < companies[row.cik]["entry_date"]:
                companies[row.cik] = {"cik": row.cik, "ticker": f"CIK{row.cik}", "company_name": row.name, "entry_date": entry}
    baseline = Path("data/processed/validation/calibration_expanded/candidate.joblib")
    manifest = {"config_sha256": digest(CONFIG), "registry_sha256": digest("data/raw/lopucki/cases.download"),
        "baseline_sha256": digest(baseline), "universes": universes,
        "companies": [companies[c] for c in sorted(companies)], "excluded_companies": sorted(excluded),
        "selection": "Separate identifier-hash samples of historical nonfinancial annual issuers; no outcome or score-based selection. Only post-entry rows may enter an experiment.",
        "limits": config["limits"]}
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != manifest:
            raise ValueError("Committed v2 acquisition plan changed")
    else:
        with PLAN.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


def prepare():
    plan = commit()  # Commit before inspecting/downloading any company features.
    config = json.loads(CONFIG.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "data_manifest.json").exists():
        record = json.loads((ROOT / "data_manifest.json").read_text())
        if record["plan_sha256"] != digest(PLAN) or record["features_sha256"] != digest(ROOT / "natural_features.csv"):
            raise ValueError("Committed v2 features changed")
        return record
    cases = load_reviewed_registry()
    events = cases.loc[cases.Chapter.eq("11")].sort_values("filing_date").drop_duplicates("cik").set_index("cik")
    frames, filings, companies, provenance, issues, attrition, errors = [], [], [], [], [], [], []
    checkpoints = ROOT / "company_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for position, original in enumerate(plan["companies"], 1):
        company = dict(original)
        cik = company["cik"]
        if cik in events.index:
            company.update(event_date=events.loc[cik, "filing_date"], event_source="https://lopucki.law.ufl.edu/download_cases_table.php")
        companies.append(company)
        marker = checkpoints / f"{cik}.json"
        print(f"V2 public-source acquisition {position}/{len(plan['companies'])}: {cik}", flush=True)
        try:
            if marker.exists():
                record = json.loads(marker.read_text())
            else:
                sources = []
                payload, source = companyfacts(cik, RAW / "companyfacts")
                sources.append(source)
                if str(payload["cik"]).zfill(10) != cik:
                    raise ValueError("Companyfacts identity mismatch")
                metadata, _ = load_submissions(company, "2010-01-01", config["observation_end"], RAW, sources)
                frame, _, diagnostics = extract_history(payload, company["ticker"], public_filings=metadata)
                # Derived annual features still include quarter-derived TTM/trends.
                annual = frame.loc[frame.filing_type.eq("10-K")].copy()
                path = checkpoints / f"{cik}.csv"
                annual.to_csv(path, index=False)
                record = {"status": "complete", "features_sha256": digest(path),
                    "filings": [r for r in metadata if r["filing_type"] == "10-K"],
                    "sources": sources, "issues": diagnostics}
                write_json(marker, record)
            if record["status"] == "attrition":
                attrition.append({"cik": cik, "reason": record["reason"]})
            else:
                path = checkpoints / f"{cik}.csv"
                if digest(path) != record["features_sha256"]:
                    raise ValueError("Acquisition checkpoint changed")
                frames.append(pd.read_csv(path, dtype={"cik": str}))
                filings.extend(record["filings"])
                provenance.extend(record["sources"])
                issues.extend({"cik": cik, **issue} for issue in record["issues"])
        except Exception as exc:
            permanent = (isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 404)
            permanent |= isinstance(exc, ValueError) and str(exc).startswith("No original 10-K/10-Q USD Assets facts")
            item = {"cik": cik, "reason": str(exc)}
            if permanent:
                attrition.append(item)
                write_json(marker, {"status": "attrition", **item})
            else:
                errors.append(item)
        if position % 10 == 0:
            write_json(ROOT / "progress.json", {"processed": position, "requested": len(plan["companies"]),
                "complete": len(frames), "attrition": len(attrition), "errors": len(errors)})
    write_json(ROOT / "acquisition_report.json", {"sources": provenance, "issues": issues, "attrition": attrition, "errors": errors})
    if errors or not frames:
        raise ValueError("Public-source acquisition incomplete; resume the same plan using its checkpoints")
    labeled = financial_labels(pd.concat(frames, ignore_index=True), companies, config["observation_end"])
    # Preserve the stricter eligibility rule of the already sealed V2 experiment.
    # Future experiments use the corrected event-stop policy by default.
    labeled, evidence = apply_historical_coverage(labeled, cases, pd.DataFrame(filings),
        observation_end=config["observation_end"], positive_followup="full_horizon")
    entry = {c["cik"]: c["entry_date"] for c in plan["companies"]}
    labeled["sampling_entry_date"] = labeled.cik.map(entry)
    labeled = labeled.loc[pd.to_datetime(labeled.filing_date) >= pd.to_datetime(labeled.sampling_entry_date)].copy()
    labeled.to_csv(ROOT / "natural_features.csv", index=False)
    evidence.to_csv(ROOT / "coverage_evidence.csv", index=False)
    record = {"plan_sha256": digest(PLAN), "features_sha256": digest(ROOT / "natural_features.csv"),
        "requested_companies": len(plan["companies"]), "companies_with_history": labeled.cik.nunique(),
        "annual_rows": len(labeled), "eligible_rows": int((labeled.historical_coverage_eligible & labeled.valid_for_training).sum()),
        "attrition": attrition, "production_ready": False,
        "note": "No final-role metrics or labels are inspected for candidate selection."}
    write_json(ROOT / "data_manifest.json", record)
    return record
