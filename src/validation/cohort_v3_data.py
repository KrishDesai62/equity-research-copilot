"""Probability-sampled historical bankruptcy cohort with explicit source evidence."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from src.data_layer.research_dataset import financial_labels, load_submissions, write_json
from src.data_layer.xbrl_bulk import download_quarterly_zip
from src.ml.financial_history import extract_history
from src.validation.download import companyfacts
from src.validation.historical_coverage import apply_historical_coverage, load_reviewed_registry

ROOT = Path("data/processed/validation/cohort_v3")
RAW = Path("data/raw/cohort_v3")
CONFIG = Path("config/cohort_v3.json")
PLAN = Path("config/cohort_v3_acquisition.json")
TARGET = "target_bankruptcy_next_1yr"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def uniform(seed, key):
    return int(sha256(f"{seed}:{key}".encode()).hexdigest(), 16) / 2**256


def ensure_archives():
    config = json.loads(CONFIG.read_text())
    dest = RAW / "bulk"
    dest.mkdir(parents=True, exist_ok=True)
    cached = {p.name: p for p in Path("data/raw").rglob("*q*.zip") if p.parent != dest}
    for year in config["universe_years"]:
        for quarter in range(1, 5):
            name = f"{year}q{quarter}.zip"
            if not (dest / name).exists() and name in cached:
                shutil.copyfile(cached[name], dest / name)
            for attempt in range(3):
                try:
                    download_quarterly_zip(year, quarter, dest, cache=False)
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)


def prior_exposures():
    excluded = set()
    for path in Path("config").glob("*.json"):
        if path in {CONFIG, PLAN}:
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("companies"), list):
            excluded |= {str(c["cik"]).zfill(10) for c in payload["companies"] if isinstance(c, dict) and c.get("cik")}
    for path in (Path("data/processed/validation/expanded/features.csv"),
                 Path("data/processed/validation/reserved_20261017/development_features.csv")):
        if path.exists():
            excluded |= set(pd.read_csv(path, usecols=["cik"], dtype={"cik": str}).cik)
    return excluded


def sampling_record(cik, is_case, config):
    probability = 1. if is_case else config["background_probability"]
    selected = is_case or uniform(config["seed"] + ":sample", cik) < probability
    role_number = int(sha256(f"{config['seed']}:role:{cik}".encode()).hexdigest(), 16) % 3
    return {"selected": bool(selected), "inclusion_probability": probability,
        "design_weight": 1. / probability, "sampling_stratum": "registry_case_company" if is_case else "background",
        "role": ("training", "calibration", "final")[role_number]}


def commit():
    config = json.loads(CONFIG.read_text())
    if not 0 < config["background_probability"] <= 1:
        raise ValueError("Invalid background inclusion probability")
    cases = load_reviewed_registry()
    first = cases.dropna(subset=["cik"]).sort_values("filing_date").drop_duplicates("cik")
    prior = set(first.loc[first.filing_date.le("2011-12-31"), "cik"])
    case_ids = set(first.loc[first.filing_date.between("2012-01-01", config["observation_end"]), "cik"])
    blocks, sources = [], []
    for year in config["universe_years"]:
        for quarter in range(1, 5):
            path = RAW / "bulk" / f"{year}q{quarter}.zip"
            with zipfile.ZipFile(path) as archive:
                frame = pd.read_csv(archive.open("sub.txt"), sep="\t", dtype={"cik": str, "filed": str}, low_memory=False)
            frame = frame.loc[frame.form.eq("10-K"), ["cik", "name", "filed", "sic"]].copy()
            frame["cik"] = frame.cik.str.zfill(10)
            frame["filed"] = pd.to_datetime(frame.filed, format="%Y%m%d", errors="raise").dt.strftime("%Y-%m-%d")
            blocks.append(frame)
            sources.append({"year": year, "quarter": quarter, "sha256": digest(path)})
    frame = pd.concat(blocks, ignore_index=True).drop_duplicates(["cik", "filed", "sic"]).sort_values(["filed", "cik"])
    excluded = prior_exposures() | prior
    sic = pd.to_numeric(frame.sic, errors="coerce")
    potential = set(frame.loc[sic.notna() & ~sic.between(6000, 6999), "cik"]) - excluded
    selected = []
    names = frame.drop_duplicates("cik").set_index("cik")["name"].to_dict()
    for cik in sorted(potential):
        record = sampling_record(cik, cik in case_ids, config)
        if record.pop("selected"):
            selected.append({"cik": cik, "ticker": f"CIK{cik}", "company_name": names[cik], **record})
    manifest = {"version": 3, "config_sha256": digest(CONFIG), "registry_sha256": digest("data/raw/lopucki/cases.download"),
        "universe_archives": sources, "frame_companies": len(potential), "excluded_companies": sorted(excluded),
        "companies": selected, "design": config["sampling"],
        "limits": "Weights address known design sampling, not source nonresponse, eligibility censoring or registry/legal identity errors."}
    ROOT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != manifest:
            raise ValueError("Committed V3 acquisition plan changed")
    else:
        with PLAN.open("x") as handle:
            json.dump(manifest, handle, indent=2)
    frame.to_csv(ROOT / "annual_frame.csv", index=False)
    return manifest


def make_snapshots(history, annual_frame, company, config):
    """Prediction date is distinct from the filing date of the selected inputs."""
    keys = {"training": "train_anchors", "calibration": "calibration_anchors", "final": "final_anchors"}
    snapshots, exclusions = [], []
    for anchor in config[keys[company["role"]]]:
        public = history.loc[pd.to_datetime(history.filing_date).lt(anchor)].sort_values(["filing_date", "accession_number"])
        members = annual_frame.loc[annual_frame.filed.lt(anchor)].sort_values("filed")
        reason = None
        if public.empty or members.empty:
            reason = "no_original_public_financials_or_annual_membership"
        else:
            industry = pd.to_numeric(members.iloc[-1].sic, errors="coerce")
            row = public.iloc[-1].to_dict()
            age = (pd.Timestamp(anchor) - pd.Timestamp(row["filing_date"])).days
            if pd.isna(industry) or 6000 <= industry <= 6999:
                reason = "missing_or_financial_industry_at_anchor"
            elif age > config["max_feature_age_days"]:
                reason = "stale_financial_inputs_abstain"
            else:
                row.update(feature_filing_date=row["filing_date"], source_accession_number=row["accession_number"],
                    source_filing_type=row["filing_type"], prediction_date=anchor, feature_age_days=age,
                    filing_date=anchor, filing_type="SNAPSHOT", accession_number=f"{row['accession_number']}@{anchor}",
                    unit_id=f"{company['cik']}@{anchor}", role=company["role"],
                    inclusion_probability=company["inclusion_probability"], design_weight=company["design_weight"],
                    sampling_stratum=company["sampling_stratum"], sic=int(industry),
                    log_total_assets=float(np.log1p(row["Assets"])) if row["Assets"] >= 0 else np.nan)
                snapshots.append(row)
        if reason:
            exclusions.append({"cik": company["cik"], "prediction_date": anchor, "reason": reason})
    return snapshots, exclusions


def prepare():
    plan = commit()
    config = json.loads(CONFIG.read_text())
    sealed_path = ROOT / "dataset_manifest.json"
    if sealed_path.exists():
        sealed = json.loads(sealed_path.read_text())
        if sealed["plan_sha256"] != digest(PLAN) or sealed["dataset_sha256"] != digest(ROOT / "dataset.csv"):
            raise ValueError("V3 dataset integrity failure")
        return sealed
    cases = load_reviewed_registry()
    first = cases.dropna(subset=["cik"]).sort_values("filing_date").drop_duplicates("cik").set_index("cik")
    frame = pd.read_csv(ROOT / "annual_frame.csv", dtype={"cik": str, "filed": str})
    checkpoints = ROOT / "company_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    errors, exclusions, sources, rows, issues = [], [], [], [], []
    for index, base in enumerate(plan["companies"], 1):
        company = dict(base)
        cik = company["cik"]
        print(f"V3 acquisition {index}/{len(plan['companies'])}: {cik}", flush=True)
        if cik in first.index:
            company.update(event_date=first.loc[cik, "filing_date"], event_source="https://lopucki.law.ufl.edu/download_cases_table.php")
        marker = checkpoints / f"{cik}.json"
        try:
            if marker.exists():
                record = json.loads(marker.read_text())
            else:
                provenance = []
                payload, source = companyfacts(cik, RAW / "companyfacts")
                provenance.append(source)
                if str(payload["cik"]).zfill(10) != cik:
                    raise ValueError("Companyfacts identity mismatch")
                metadata, _ = load_submissions(company, "2010-01-01", config["observation_end"], RAW, provenance)
                history, _, diagnostics = extract_history(payload, company["ticker"], public_filings=metadata)
                links = {m["accession_number"]: m.get("source_url") for m in metadata}
                history["source_url"] = history.accession_number.map(links)
                snapshots, skipped = make_snapshots(history, frame.loc[frame.cik.eq(cik)], company, config)
                if snapshots:
                    annuals = history.loc[history.filing_type.eq("10-K")]
                    combined = financial_labels(pd.concat([annuals, pd.DataFrame(snapshots)], ignore_index=True), [company], config["observation_end"])
                    labeled, evidence = apply_historical_coverage(combined, cases, pd.DataFrame(metadata), observation_end=config["observation_end"])
                    result = labeled.loc[labeled.filing_type.eq("SNAPSHOT")].copy()
                    records = json.loads(result.to_json(orient="records", date_format="iso"))
                    proof = [] if evidence.empty else json.loads(evidence.loc[evidence.accession_number.isin(result.accession_number)].to_json(orient="records"))
                else:
                    records, proof = [], []
                record = {"status": "complete", "rows": records, "coverage_evidence": proof,
                    "exclusions": skipped, "sources": provenance, "issues": diagnostics}
                write_json(marker, record)
            if record["status"] == "unavailable":
                exclusions.append({"cik": cik, "reason": record["reason"]})
            else:
                rows.extend(record["rows"])
                exclusions.extend(record["exclusions"])
                sources.extend(record["sources"])
                issues.extend({"cik": cik, **issue} for issue in record["issues"])
        except Exception as exc:
            permanent = isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 404
            permanent |= isinstance(exc, ValueError) and str(exc).startswith("No original 10-K/10-Q USD Assets facts")
            item = {"cik": cik, "reason": str(exc)}
            if permanent:
                exclusions.append(item)
                write_json(marker, {"status": "unavailable", **item})
            else:
                errors.append(item)
        if index % 10 == 0:
            write_json(ROOT / "progress.json", {"processed": index, "requested": len(plan["companies"]),
                "rows": len(rows), "errors": len(errors), "exclusions": len(exclusions)})
    write_json(ROOT / "acquisition_report.json", {"errors": errors, "exclusions": exclusions, "sources": sources, "issues": issues})
    if errors or not rows:
        raise ValueError("V3 acquisition incomplete; resolve errors and resume the same fixed plan")
    dataset = pd.DataFrame(rows)
    for column in ("filing_date", "prediction_date", "feature_filing_date", "latest_input_filing_date", "label_available_date_1yr"):
        dataset[column] = pd.to_datetime(dataset[column]).dt.strftime("%Y-%m-%d")
    dataset[TARGET] = pd.to_numeric(dataset.target_distressed_1yr, errors="raise").astype("Int64")
    dataset["eligible"] = dataset.historical_coverage_eligible.eq(True) & dataset[TARGET].notna()
    dataset["label_status"] = np.where(dataset.eligible, "source_reviewed", "unknown_or_outside_scope")
    # Known-but-ineligible outcomes remain in the audit columns, never a training label.
    dataset.loc[~dataset.eligible, TARGET] = pd.NA
    dataset["filing_date"] = dataset.feature_filing_date
    dataset["accession_number"] = dataset.source_accession_number
    dataset.to_csv(ROOT / "dataset.csv", index=False)
    proof = []
    for marker in checkpoints.glob("*.json"):
        proof.extend(json.loads(marker.read_text()).get("coverage_evidence", []))
    write_json(ROOT / "coverage_evidence.json", proof)
    manifest = {"version": 3, "plan_sha256": digest(PLAN), "config_sha256": digest(CONFIG),
        "dataset_sha256": digest(ROOT / "dataset.csv"), "frame_companies": plan["frame_companies"],
        "sampled_companies": len(plan["companies"]), "snapshot_rows": len(dataset),
        "eligible_snapshot_rows": int(dataset.eligible.sum()), "unknown_rows": int((~dataset.eligible).sum()),
        "schema": {key: str(value) for key, value in dataset.dtypes.items()},
        "target": config["target"], "scope": config["scope"], "production_ready": False,
        "final_labels_inspected_for_selection": False, "note": "Design weights do not certify representativeness or correct source nonresponse."}
    write_json(sealed_path, manifest)
    return manifest
