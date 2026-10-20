"""Verify the V3 dataset against source checkpoints without scoring a model."""
from collections import Counter
import json

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import ROOT, RAW, CONFIG, PLAN, TARGET, digest


def audit(*, verify_raw=True):
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    plan = json.loads(PLAN.read_text())
    if (digest(CONFIG) != manifest["config_sha256"] or digest(PLAN) != manifest["plan_sha256"] or
            digest(ROOT / "dataset.csv") != manifest["dataset_sha256"]):
        raise ValueError("Dataset seal changed")
    acquisition = json.loads((ROOT / "acquisition_report.json").read_text())
    if acquisition["errors"]:
        raise ValueError("Unresolved source acquisition errors")
    frame = pd.read_csv(ROOT / "dataset.csv", dtype={"cik": str})
    eligible = frame.eligible.astype(str).str.lower().eq("true")
    if frame.unit_id.duplicated().any() or frame.loc[~eligible, TARGET].notna().any():
        raise ValueError("Duplicate snapshot or unknown outcome used as a label")
    if frame.loc[eligible, TARGET].isna().any():
        raise ValueError("Eligible snapshot missing outcome")
    proof = json.loads((ROOT / "coverage_evidence.json").read_text())
    proof_keys = {(r["cik"], r["accession_number"]) for r in proof}
    needed = {(r.cik, f"{r.source_accession_number}@{r.prediction_date}") for r in frame.loc[eligible].itertuples()}
    if not needed <= proof_keys:
        raise ValueError("Eligible labels missing coverage evidence")
    source_records, checkpoint_hashes, statuses = {}, {}, Counter()
    planned = {c["cik"]: c for c in plan["companies"]}
    for cik, company in planned.items():
        path = ROOT / "company_checkpoints" / f"{cik}.json"
        record = json.loads(path.read_text())
        if record["status"] not in {"complete", "unavailable"}:
            raise ValueError("Unfinished company checkpoint")
        statuses[record["status"]] += 1
        checkpoint_hashes[cik] = digest(path)
        for source in record.get("sources", []):
            path = source["path"]
            if path in source_records and source_records[path] != source["sha256"]:
                raise ValueError("Conflicting raw source hashes")
            source_records[path] = source["sha256"]
        for row in record.get("rows", []):
            if (row["role"] != company["role"] or row["design_weight"] != company["design_weight"] or
                    row["inclusion_probability"] != company["inclusion_probability"]):
                raise ValueError("Checkpoint differs from committed sampling design")
    for row in frame.itertuples():
        company = planned[row.cik]
        if row.role != company["role"] or row.design_weight != company["design_weight"]:
            raise ValueError("Dataset differs from committed sampling design")
    if verify_raw:
        for path, expected in source_records.items():
            if digest(path) != expected:
                raise ValueError(f"Raw source changed: {path}")
        for source in plan["universe_archives"]:
            if digest(RAW / "bulk" / f"{source['year']}q{source['quarter']}.zip") != source["sha256"]:
                raise ValueError("Universe archive changed")
    # Deliberately report counts without final outcome breakdown before selection.
    attrition = []
    for role in ("training", "calibration", "final"):
        for stratum in ("registry_case_company", "background"):
            requested = [c for c in plan["companies"] if c["role"] == role and c["sampling_stratum"] == stratum]
            selected = frame.loc[frame.role.eq(role) & frame.sampling_stratum.eq(stratum)]
            kept = selected.loc[selected.eligible.astype(str).str.lower().eq("true")]
            weights = kept.design_weight.to_numpy(float)
            attrition.append({"role": role, "sampling_stratum": stratum, "requested_companies": len(requested),
                "companies_with_snapshots": int(selected.cik.nunique()), "eligible_companies": int(kept.cik.nunique()),
                "snapshot_rows": len(selected), "eligible_rows": len(kept),
                "row_weight_effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) else 0.})
    seal = {"dataset_sha256": digest(ROOT / "dataset.csv"), "checkpoints": checkpoint_hashes,
            "raw_sources": source_records, "coverage_evidence_sha256": digest(ROOT / "coverage_evidence.json"),
            "acquisition_report_sha256": digest(ROOT / "acquisition_report.json")}
    seal_path = ROOT / "source_seal.json"
    if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
        raise ValueError("Frozen source evidence changed")
    if not seal_path.exists():
        write_json(seal_path, seal)
    report = {"dataset_sha256": manifest["dataset_sha256"], "source_seal_sha256": digest(seal_path),
              "source_integrity_verified": bool(verify_raw), "raw_sources": len(source_records),
              "checkpoints": dict(statuses), "attrition_by_design_stratum": attrition,
              "exclusion_reasons": dict(Counter(r["reason"] for r in acquisition["exclusions"])),
              "limits": ["Eligibility uses full follow-up and defines a restricted historical population; attrition may bias results.",
                         "Source review is automated registry/filing reconciliation, not an individual court search for every negative.",
                         "Effective sample sizes describe unequal row weights; confidence intervals must also cluster companies."],
              "production_ready": False}
    write_json(ROOT / "data_quality_report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
