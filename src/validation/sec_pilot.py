"""As-filed annual SEC feature extraction with cell-level provenance and label audit.

This small case-control sample tests extraction. Negative labels remain provisional
until a complete event-coverage registry is independently reviewed. Such rows are
explicitly excluded from production training eligibility.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.features_distress import compute_altman_components
from src.validation.download import companyfacts

FEATURES = ["wc_to_ta", "re_to_ta", "ebit_to_ta", "sales_to_ta"]
TAGS = {
    "Assets": ["Assets"], "AssetsCurrent": ["AssetsCurrent"],
    "LiabilitiesCurrent": ["LiabilitiesCurrent"], "Liabilities": ["Liabilities"],
    "RetainedEarningsAccumulatedDeficit": ["RetainedEarningsAccumulatedDeficit"],
    "OperatingIncomeLoss": ["OperatingIncomeLoss"],
    "Revenues": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "Revenues", "SalesRevenueNet"]}
DURATION_TAGS = {"OperatingIncomeLoss", "Revenues"}


def extract_annual_rows(payload, ticker):
    """Use only same-accession facts and current fiscal ends, never later restatements."""
    cik = str(payload["cik"]).zfill(10)
    gaap = payload.get("facts", {}).get("us-gaap", {})
    assets = [fact for fact in gaap.get("Assets", {}).get("units", {}).get("USD", [])
              if fact.get("form") == "10-K"]
    accessions = sorted({fact["accn"] for fact in assets})
    rows, provenance, issues = [], [], []
    for accession in accessions:
        candidates = [fact for fact in assets if fact["accn"] == accession]
        end = max(fact["end"] for fact in candidates)
        dates = {fact["filed"] for fact in candidates}
        if len(dates) != 1:
            issues.append({"cik": cik, "accession": accession, "reason": "Conflicting filing dates"})
            continue
        filed = dates.pop()
        if end > filed:
            raise ValueError("Fiscal period ends after public filing date")
        row = {"cik": cik, "ticker": ticker, "accession_number": accession,
               "filing_date": filed, "fiscal_period_end": end, "filing_type": "10-K"}
        for canonical, tags in TAGS.items():
            selected = None
            for tag in tags:
                facts = [fact for fact in gaap.get(tag, {}).get("units", {}).get("USD", [])
                         if fact.get("accn") == accession and fact.get("end") == end
                         and fact.get("filed") == filed and fact.get("form") == "10-K"]
                if canonical in DURATION_TAGS:
                    facts = [fact for fact in facts if fact.get("start") and
                             300 <= (pd.Timestamp(end) - pd.Timestamp(fact["start"])).days <= 380]
                else:
                    facts = [fact for fact in facts if not fact.get("start")]
                values = {fact["val"] for fact in facts}
                if len(values) > 1:
                    issues.append({"cik": cik, "accession": accession, "tag": tag,
                                   "reason": "Conflicting facts; value withheld"})
                    break
                if len(values) == 1:
                    selected = facts[0]
                    provenance.append({"cik": cik, "accession_number": accession, "field": canonical,
                                       "source_tag": tag, "unit": "USD", **selected})
                    break
            row[canonical] = selected["val"] if selected else np.nan
        row.update(compute_altman_components(row))
        rows.append(row)
    return pd.DataFrame(rows), provenance, issues


def label_rows(frame, company, observation_end):
    """Calendar-year horizon; no post-event rows or immature negative outcomes."""
    frame = frame.copy()
    dates = pd.to_datetime(frame["filing_date"])
    ends = dates + pd.DateOffset(years=1)
    event = pd.Timestamp(company["event_date"]) if company.get("event_date") else None
    observed = pd.Timestamp(observation_end)
    positive = (dates < event) & (event <= ends) if event is not None else pd.Series(False, index=frame.index)
    valid = (dates < event) if event is not None else pd.Series(True, index=frame.index)
    mature = positive | (ends <= observed)
    frame["target_distressed_1yr"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[valid & mature, "target_distressed_1yr"] = positive[valid & mature].astype(int)
    frame["label_available_date"] = ends
    if event is not None:
        frame.loc[positive, "label_available_date"] = event
    frame["event_source"] = company.get("event_source")
    verified_positive = positive & bool(company.get("event_source"))
    frame["label_review_status"] = np.where(verified_positive, "positive_event_cited", "coverage_review_required")
    frame["valid_for_training"] = valid & mature & verified_positive
    frame["exclusion_reason"] = np.where(~valid, "filing_on_or_after_bankruptcy",
        np.where(~mature, "incomplete_prediction_horizon", ""))
    return frame


def prepare_pilot(config="config/validation_pilot.json", output="data/processed/validation/sec_pilot"):
    config = json.loads(Path(config).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames, provenance, sources, issues = [], [], [], []
    for company in config["companies"]:
        payload, source = companyfacts(company["cik"])
        sources.append(source)
        frame, cells, conflicts = extract_annual_rows(payload, company["ticker"])
        provenance.extend(cells)
        issues.extend(conflicts)
        frames.append(label_rows(frame, company, config["observation_end"]))
    all_rows = pd.concat(frames, ignore_index=True)
    usable = all_rows[all_rows.exclusion_reason == ""].copy()
    # Provisional labels are available for inspection, but production eligibility remains false.
    all_rows.to_csv(output / "all_rows_with_exclusions.csv", index=False)
    usable.to_csv(output / "features.csv", index=False)
    pd.DataFrame(provenance).to_csv(output / "fact_provenance.csv", index=False)
    report = {"purpose": config["description"], "sources": sources,
              "raw_filing_rows": len(all_rows), "uncensored_pre_event_rows": len(usable),
              "company_count": usable.cik.nunique(), "positive_rows": int(usable.target_distressed_1yr.sum()),
              "positive_companies": usable.loc[usable.target_distressed_1yr == 1, "cik"].nunique(),
              "excluded_rows": all_rows.exclusion_reason.value_counts().to_dict(),
              "missing_feature_fraction": usable[FEATURES].isna().mean().to_dict(),
              "fact_conflicts": issues, "features": FEATURES,
              "production_ready": False,
              "limitations": ["Negative outcomes require independently reviewed complete bankruptcy-event coverage; absence from the event list is not proof of no bankruptcy.",
                              "Purposive ten-company pilot; no representative point-in-time index universe or production performance claim.",
                              "Annual filings only. Operating income is an EBIT proxy, not an exact standardized EBIT definition.",
                              "Historical market capitalization was not sourced; mve_to_tl is excluded, not imputed into a fake full Altman score."]}
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: report[key] for key in ("raw_filing_rows", "uncensored_pre_event_rows", "positive_rows", "positive_companies", "production_ready")}, indent=2))
    return usable, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/validation_pilot.json")
    parser.add_argument("--output", default="data/processed/validation/sec_pilot")
    args = parser.parse_args()
    prepare_pilot(args.config, args.output)
