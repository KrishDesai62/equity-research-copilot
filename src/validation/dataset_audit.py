"""Validate serving feature provenance without converting unknown outcomes to zero."""
import pandas as pd


def audit_features(frame, as_of):
    issues = []
    required = {"cik", "accession_number", "filing_date", "latest_input_filing_date", "source_url"}
    if frame.empty:
        issues.append("Research feature dataset is empty")
    if not required <= set(frame):
        return issues + ["Required feature provenance columns missing"]
    for column in required:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            issues.append(f"Missing feature provenance: {column}")
    if not frame.cik.astype(str).str.fullmatch(r"\d{10}").all():
        issues.append("Company identifiers must be zero-padded 10-digit CIKs")
    if not frame.source_url.astype(str).str.match(r"https?://[^\s/]+/\S+$").all():
        issues.append("Invalid feature source URLs")
    if frame.duplicated(["cik", "accession_number"]).any():
        issues.append("Duplicate company/accession feature rows")
    dates = pd.to_datetime(frame.filing_date, errors="coerce")
    inputs = pd.to_datetime(frame.latest_input_filing_date, errors="coerce")
    if dates.isna().any() or inputs.isna().any():
        issues.append("Invalid or missing public feature dates")
    if inputs.gt(dates).any():
        issues.append("Future inputs in financial features")
    if dates.gt(pd.Timestamp(as_of)).any():
        issues.append("Future feature filing dates")
    return issues
