"""Refresh a dated local SEC snapshot before considering live AWS deployment."""
import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from src.data_layer.filing_changes import normalize_filing
from src.data_layer.research_dataset import build, write_json
from src.data_layer.xbrl_bulk import download_quarterly_zip, parse_zip_to_dataframe, build_feature_table
from src.validation.model_readiness import run as readiness
from src.validation.research_checks import run as research_checks
from src.data_layer.snapshots import publish_snapshot


def last_completed_quarter(as_of):
    today = date.fromisoformat(as_of)
    quarter = (today.month - 1) // 3 + 1
    return (today.year - 1, 4) if quarter == 1 else (today.year, quarter - 1)


def preserve_snapshot(root):
    """Keep old research inputs and human review work before refreshing."""
    summary = root / "summary.json"
    if not summary.exists():
        return None
    payload = summary.read_bytes()
    previous = json.loads(payload)
    if not previous.get("complete", True):
        return None
    label = previous["observation_end"]
    destination = root.parent / "research_snapshots" / f"{label}_{sha256(payload).hexdigest()[:12]}"
    if not destination.exists():
        shutil.copytree(root, destination)
    return str(destination)


def add_bulk_dataset(config, raw, output, as_of):
    year, quarter = last_completed_quarter(as_of)
    # A closed historical quarter must be publicly available; a missing archive
    # raises instead of silently substituting an older period.
    archive = download_quarterly_zip(year, quarter, raw / "bulk", cache=False)
    frame = parse_zip_to_dataframe(archive, ciks=[c["cik"] for c in config["companies"]])
    frame = frame.loc[frame.form.isin(["10-K", "10-Q"]) & frame.filing_date.le(pd.Timestamp(as_of))].copy()
    if frame.empty:
        raise ValueError("New SEC bulk dataset has no matching original filings")
    keys = ["cik", "adsh", "tag"]
    conflicts = frame.groupby(keys).value.transform("nunique").gt(1)
    frame.loc[conflicts].to_csv(output / "bulk_conflicts.csv", index=False)
    frame = frame.loc[~conflicts].drop_duplicates(keys)
    frame.to_csv(output / "bulk_facts.csv", index=False)
    build_feature_table([frame]).to_csv(output / "bulk_financial_table.csv", index=False)
    report = {"dataset": "SEC Financial Statement Data Sets", "year": year, "quarter": quarter,
        "source_url": f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{quarter}.zip",
        "sha256": sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size,
        "matching_companies": frame.cik.nunique(), "matching_filings": frame.adsh.nunique(),
        "consolidated_usd_facts": len(frame), "conflicting_facts_withheld": int(conflicts.sum()),
        "latest_public_filing_date": frame.filing_date.max().date().isoformat(),
        "bankruptcy_labels_supplied": False,
        "note": "Additional financial source; not a labeled outcome dataset or independent performance test."}
    write_json(output / "bulk_dataset.json", report)
    return report


def run(config_path="config/research_recent.json"):
    config = json.loads(Path(config_path).read_text())
    as_of = config["observation_end"]
    raw = Path("data/raw/research_updates") / as_of
    current = Path("data/processed/research")
    snapshot = preserve_snapshot(current)
    output = Path("data/processed/research_versions") / f"{as_of}_{uuid4().hex}"
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / "research_config.json")
    for name in ("qa_review.jsonl", "outcome_coverage_review.csv"):
        if (current / name).exists():
            shutil.copyfile(current / name, output / name)
    # Dated caches force the first run to fetch current metadata/facts without
    # replacing raw inputs used in previous historical experiments.
    dataset = build(config_path=config_path, output=str(output), raw=str(raw), facts_directory=str(raw / "companyfacts"))
    if not dataset["complete"]:
        raise ValueError("Refresh is partial; resolve source errors before continuing validation")
    bulk = add_bulk_dataset(config, raw, output, as_of)
    checks = research_checks(research=str(output))
    # Data refresh is not permission to retune models against an exposed test.
    model_report = readiness(research=str(output), config=config_path, output=str(output / "model_validation"), evaluate=False)
    event_checks = []
    for company in config["companies"]:
        if not company.get("event_date") or company["event_date"] < "2024-01-01":
            continue
        path = Path("data/raw/validation/events") / (company["ticker"] + ".html")
        text = normalize_filing(path.read_text())
        expected = pd.Timestamp(company["event_date"]).strftime("%B %d, %Y").replace(" 0", " ")
        event_checks.append({"ticker": company["ticker"], "source_url": company["event_source"],
            "event_date": company["event_date"], "event_date_in_source": expected in text,
            "chapter_11_in_source": "chapter 11" in text.lower(), "sha256": sha256(path.read_bytes()).hexdigest()})
    if not all(e["event_date_in_source"] and e["chapter_11_in_source"] for e in event_checks):
        raise ValueError("A recent bankruptcy source did not corroborate the configured event date")
    report = {"as_of": as_of, "previous_research_snapshot": snapshot, "research": dataset,
        "additional_dataset": bulk, "recent_bankruptcy_source_checks": event_checks,
        "checks": checks, "models": model_report, "aws_deployed": False,
        "aws_next_step": "Configure and test live AWS after reviewed labels and independent model validation.",
        "production_ready": False}
    write_json(output / "recent_update.json", report)
    report["publication"] = publish_snapshot(output, current)
    write_json(output / "recent_update.json", report)
    print(json.dumps({"as_of": as_of, "companies": dataset["companies"], "financial_rows": dataset["financial_rows"],
        "additional_dataset": f"{bulk['year']} Q{bulk['quarter']}", "model_status": model_report["status"],
        "accuracy": model_report["accuracy"], "aws_deployed": False}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="config/research_recent.json")
    run(**vars(parser.parse_args()))
