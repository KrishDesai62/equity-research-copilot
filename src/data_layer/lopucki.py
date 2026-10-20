"""Import the free Florida-UCLA-LoPucki BRD with its actual coverage limits.

The published January 2023 release explicitly ends on 2022-12-31. This is an
event registry, not a table of healthy companies. CIK/name suggestions are never
automatically accepted, and a missing case never becomes a negative label.
"""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import zipfile

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.download import fetch

URL = "https://lopucki.law.ufl.edu/download_cases_table.php"
PROTOCOLS = "https://lopucki.law.ufl.edu/documentation/BRD_protocols.pdf"
COVERAGE_END = "2022-12-31"


def normalize_cik(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    raw = str(value).strip()
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw[:-2]
    if not re.fullmatch(r"\d{1,10}", raw) or int(raw) == 0:
        return None
    return raw.zfill(10)


def parse_archive(archive):
    """Read one versioned CSV inside the ZIP; no arbitrary archive extraction."""
    with zipfile.ZipFile(archive) as data:
        names = [n for n in data.namelist() if n.endswith(".csv")]
        if len(names) != 1 or "1-12-2023" not in names[0]:
            raise ValueError("Unrecognized BRD release; review its coverage before importing")
        member = data.getinfo(names[0])
        if member.file_size > 50_000_000:
            raise ValueError("BRD CSV exceeds expected size")
        payload = data.read(names[0])
    frame = pd.read_csv(io.BytesIO(payload), encoding="cp1252", dtype=str)
    required = {"NameCorp", "DateFiled", "CikBefore", "PrimaryKey", "Chapter"}
    if not required <= set(frame):
        raise ValueError(f"Missing BRD fields: {sorted(required - set(frame))}")
    if frame.PrimaryKey.isna().any() or frame.PrimaryKey.duplicated().any():
        raise ValueError("BRD primary keys must be unique and present")
    dates = pd.to_datetime(frame.DateFiled, format="%m/%d/%Y", errors="raise")
    if dates.isna().any() or dates.gt(pd.Timestamp(COVERAGE_END)).any():
        raise ValueError("Case dates exceed the documented snapshot or are missing")
    frame["cik"] = frame.CikBefore.map(normalize_cik)
    frame["filing_date"] = dates.dt.strftime("%Y-%m-%d")
    return frame, {"member": names[0], "member_sha256": sha256(payload).hexdigest()}


def reconcile_events(companies, path="data/processed/lopucki/chapter11_events.csv"):
    """Join exact CIKs; conflicting cited dates quarantine targets, never overwrite."""
    events = pd.read_csv(path, dtype={"cik": str})
    required = {"cik", "filing_date", "source_url", "chapter", "coverage_end"}
    if not required <= set(events):
        raise ValueError("Imported BRD event schema is incomplete")
    if not events.chapter.astype(str).eq("11").all():
        raise ValueError("Only Chapter 11 events may enter the distress target")
    dates = pd.to_datetime(events.filing_date, errors="raise")
    if dates.isna().any() or dates.gt(pd.Timestamp(COVERAGE_END)).any():
        raise ValueError("BRD events exceed documented coverage")
    result, audit = [], []
    decision_path = Path("data/curated/bankruptcy_event_adjudications.csv")
    decisions = pd.read_csv(decision_path, dtype={"cik": str}).set_index("cik") if decision_path.exists() else None
    for company in companies:
        company = dict(company)
        cik = str(company["cik"]).zfill(10)
        matched = events.loc[events.cik.eq(cik)].sort_values("filing_date")
        if len(matched):
            earliest = matched.iloc[0]
            conflict = bool(company.get("event_date") and company["event_date"] != earliest.filing_date)
            adjudicated = False
            if conflict and decisions is not None and cik in decisions.index:
                decision = decisions.loc[cik]
                evidence = Path(decision.source_path)
                if decision.issuer_reported_date != company["event_date"] or decision.chosen_event_date != earliest.filing_date:
                    raise ValueError("Adjudication does not reconcile the actual source dates")
                if sha256(evidence.read_bytes()).hexdigest() != decision.source_sha256:
                    raise ValueError("Adjudication source checksum differs")
                company["issuer_reported_event_date"] = company["event_date"]
                company["event_date"] = decision.chosen_event_date
                company["event_source"] = decision.source_url
                company["event_source_path"] = str(evidence)
                company["event_date_policy"] = decision.date_policy
                conflict, adjudicated = False, True
            company["bankruptcy_label_conflict"] = conflict
            if not company.get("event_date"):
                company["event_date"] = earliest.filing_date
                company["event_source"] = earliest.source_url
            audit.append({"cik": cik, "ticker": company["ticker"], "configured_date": company.get("event_date"),
                "brd_date": earliest.filing_date, "source_url": earliest.source_url,
                "status": "adjudicated_court_filing_date" if adjudicated else "date_conflict_requires_review" if conflict else "exact_cik_date_agreement"})
        result.append(company)
    return result, audit


def import_brd(archive="data/raw/lopucki/cases.download", output="data/processed/lopucki",
               config="config/research_universe.json", download=False):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if download:
        fetch(URL, archive)
        fetch(PROTOCOLS, archive.parent / "protocols.pdf")
    frame, member = parse_archive(archive)
    companies = json.loads(Path(config).read_text())["companies"]
    mapping = {str(c["cik"]).zfill(10): c for c in companies}
    # Chapter 7 is retained in the raw table but is not silently relabeled as
    # Chapter 11. Records without an exact CIK remain identity-review candidates.
    selected = frame.loc[frame.Chapter.eq("11") & frame.cik.notna()].copy()
    events = pd.DataFrame({"company_name": selected.NameCorp, "cik": selected.cik,
        "filing_date": selected.filing_date, "chapter": 11, "brd_case_id": selected.PrimaryKey,
        "source_url": URL, "coverage_end": COVERAGE_END,
        "identity_match_method": "exact_CikBefore", "source": "Florida-UCLA-LoPucki Bankruptcy Research Database"})
    events["ticker"] = events.cik.map({cik: c["ticker"] for cik, c in mapping.items()})
    events = events.sort_values(["cik", "filing_date", "brd_case_id"])
    events.to_csv(output / "chapter11_events.csv", index=False)
    frame.to_csv(output / "cases.csv", index=False)
    matches = events.loc[events.cik.isin(mapping)].copy()
    matches["configured_event_date"] = matches.cik.map({cik: c.get("event_date") for cik, c in mapping.items()})
    matches["date_agrees"] = matches.filing_date.eq(matches.configured_event_date)
    matches.to_csv(output / "research_matches.csv", index=False)
    frame.loc[frame.cik.isna(), ["PrimaryKey", "NameCorp", "DateFiled", "Chapter"]].to_csv(
        output / "identity_review.csv", index=False)
    coverage = [{"cik": cik, "ticker": c["ticker"], "event_coverage_reviewed": False,
        "coverage_start": "", "observed_through": "", "reviewer": "", "coverage_source_url": "",
        "brd_snapshot_end": COVERAGE_END,
        "review_note": "Verify large/public/group inclusion and identity completeness; no negative label inferred"}
        for cik, c in mapping.items()]
    # Preserve reviewer work on subsequent imports.
    if not (output / "coverage_review.csv").exists():
        pd.DataFrame(coverage).to_csv(output / "coverage_review.csv", index=False)
    report = {"source_url": URL, "archive_sha256": sha256(archive.read_bytes()).hexdigest(), **member,
        "snapshot_date": "2023-01-12", "coverage_end": COVERAGE_END,
        "total_cases": len(frame), "chapter11_events_with_exact_cik": len(events),
        "missing_or_invalid_cik_cases": int(frame.cik.isna().sum()),
        "research_matched_cases": len(matches), "research_matched_companies": matches.cik.nunique(),
        "configured_date_disagreements": int((matches.configured_event_date.notna() & ~matches.date_agrees).sum()),
        "negative_labels_created": 0, "production_ready": False,
        "coverage_definition": "Chapter 7/11 US debtor groups, >=$100m assets in 1980 dollars at last pre-filing 10-K; qualifying 10-K within three years, with group/subsidiary rules.",
        "limitations": ["BRD protocols explicitly say coverage ends December 31, 2022 and no further updates are intended.",
            "Absence is not evidence of no bankruptcy for an arbitrary SEC registrant.",
            "CIKBefore is missing for some cases; unresolved identities require review.",
            "Refilings and group identity rules require adjudication; earliest cited event controls pre-event modeling."]}
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/raw/lopucki/cases.download")
    parser.add_argument("--output", default="data/processed/lopucki")
    parser.add_argument("--config", default="config/research_universe.json")
    parser.add_argument("--download", action="store_true")
    import_brd(**vars(parser.parse_args()))
