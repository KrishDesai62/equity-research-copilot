"""Build a cited, connected research corpus; never train on unreviewed outcomes.

Run with ``python -m src.data_layer.research_dataset``. Raw downloads are cached;
the output is a research snapshot, not a representative investment universe.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

import pandas as pd

from src.data_layer.bankruptcy_labels import build_distress_labels
from src.data_layer.filing_changes import Filing, compare_filings, extract_sections
from src.ml.financial_history import extract_history
from src.rag.chunking import chunk_section
from src.validation.download import companyfacts, fetch

FORMS = {"10-K", "10-Q", "8-K", "4"}
FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def merge_review_rows(path, rows, key):
    """Append new review tasks without overwriting human answers or approvals."""
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    known = {row[key] for row in existing}
    write_jsonl(path, existing + [row for row in rows if row[key] not in known])


def submission_rows(block, cik, ticker, start, end):
    """Normalize SEC column arrays; reject malformed metadata, filter public dates."""
    n = len(block.get("accessionNumber", []))
    required = ("accessionNumber", "filingDate", "form", "primaryDocument")
    if any(len(block.get(key, [])) != n for key in required):
        raise ValueError("Inconsistent SEC submissions arrays")
    rows = []
    for i in range(n):
        form, filed = block["form"][i], block["filingDate"][i]
        if form not in FORMS or not start <= filed <= end:
            continue
        accession, document = block["accessionNumber"][i], block["primaryDocument"][i]
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("Invalid accession")
        if not document or ".." in document or document.startswith("/") or "\\" in document:
            raise ValueError("Invalid SEC document path")
        # EDGAR often points Form 4 at an XSL-rendered HTML view. Download the
        # underlying XML instead; the rendered view cannot be parsed as XML.
        if form == "4" and document.startswith("xsl"):
            document = document.split("/")[-1]
        items = block.get("items", [""] * n)
        rows.append({"cik": cik, "ticker": ticker, "accession_number": accession,
            "filing_date": filed, "filing_type": form, "primary_document": document,
            "items": items[i] if len(items) == n else "",
            "source_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{quote(document, safe='/')}",
            "acceptance_datetime": block.get("acceptanceDateTime", [""] * n)[i]
                if len(block.get("acceptanceDateTime", [])) == n else ""})
    return rows


def load_submissions(company, start, end, raw, sources):
    cik, ticker = company["cik"], company["ticker"]
    path = raw / "submissions" / f"CIK{cik}.json"
    sources.append(fetch(f"https://data.sec.gov/submissions/CIK{cik}.json", path))
    payload = json.loads(path.read_text())
    if str(payload["cik"]).zfill(10) != cik:
        raise ValueError("Submissions CIK mismatch")
    blocks = [payload.get("filings", {}).get("recent", {})]
    for archived in payload.get("filings", {}).get("files", []):
        if archived["filingFrom"] > end or archived["filingTo"] < start:
            continue
        name = archived["name"]
        if Path(name).name != name:
            raise ValueError("Invalid submissions archive name")
        path = raw / "submissions" / name
        sources.append(fetch(f"https://data.sec.gov/submissions/{name}", path))
        blocks.append(json.loads(path.read_text()))
    rows = {}
    for block in blocks:
        for row in submission_rows(block, cik, ticker, start, end):
            key = row["accession_number"]
            if key in rows and rows[key] != row:
                raise ValueError(f"Conflicting submission metadata: {key}")
            rows[key] = row
    return list(rows.values()), payload.get("name", "")


def accounting_candidates(filings):
    """Item 4.02 identifies non-reliance disclosures, not proven fraud."""
    result = []
    for row in filings:
        if row["filing_type"] != "8-K":
            continue
        items = set(re.findall(r"\d+\.\d{2}", row["items"]))
        for item, kind in (("4.02", "non_reliance"), ("1.03", "bankruptcy_or_receivership")):
            if item in items:
                result.append({**row, "event_kind": kind, "item": item,
                               "review_status": "candidate_requires_review"})
    return result


def parse_insider_xml(text, filing):
    """Keep transaction codes distinct: awards/exercises are not purchases."""
    root = ET.fromstring(text)
    for element in root.iter():
        element.tag = element.tag.split("}")[-1]
    issuer = root.findtext(".//issuer/issuerCik")
    if issuer is None or issuer.zfill(10) != filing["cik"]:
        raise ValueError("Form 4 issuer does not match company")
    result = []
    for ordinal, tx in enumerate(root.findall(".//nonDerivativeTransaction")):
        def get(path):
            value = tx.findtext(path)
            return value.strip() if value else None
        def number(path):
            value = get(path)
            return float(value) if value else None
        code = get("transactionCoding/transactionCode")
        result.append({**filing, "transaction_index": ordinal,
            "transaction_date": get("transactionDate/value"), "transaction_code": code,
            "shares": number("transactionAmounts/transactionShares/value"),
            "price_per_share": number("transactionAmounts/transactionPricePerShare/value"),
            "acquired_disposed": get("transactionAmounts/transactionAcquiredDisposedCode/value"),
            "purchase_sale_signal": "purchase" if code == "P" else "sale" if code == "S" else "other",
            "available_date": filing["filing_date"]})
    return result


def financial_labels(frame, companies, observation_end):
    """Preserve positive cited outcomes; never turn missing event lists into zeros."""
    events = pd.DataFrame([{"cik": c["cik"], "filing_date": c["event_date"],
                            "source_url": c["event_source"]}
        for c in companies if c.get("event_date") and c.get("event_source")],
        columns=["cik", "filing_date", "source_url"])
    frame = frame.copy()
    frame["removed_date"] = pd.NaT
    frame["event_coverage_reviewed"] = False
    labeled = build_distress_labels(frame, events, drop_invalid_rows=False, observation_end=observation_end)
    # These are separate tasks. No reviewed AAER/restatement coverage exists yet.
    labeled["target_aaer_enforcement_flag"] = pd.Series(pd.NA, index=labeled.index, dtype="Int64")
    labeled["target_restatement_flag"] = pd.Series(pd.NA, index=labeled.index, dtype="Int64")
    labeled["aaer_label_available_date"] = pd.NaT
    labeled["aaer_source_url"] = None
    labeled["restatement_label_available_date"] = pd.NaT
    labeled["restatement_source_url"] = None
    conflicted = {str(c["cik"]).zfill(10) for c in companies if c.get("bankruptcy_label_conflict")}
    labeled["bankruptcy_label_conflict"] = labeled.cik.isin(conflicted)
    for years in (1, 3):
        labeled.loc[labeled.bankruptcy_label_conflict, f"target_distressed_{years}yr"] = pd.NA
        labeled.loc[labeled.bankruptcy_label_conflict, f"label_available_date_{years}yr"] = pd.NaT
    labeled.loc[labeled.bankruptcy_label_conflict, "valid_for_training"] = False
    dates = pd.to_datetime(labeled.filing_date)
    for company in companies:
        for event in sorted(company.get("aaer_events", []), key=lambda e: e["release_date"], reverse=True):
            if not event.get("source_url") or not event.get("aaer_number"):
                raise ValueError("AAER events require a cited release number")
            released = pd.Timestamp(event["release_date"])
            positive = (labeled.cik.eq(company["cik"]) & dates.lt(released) &
                (dates + pd.DateOffset(years=3)).ge(released) & (released <= pd.Timestamp(observation_end)))
            labeled.loc[positive, "target_aaer_enforcement_flag"] = 1
            labeled.loc[positive, "aaer_label_available_date"] = released
            labeled.loc[positive, "aaer_source_url"] = event["source_url"]
        for event in sorted(company.get("non_reliance_events", []), key=lambda e: e["filing_date"], reverse=True):
            disclosed = pd.Timestamp(event["filing_date"])
            positive = (labeled.cik.eq(company["cik"]) & dates.lt(disclosed) &
                (dates + pd.DateOffset(years=3)).ge(disclosed) & (disclosed <= pd.Timestamp(observation_end)))
            labeled.loc[positive, "target_restatement_flag"] = 1
            labeled.loc[positive, "restatement_label_available_date"] = disclosed
            labeled.loc[positive, "restatement_source_url"] = event["source_url"]
    return labeled


def export_finqa(raw, output, sources):
    """Development benchmark only; retain original tables/evidence and attribution."""
    base = f"https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_COMMIT}"
    for name in ("README.md", "LICENSE"):
        sources.append(fetch(f"{base}/{name}", raw / "finqa" / name))
    path = raw / "finqa" / "dev.json"
    sources.append(fetch(f"{base}/dataset/dev.json", path))
    records = json.loads(path.read_text())
    write_jsonl(output / "finqa_development.jsonl", [{"id": r["id"], "split": "development",
        "question": r["qa"]["question"], "expected_answer": r["qa"]["answer"],
        "evidence": r["qa"].get("gold_inds", {}), "program": r["qa"].get("program"),
        "table": r["table"], "pre_text": r["pre_text"], "post_text": r["post_text"],
        "source_url": f"{base}/dataset/dev.json", "sec_cik_verified": False} for r in records])
    return len(records)


def build(config_path="config/research_universe.json", output="data/processed/research",
          raw="data/raw/research", start="2010-01-01", annual_documents=2,
          quarterly_documents=1, insider_documents=1, finqa=True,
          bankruptcy_events="data/processed/lopucki/chapter11_events.csv",
          facts_directory="data/raw/validation"):
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    companies = config["companies"]
    event_reconciliation = []
    if Path(bankruptcy_events).exists():
        from src.data_layer.lopucki import reconcile_events
        companies, event_reconciliation = reconcile_events(companies, bankruptcy_events)
    accounting_registry = Path("data/curated/accounting_review_decisions.csv")
    accounting_events = []
    if accounting_registry.exists():
        decisions = pd.read_csv(accounting_registry, dtype={"cik": str})
        confirmed = decisions.loc[decisions.review_status.eq("confirmed_non_reliance_disclosure")]
        accounting_events = confirmed.to_dict("records")
        for company in companies:
            company["non_reliance_events"] = confirmed.loc[confirmed.cik.eq(str(company["cik"]).zfill(10))].to_dict("records")
    ciks = [str(c["cik"]).zfill(10) for c in companies]
    if len(set(ciks)) != len(ciks) or any(not re.fullmatch(r"\d{10}", c) for c in ciks):
        raise ValueError("Companies require unique valid CIKs")
    if min(annual_documents, quarterly_documents, insider_documents) < 0:
        raise ValueError("Document limits must be nonnegative")
    if pd.Timestamp(start) > pd.Timestamp(config["observation_end"]):
        raise ValueError("Start follows observation end")
    output, raw = Path(output), Path(raw)
    output.mkdir(parents=True, exist_ok=True)
    sources, errors, registry, filings, frames, cells, conflicts = [], [], [], [], [], [], []
    if Path(bankruptcy_events).exists():
        sources.append({"url": "https://lopucki.law.ufl.edu/download_cases_table.php",
                        "path": bankruptcy_events, "sha256": sha256(Path(bankruptcy_events).read_bytes()).hexdigest()})
    if accounting_registry.exists():
        sources.append({"path": str(accounting_registry), "sha256": sha256(accounting_registry.read_bytes()).hexdigest(),
                        "description": "Inspected Item 4.02 disclosure decisions with original SEC source URLs and hashes"})
    documents, chunks, transactions, pairs, questions, excluded_inputs = [], [], [], [], [], []
    for company, cik in zip(companies, ciks):
        company = {**company, "cik": cik}
        print(f"Building {company['ticker']} ({cik})", flush=True)
        payload, source = companyfacts(cik) if facts_directory == "data/raw/validation" else companyfacts(cik, directory=facts_directory)
        sources.append(source)
        if company.get("event_source"):
            # Reuse the original, already-cached bankruptcy source documents,
            # including their checksums in this dataset's own manifest.
            sources.append(fetch(company["event_source"],
                Path(company["event_source_path"]) if company.get("event_source_path") else
                Path("data/raw/validation/events") / f"{company['ticker']}.html"))
        for event in company.get("aaer_events", []):
            sources.append(fetch(event["source_url"], raw / "aaer" / f"{cik}_{event['aaer_number']}.pdf"))
        if str(payload["cik"]).zfill(10) != cik:
            raise ValueError("Company Facts CIK mismatch")
        rows = []
        try:
            rows, current_name = load_submissions(company, start, config["observation_end"], raw, sources)
        except Exception as exc:
            current_name = payload.get("entityName", "")
            errors.append({"cik": cik, "stage": "submissions", "error": str(exc)})
        if rows:
            frame, provenance, issues = extract_history(payload, company["ticker"], public_filings=rows)
            frame = frame.loc[frame.filing_date.between(start, config["observation_end"])].copy()
            frames.append(frame)
            cells.extend(p for p in provenance if start <= p["filed"] <= config["observation_end"])
            for issue in issues:
                if issue.get("reason", "").startswith("Unreconciled"):
                    if start <= issue["filing_date"] <= config["observation_end"]:
                        excluded_inputs.append({"cik": cik, "ticker": company["ticker"],
                            "accession_number": issue["accession"], "filing_date": issue["filing_date"],
                            "withheld_reason": issue["reason"]})
                else:
                    conflicts.append(issue)
        registry.append({"cik": cik, "research_ticker": company["ticker"], "current_sec_name": current_name,
            "historical_ticker_verified": False, "event_coverage_reviewed": False,
            "bankruptcy_event_date": company.get("event_date"), "bankruptcy_source": company.get("event_source"),
            "submissions_status": "available" if rows else "unavailable_or_empty"})
        filings.extend(rows)
        # Pre-bankruptcy document pairs preserve distressed-company research coverage.
        cutoff = min(config["observation_end"], company.get("event_date") or config["observation_end"])
        for form, limit in (("10-K", annual_documents), ("10-Q", quarterly_documents), ("4", insider_documents)):
            selected = sorted([r for r in rows if r["filing_type"] == form and
                (r["filing_date"] < cutoff if company.get("event_date") else r["filing_date"] <= cutoff)],
                key=lambda r: (r["filing_date"], r["accession_number"]))
            selected = selected[-limit:] if limit else []
            texts = []
            for row in selected:
                path = raw / "filings" / cik / (row["accession_number"] + (".raw.xml" if form == "4" else ".html"))
                try:
                    source = fetch(row["source_url"], path)
                    sources.append(source)
                    text = path.read_text(encoding="utf-8-sig")
                    if form == "4":
                        transactions.extend(parse_insider_xml(text, row))
                        continue
                    sections = extract_sections(text, form)
                    documents.append({**row, "local_path": str(path), "sha256": source["sha256"],
                        "available_sections": list(sections), "missing_sections": sorted({"risk_factors", "management_discussion"} - sections.keys())})
                    for section, body in sections.items():
                        for i, chunk in enumerate(chunk_section(body)):
                            chunk_id = sha256(f"{cik}:{row['accession_number']}:{section}:{i}:{chunk}".encode()).hexdigest()
                            chunks.append({**row, "section": section, "chunk_id": chunk_id,
                                           "chunk_index": i, "text": chunk, "document_sha256": source["sha256"]})
                    texts.append((row, text))
                except Exception as exc:
                    errors.append({"cik": cik, "accession_number": row["accession_number"], "stage": "document", "error": str(exc)})
            for (previous, old), (current, new) in zip(texts, texts[1:]):
                if previous["filing_date"] >= current["filing_date"]:
                    continue
                def filing(r):
                    return Filing(**{key: r[key] for key in ("cik", "filing_type", "filing_date", "accession_number", "primary_document")})
                report = compare_filings(filing(previous), filing(current), old, new)
                pair_id = f"{cik}_{previous['accession_number']}_{current['accession_number']}"
                write_json(output / "changes" / f"{pair_id}.json", report)
                pairs.append({"pair_id": pair_id, "cik": cik, "filing_type": form,
                    "previous_accession": previous["accession_number"], "current_accession": current["accession_number"],
                    "report_path": str(output / "changes" / f"{pair_id}.json"), "human_reviewed": False})
                questions.append({"id": pair_id, "cik": cik, "ticker": company["ticker"],
                    "question": "What changed in Risk Factors and MD&A between these filings?",
                    "as_of": current["filing_date"], "expected_answer": None, "review_status": "pending",
                    "source_urls": [previous["source_url"], current["source_url"]],
                    "expected_evidence_chunk_ids": [], "automated_report_is_ground_truth": False})
    features = pd.concat(frames, ignore_index=True)
    if features.latest_input_filing_date.gt(features.filing_date).any():
        raise ValueError("Future inputs in financial features")
    if filings:
        metadata = pd.DataFrame(filings)[["cik", "accession_number", "filing_date",
            "source_url", "primary_document", "acceptance_datetime"]]
        features = features.merge(metadata, how="left", on=["cik", "accession_number", "filing_date"],
                                  validate="one_to_one")
    else:
        features["source_url"] = None
    withheld = features.loc[features.source_url.isna()].copy()
    withheld["withheld_reason"] = "No matching original submission with identical CIK, accession and public date"
    withheld = pd.concat([withheld, pd.DataFrame(excluded_inputs)], ignore_index=True)
    withheld.to_csv(output / "financial_rows_withheld.csv", index=False)
    features = features.loc[features.source_url.notna()].copy().reset_index(drop=True)
    if features.empty:
        raise ValueError("No financial rows reconcile to original SEC submissions; inspect financial_rows_withheld.csv")
    features = financial_labels(features, [{**c, "cik": cik} for c, cik in zip(companies, ciks)], config["observation_end"])
    features.to_csv(output / "financial_features.csv", index=False)
    pd.DataFrame(cells).to_csv(output / "fact_provenance.csv", index=False)
    for name, records in (("companies", registry), ("filings", filings), ("documents", documents),
        ("accounting_event_candidates", accounting_candidates(filings)), ("insider_transactions", transactions),
        ("filing_pairs", pairs)):
        write_jsonl(output / f"{name}.jsonl", records)
    write_jsonl(output / "retrieval_chunks.jsonl", chunks)
    aaer_events = [{"cik": cik, "ticker": company["ticker"], **event}
                   for company, cik in zip(companies, ciks) for event in company.get("aaer_events", [])]
    write_jsonl(output / "aaer_events.jsonl", aaer_events)
    write_jsonl(output / "non_reliance_events.jsonl", accounting_events)
    write_json(output / "bankruptcy_event_reconciliation.json", event_reconciliation)
    # Keep human review files intact across rebuilds.
    merge_review_rows(output / "qa_review.jsonl", questions, "id")
    coverage = output / "outcome_coverage_review.csv"
    new_coverage = pd.DataFrame([{**r, "coverage_start": "", "observed_through": "", "reviewer": "",
            "coverage_source_url": "", "aaer_coverage_reviewed": False,
            "restatement_coverage_reviewed": False} for r in registry])
    if coverage.exists():
        old_coverage = pd.read_csv(coverage, dtype={"cik": str})
        new_coverage = pd.concat([old_coverage, new_coverage.loc[~new_coverage.cik.isin(old_coverage.cik)]], ignore_index=True)
    new_coverage.to_csv(coverage, index=False)
    finqa_count = 0
    if finqa:
        try:
            finqa_count = export_finqa(raw, output, sources)
        except Exception as exc:
            errors.append({"stage": "finqa", "error": str(exc)})
    summary = {"companies": len(registry), "financial_rows": len(features), "filings": len(filings),
        "financial_rows_linked_to_filings": int(features.source_url.notna().sum()),
        "financial_rows_withheld": len(withheld),
        "documents": len(documents), "chunks": len(chunks), "filing_pairs": len(pairs),
        "documents_with_missing_sections": sum(bool(d["missing_sections"]) for d in documents),
        "generated_qa_review_tasks": len(questions),
        "insider_transactions": len(transactions), "accounting_event_candidates": len(accounting_candidates(filings)),
        "bankruptcy_companies": sum(bool(c.get("event_date")) for c in companies),
        "positive_bankruptcy_1yr_rows": int(features.target_distressed_1yr.eq(1).sum()),
        "bankruptcy_conflict_companies": sum(bool(c.get("bankruptcy_label_conflict")) for c in companies),
        "unknown_bankruptcy_1yr_rows": int(features.target_distressed_1yr.isna().sum()),
        "cited_aaer_events": len(aaer_events),
        "positive_aaer_3yr_rows": int(features.target_aaer_enforcement_flag.eq(1).sum()),
        "confirmed_non_reliance_disclosures": len(accounting_events),
        "positive_non_reliance_3yr_rows": int(features.target_restatement_flag.eq(1).sum()),
        "finqa_development_questions": finqa_count, "errors": errors, "fact_conflicts": conflicts,
        "production_ready": False, "complete": not errors,
        "limitations": ["Purposive sample; not a historical index membership dataset.",
            "Unknown outcome coverage is not a negative label. AAER and non-reliance negatives remain unknown.",
            "Document and insider samples are bounded; not complete historical transaction coverage.",
            "FinQA is a separate numerical reasoning benchmark, not joined to SEC identities.",
            "Automated filing diffs are review aids, not labeled materiality judgments.",
            "No model retraining or performance claim; existing holdout remains exposed.",
            "No stock returns or validated historical index membership included."],
        "config_sha256": sha256(config_bytes).hexdigest(),
        "start": start, "observation_end": config["observation_end"],
        "document_limits": {"annual": annual_documents, "quarterly": quarterly_documents, "insider": insider_documents}}
    write_json(output / "source_manifest.json", sources)
    write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"errors", "fact_conflicts", "limitations"}}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/research_universe.json")
    parser.add_argument("--output", default="data/processed/research")
    parser.add_argument("--raw", default="data/raw/research")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--annual-documents", type=int, default=2)
    parser.add_argument("--quarterly-documents", type=int, default=1)
    parser.add_argument("--insider-documents", type=int, default=1)
    parser.add_argument("--skip-finqa", action="store_true")
    parser.add_argument("--bankruptcy-events", default="data/processed/lopucki/chapter11_events.csv")
    parser.add_argument("--facts-directory", default="data/raw/validation")
    args = vars(parser.parse_args())
    args["config_path"] = args.pop("config")
    args["finqa"] = not args.pop("skip_finqa")
    summary = build(**args)
    if not summary["complete"]:
        raise SystemExit("Partial build: inspect summary.json errors; cached downloads can be reused.")


if __name__ == "__main__":
    main()
