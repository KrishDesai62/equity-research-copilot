"""Live-file checks for cited Q&A, section availability and accounting review."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from src.agents.qa_agent import answer_question
from src.data_layer.filing_changes import normalize_filing, _HEADINGS
from src.data_layer.research_dataset import write_json, write_jsonl
from src.rag.local_index import build_index, LocalIndex
from src.validation.download import fetch


def run(research="data/processed/research", download_accounting=False, as_of=None):
    root = Path(research)
    as_of = as_of or json.loads((root / "summary.json").read_text())["observation_end"]
    report = {"index": build_index(root / "retrieval_chunks.jsonl", root / "filings.sqlite")}
    index = LocalIndex(root / "filings.sqlite")
    chunks = [json.loads(line) for line in (root / "retrieval_chunks.jsonl").read_text().splitlines()]
    by_id = {c["chunk_id"]: c for c in chunks}
    companies = sorted({c["ticker"] for c in chunks})
    checks, samples = [], []
    for ticker in companies:
        result = answer_question("revenue cash flow liquidity risk", ticker, as_of=as_of, retriever=index.retrieve)
        valid = bool(result["citations"]) and all(
            c["chunk_id"] in by_id and c["excerpt"] == by_id[c["chunk_id"]]["text"] and
            c["source_url"] == by_id[c["chunk_id"]]["source_url"] and
            c["filing_date"] <= as_of for c in result["citations"])
        checks.append({"ticker": ticker, "cited_passages_match_index": valid})
        if ticker in {"AAPL", "BBBY", "KHC"}:
            samples.append({"ticker": ticker, **result})
    early = answer_question("revenue", companies[0], as_of="1900-01-01", retriever=index.retrieve)
    unknown = answer_question("revenue", "NOT_A_COMPANY", retriever=index.retrieve)
    report["qa_checks"] = {"companies_checked": len(checks), "passed": sum(c["cited_passages_match_index"] for c in checks),
        "as_of": as_of,
        "early_date_abstains": early["status"] == "unavailable", "unknown_company_abstains": unknown["status"] == "unavailable",
        "scope": "Source fidelity and filtering checks only; not answer relevance or FinQA numerical reasoning accuracy."}
    write_json(root / "qa_smoke_samples.json", samples)
    reviews = []
    for line in (root / "documents.jsonl").read_text().splitlines():
        doc = json.loads(line)
        if not doc["missing_sections"]:
            continue
        path = Path(doc["local_path"])
        text = normalize_filing(path.read_text())
        headings = [(h.group(1), h.group(2)) for h in _HEADINGS.finditer(text)]
        for section in doc["missing_sections"]:
            item = "1A" if section == "risk_factors" else "7" if doc["filing_type"] == "10-K" else "2"
            exists = any(number.upper() == item for number, _ in headings)
            reviews.append({"ticker": doc["ticker"], "accession_number": doc["accession_number"],
                "section": section, "source_url": doc["source_url"], "sha256": sha256(path.read_bytes()).hexdigest(),
                "status": "heading_present_needs_parser_review" if exists else "no_standalone_heading_in_source",
                "note": "Missing quarterly disclosure is not unchanged risk or a deleted annual risk inventory.",
                "heading_evidence": headings[-10:]})
    write_json(root / "section_availability_review.json", reviews)
    report["section_review"] = {"missing_sections_checked": len(reviews),
        "without_standalone_heading": sum(r["status"] == "no_standalone_heading_in_source" for r in reviews)}
    accounting, failures = [], []
    for line in (root / "accounting_event_candidates.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["event_kind"] != "non_reliance":
            continue
        path = Path("data/raw/research/accounting") / (row["accession_number"] + ".html")
        try:
            if download_accounting:
                source = fetch(row["source_url"], path)
            elif path.exists():
                source = {"sha256": sha256(path.read_bytes()).hexdigest()}
            else:
                continue
            text = normalize_filing(path.read_text())
            starts = list(re.finditer(r"(?im)^\s*(?:item\s+)?4\.02(?=\s|\.|Non-Reliance)", text))
            excerpts = []
            for match in starts:
                tail = text[match.start():]
                end = re.search(r"(?im)^\s*item\s+\d\.\d{2}\b", tail[20:])
                excerpts.append(tail[:20 + end.start()] if end else tail[:12000])
            excerpt = max(excerpts, key=lambda e: (bool(re.search(r"non[\s-]*reliance", e[:300], re.I)), len(e)), default="")
            accounting.append({**row, "document_sha256": source["sha256"], "excerpt": excerpt,
                "review_status": "source_excerpt_ready_for_review" if excerpt else "heading_not_located_requires_manual_review",
                "is_fraud_label": False, "local_path": str(path)})
        except Exception as exc:
            failures.append({"accession_number": row["accession_number"], "error": str(exc)})
    write_jsonl(root / "accounting_source_review.jsonl", accounting)
    report["accounting_review"] = {"documents_available": len(accounting), "failures": failures,
        "reviewed_labels_created": 0}
    report["production_ready"] = False
    write_json(root / "research_checks.json", report)
    print(json.dumps(report, indent=2))
    if report["qa_checks"]["passed"] != len(checks) or not report["qa_checks"]["early_date_abstains"] or not report["qa_checks"]["unknown_company_abstains"] or failures:
        raise ValueError("Research checks failed; inspect research_checks.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", default="data/processed/research")
    parser.add_argument("--download-accounting", action="store_true")
    parser.add_argument("--as-of")
    run(**vars(parser.parse_args()))
