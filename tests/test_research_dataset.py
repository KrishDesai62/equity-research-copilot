import json

import pandas as pd
import pytest

from src.data_layer.research_dataset import (
    accounting_candidates, financial_labels, merge_review_rows,
    parse_insider_xml, submission_rows,
)
from src.data_layer import research_dataset as dataset
from src.rag.chunking import chunk_section


def block():
    return {"accessionNumber": ["0000000001-23-000001", "0000000001-25-000001"],
        "filingDate": ["2023-01-02", "2025-01-02"], "form": ["8-K", "10-K"],
        "primaryDocument": ["event.htm", "annual.htm"], "items": ["1.03,4.02", ""]}


def test_long_filing_sentence_chunks_terminate_with_bounded_overlap():
    words = [f"word{i}" for i in range(1200)]
    chunks = chunk_section("Short introduction. " + " ".join(words), max_tokens=500, overlap=50)
    assert all(len(chunk.split()) <= 500 for chunk in chunks)
    assert len(chunks) < 10
    assert all(word in " ".join(chunks).split() for word in words)
    assert chunks[-1].split()[-1] == "word1199"
    with pytest.raises(ValueError):
        chunk_section("text", max_tokens=50, overlap=50)


def test_submissions_use_public_date_and_keep_distinct_event_types():
    rows = submission_rows(block(), "0000000001", "TEST", "2020-01-01", "2024-12-31")
    assert len(rows) == 1
    candidates = accounting_candidates(rows)
    assert {r["event_kind"] for r in candidates} == {"non_reliance", "bankruptcy_or_receivership"}
    assert all(r["review_status"] == "candidate_requires_review" for r in candidates)


def test_malformed_submissions_are_not_silently_truncated():
    data = block()
    data["filingDate"] = []
    with pytest.raises(ValueError, match="arrays"):
        submission_rows(data, "0000000001", "TEST", "2020-01-01", "2024-12-31")


def test_form4_downloads_underlying_xml_instead_of_xsl_html_view():
    data = block()
    data["form"][0] = "4"
    data["primaryDocument"][0] = "xslF345X05/ownership.xml"
    row = submission_rows(data, "0000000001", "TEST", "2020-01-01", "2024-12-31")[0]
    assert row["primary_document"] == "ownership.xml"
    assert row["source_url"].endswith("/ownership.xml")
    assert "xsl" not in row["source_url"]


def test_unknown_outcomes_stay_unknown_and_bankruptcy_is_retained():
    frame = pd.DataFrame({"cik": ["0000000001"] * 3 + ["0000000002"],
        "filing_date": ["2022-01-01", "2023-01-01", "2023-07-01", "2022-01-01"]})
    companies = [{"cik": "0000000001", "event_date": "2023-06-01", "event_source": "https://www.sec.gov/event"},
                 {"cik": "0000000002", "event_date": None}]
    labeled = financial_labels(frame, companies, "2024-12-31")
    assert pd.isna(labeled.target_distressed_1yr.iloc[0])
    assert labeled.target_distressed_1yr.iloc[1] == 1
    assert pd.isna(labeled.target_distressed_1yr.iloc[2])  # Post-bankruptcy excluded.
    assert pd.isna(labeled.target_distressed_1yr.iloc[3])  # No event != healthy.
    assert labeled.label_available_date_1yr.iloc[1] == pd.Timestamp("2023-06-01")
    assert labeled.target_aaer_enforcement_flag.isna().all()
    assert labeled.target_restatement_flag.isna().all()


def test_insider_awards_are_not_purchase_signals_and_availability_is_filing_date():
    xml = """<ownershipDocument><issuer><issuerCik>1</issuerCik></issuer>
      <nonDerivativeTable><nonDerivativeTransaction>
        <transactionDate><value>2023-01-01</value></transactionDate>
        <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
        <transactionAmounts><transactionShares><value>100</value></transactionShares>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
      </nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""
    filing = {"cik": "0000000001", "filing_date": "2023-01-05"}
    result = parse_insider_xml(xml, filing)[0]
    assert result["purchase_sale_signal"] == "other"
    assert result["available_date"] == "2023-01-05"
    assert result["price_per_share"] is None
    with pytest.raises(ValueError, match="issuer"):
        parse_insider_xml(xml, {**filing, "cik": "0000000002"})


def test_cited_aaer_is_separate_from_bankruptcy_and_unreviewed_negatives():
    frame = pd.DataFrame({"cik": ["0000000001"] * 4,
        "filing_date": ["2017-01-01", "2020-01-01", "2021-09-03", "2022-01-01"]})
    companies = [{"cik": "0000000001", "aaer_events": [{"release_date": "2021-09-03",
        "aaer_number": "4248", "source_url": "https://www.sec.gov/order.pdf"}]}]
    labeled = financial_labels(frame, companies, "2024-12-31")
    assert labeled.target_aaer_enforcement_flag.iloc[1] == 1
    assert labeled.target_aaer_enforcement_flag.drop(index=1).isna().all()
    assert labeled.aaer_label_available_date.iloc[1] == pd.Timestamp("2021-09-03")
    assert labeled.target_distressed_1yr.isna().all()
    assert labeled.target_restatement_flag.isna().all()


def test_rebuild_preserves_human_review_and_appends_new_pairs(tmp_path):
    path = tmp_path / "review.jsonl"
    merge_review_rows(path, [{"id": "a", "answer": "human reviewed"}], "id")
    merge_review_rows(path, [{"id": "a", "answer": None}, {"id": "b", "answer": None}], "id")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"id": "a", "answer": "human reviewed"}, {"id": "b", "answer": None}]


def test_archived_submission_history_is_loaded_for_old_companies(tmp_path, monkeypatch):
    recent = {k: [] for k in ("accessionNumber", "filingDate", "form", "primaryDocument")}
    payload = {"cik": 1, "name": "Old Company", "filings": {"recent": recent,
        "files": [{"name": "CIK0000000001-submissions-001.json", "filingFrom": "2020-01-01", "filingTo": "2023-12-31"}]}}
    urls = []
    def fake_fetch(url, path):
        urls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(block() if "submissions-001" in url else payload))
        return {"url": url}
    monkeypatch.setattr(dataset, "fetch", fake_fetch)
    rows, name = dataset.load_submissions({"cik": "0000000001", "ticker": "OLD"},
        "2020-01-01", "2024-12-31", tmp_path, [])
    assert len(rows) == 1 and name == "Old Company"
    assert len(urls) == 2


def test_connected_build_retains_unknown_labels_and_emits_joinable_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"observation_end": "2024-12-31", "companies": [
        {"cik": "0000000001", "ticker": "TEST", "event_date": "2023-06-01", "event_source": "https://www.sec.gov/event"}]}))
    frame = pd.DataFrame({"cik": ["0000000001", "0000000001"],
        "accession_number": ["0000000001-23-000001", "0000000001-23-000099"],
        "filing_date": ["2023-01-02", "2023-02-01"],
        "latest_input_filing_date": ["2023-01-02", "2023-02-01"]})
    monkeypatch.setattr(dataset, "companyfacts", lambda cik: ({"cik": 1}, {"url": "facts"}))
    monkeypatch.setattr(dataset, "extract_history", lambda *args, **kwargs: (frame.copy(), [], []))
    data = block()
    data["form"][0] = "10-K"
    rows = submission_rows(data, "0000000001", "TEST", "2020-01-01", "2024-12-31")
    monkeypatch.setattr(dataset, "load_submissions", lambda *args: (rows, "Test Company"))
    def fake_fetch(url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("document")
        return {"url": url, "sha256": "hash"}
    monkeypatch.setattr(dataset, "fetch", fake_fetch)
    monkeypatch.setattr(dataset, "extract_sections", lambda *args: {"risk_factors": "A cited risk paragraph."})
    output = tmp_path / "output"
    summary = dataset.build(config, output, tmp_path / "raw", finqa=False)
    chunks = [json.loads(line) for line in (output / "retrieval_chunks.jsonl").read_text().splitlines()]
    assert summary["complete"] and not summary["production_ready"]
    assert summary["positive_bankruptcy_1yr_rows"] == 1
    assert summary["financial_rows_withheld"] == 1
    assert summary["financial_rows"] == 1
    assert len(pd.read_csv(output / "financial_rows_withheld.csv")) == 1
    assert chunks[0]["accession_number"] == rows[0]["accession_number"]
    assert chunks[0]["source_url"] == rows[0]["source_url"]
    assert (output / "outcome_coverage_review.csv").exists()
