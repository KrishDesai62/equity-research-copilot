import io
import json
import zipfile

import pandas as pd
import pytest

from src.agents.qa_agent import answer_question
from src.data_layer.lopucki import import_brd, normalize_cik, parse_archive, reconcile_events
from src.rag.local_index import build_index, LocalIndex
from src.validation.model_readiness import reviewed_labels
from src.data_layer.research_dataset import financial_labels


def archive(path, date="6/1/2020"):
    csv = f"NameCorp,DateFiled,CikBefore,PrimaryKey,Chapter\nExample,{date},123.0,1,11\nOther,1/1/2010,,2,7\n"
    with zipfile.ZipFile(path, "w") as data:
        data.writestr("Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.csv", csv)


def test_free_brd_import_keeps_missing_identity_and_chapters_separate(tmp_path):
    path = tmp_path / "archive.zip"
    archive(path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"companies": [{"ticker": "EX", "cik": "0000000123", "event_date": "2020-06-01"}]}))
    report = import_brd(path, tmp_path / "out", config)
    assert report["total_cases"] == 2
    assert report["chapter11_events_with_exact_cik"] == 1
    assert report["missing_or_invalid_cik_cases"] == 1
    assert report["negative_labels_created"] == 0
    assert normalize_cik("123.0") == "0000000123"
    assert normalize_cik("123.5") is None


def test_release_cutoff_is_enforced(tmp_path):
    path = tmp_path / "future.zip"
    archive(path, "1/1/2024")
    with pytest.raises(ValueError, match="snapshot"):
        parse_archive(path)


def test_conflicting_dates_are_flagged_and_never_overwrite_existing_citation(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame([{"cik": "0000000123", "filing_date": "2020-06-02", "source_url": "https://lopucki.law.ufl.edu/",
                   "chapter": 11, "coverage_end": "2022-12-31"}]).to_csv(path, index=False)
    companies, audit = reconcile_events([{"cik": "0000000123", "ticker": "EX", "event_date": "2020-06-01", "event_source": "https://www.sec.gov/original"}], path)
    assert companies[0]["event_date"] == "2020-06-01"
    assert companies[0]["bankruptcy_label_conflict"]
    assert audit[0]["status"] == "date_conflict_requires_review"


def test_brd_does_not_authorize_2024_negative_labels():
    features = pd.DataFrame({"cik": ["0000000123"], "filing_date": ["2023-01-01"], "removed_date": [pd.NaT]})
    coverage = pd.DataFrame([{"cik": "0000000123", "event_coverage_reviewed": True,
        "coverage_start": "2020-01-01", "observed_through": "2024-12-31", "reviewer": "Analyst",
        "coverage_source_url": "https://lopucki.law.ufl.edu/download_cases_table.php"}])
    with pytest.raises(ValueError, match="2022-12-31"):
        reviewed_labels(features, coverage, [], "2024-12-31")


def test_source_conflict_and_non_reliance_are_separate_targets():
    features = pd.DataFrame({"cik": ["0000000123", "0000000123"], "filing_date": ["2020-01-01", "2021-02-01"]})
    companies = [{"cik": "0000000123", "event_date": "2020-06-01", "event_source": "https://www.sec.gov/event",
        "bankruptcy_label_conflict": True, "non_reliance_events": [{"filing_date": "2021-01-01", "source_url": "https://www.sec.gov/nonreliance"}]}]
    labeled = financial_labels(features, companies, "2022-12-31")
    assert labeled.target_distressed_1yr.isna().all()
    assert not labeled.valid_for_training.any()
    assert labeled.target_restatement_flag.iloc[0] == 1
    assert pd.isna(labeled.target_restatement_flag.iloc[1])
    assert labeled.restatement_label_available_date.iloc[0] == pd.Timestamp("2021-01-01")
    assert labeled.target_aaer_enforcement_flag.isna().all()


def index_rows():
    return [{"chunk_id": str(i), "text": "Liquidity risk and cash flows.", "ticker": ticker,
        "filing_type": "10-K", "filing_date": filed, "section": "risk_factors",
        "source_url": f"https://www.sec.gov/{i}", "accession_number": str(i), "cik": "1"}
        for i, ticker, filed in [(1, "EX", "2020-01-01"), (2, "EX", "2025-01-01"), (3, "OTHER", "2020-01-01")]]


def test_local_index_qa_cites_exact_passages_and_excludes_future_and_other_firms(tmp_path):
    source = tmp_path / "chunks.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in index_rows()))
    path = tmp_path / "index.sqlite"
    assert build_index(source, path)["chunks"] == 3
    index = LocalIndex(path)
    answer = answer_question("liquidity risk", "EX", as_of="2024-12-31", retriever=index.retrieve)
    assert len(answer["citations"]) == 1
    assert answer["citations"][0]["chunk_id"] == "1"
    assert answer["citations"][0]["excerpt"] in answer["answer"]
    assert answer_question("cash", "EX", as_of="1900-01-01", retriever=index.retrieve)["status"] == "unavailable"
    assert index.retrieve('" OR 1=1 --', ticker="MISSING") == []
    # A bad rebuild preserves the last valid persistent index.
    source.write_text("")
    with pytest.raises(ValueError, match="empty"):
        build_index(source, path)
    assert index.retrieve("cash", ticker="EX")


def test_qa_rejects_uncited_or_future_passages_even_if_retriever_is_wrong():
    def wrong(*args, **kwargs):
        return [{"ticker": "EX", "filing_date": "2025-01-01", "text": "Unsupported", "metadata": {}}]
    assert answer_question("cash", "EX", as_of="2020-01-01", retriever=wrong)["status"] == "unavailable"
