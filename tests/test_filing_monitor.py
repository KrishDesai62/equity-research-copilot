from dataclasses import asdict, replace
import io
import json
from pathlib import Path
from tempfile import gettempdir

from botocore.exceptions import ClientError
import pytest

from src.aws.filing_monitor import MonitorStore, handler, run_monitor, validate_watchlist
from src.aws.s3_cache import S3Cache
from src.data_layer.filing_changes import Filing


WATCHLIST = {"companies": [{"ticker": "TEST", "cik": "1"}]}
OLD = Filing("1", "10-K", "2024-02-01", "0000000001-24-000001", "annual.htm")
NEW = replace(OLD, filing_date="2025-02-01", accession_number="0000000001-25-000001")
NEXT = replace(NEW, filing_date="2026-02-01", accession_number="0000000001-26-000001")
STATE_KEY = "equity-research/monitor/state/0000000001.json"


class S3Client:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.fail_suffix = None

    def get_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}

    def put_object(self, **kwargs):
        if self.fail_suffix and kwargs["Key"].endswith(self.fail_suffix):
            raise RuntimeError("S3 upload failed")
        assert kwargs["ServerSideEncryption"] == "AES256"
        body = kwargs["Body"]
        self.objects[kwargs["Key"]] = body.read() if hasattr(body, "read") else body
        self.writes.append(kwargs["Key"])


@pytest.fixture
def store():
    return MonitorStore(S3Cache("test-bucket", client=S3Client()))


def records(*filings):
    return [{key: value for key, value in asdict(filing).items() if key != "cik"} for filing in filings]


def downloader(calls):
    def download(cik, accession, primary_document, *, cache, dest_dir):
        calls.append(accession)
        assert Path(gettempdir()).resolve() in dest_dir.resolve().parents
        return ("Item 1A. Risk Factors\nWe depend on suppliers and adverse market conditions may affect our future operations.\n"
                "Item 1B. Comments\nNone.\nItem 7. Management's Discussion and Analysis\n"
                "Our revenue increased during the year as demand grew throughout our principal markets.\n"
                "Item 7A. Market Risk\nNone.")
    return download


def run(store, filings, calls=None, **kwargs):
    calls = calls if calls is not None else []
    return run_monitor(WATCHLIST, store, list_fn=lambda *args: records(*filings),
                       download_fn=downloader(calls), today="2026-10-16", **kwargs)


def test_first_run_processes_only_latest_per_form_then_skips_duplicate(store):
    quarterly = replace(NEW, filing_type="10-Q", accession_number="0000000001-25-000002", filing_date="2025-05-01")
    calls = []
    summary = run(store, [OLD, NEW, quarterly], calls)
    assert summary["processed"] == 2
    assert calls.count(OLD.accession_number) == 1  # Downloaded only as the comparison baseline.
    assert summary["reports"][0].endswith(f"{NEW.accession_number}.md")
    writes_before = len(store.cache.client.writes)
    assert run(store, [OLD, NEW, quarterly], calls)["processed"] == 0
    assert len(store.cache.client.writes) == writes_before
    assert set(json.loads(store.cache.client.objects[STATE_KEY])["known_accessions"]) == {
        OLD.accession_number, NEW.accession_number, quarterly.accession_number}


def test_new_filing_uses_immediately_previous_same_form(store):
    run(store, [OLD, NEW])
    calls = []
    summary = run(store, [OLD, NEW, NEXT], calls)
    assert summary["processed"] == 1
    assert calls == [NEXT.accession_number, NEW.accession_number]
    report = json.loads(store.cache.client.objects[summary["reports"][0].replace(".md", ".json")])
    assert report["previous"]["accession_number"] == NEW.accession_number
    assert report["current"]["accession_number"] == NEXT.accession_number


def test_report_failure_is_not_checkpointed_and_retry_finishes(store):
    store.cache.client.fail_suffix = ".md"
    with pytest.raises(RuntimeError, match="incomplete"):
        run(store, [OLD, NEW])
    state = json.loads(store.cache.client.objects[STATE_KEY])
    assert NEW.accession_number not in state["known_accessions"]
    store.cache.client.fail_suffix = None
    assert run(store, [OLD, NEW])["processed"] == 1
    assert run(store, [OLD, NEW])["processed"] == 0
    reports = [key for key in store.cache.client.objects if "/reports/" in key]
    assert len(reports) == 2  # One JSON and one Markdown key, even after a retry.


def test_multiple_new_filings_and_late_arrivals_are_not_skipped(store):
    run(store, [OLD, NEW])
    late = replace(NEW, filing_date="2024-08-01", accession_number="0000000001-24-000005")
    summary = run(store, [OLD, NEW, NEXT, late])
    assert summary["processed"] == 2


def test_no_predecessor_still_saves_filing_and_status_report(store):
    calls = []
    summary = run(store, [NEW], calls)
    assert calls == [NEW.accession_number]
    report = json.loads(store.cache.client.objects[summary["reports"][0].replace(".md", ".json")])
    assert report["status"] == "no_comparable_predecessor"


def test_other_companies_continue_after_one_fails(store):
    watchlist = {"companies": WATCHLIST["companies"] + [{"ticker": "OTHER", "cik": "2"}]}
    def listings(cik, forms):
        if cik == "0000000001":
            raise RuntimeError("SEC temporarily unavailable")
        return records(NEW)
    with pytest.raises(RuntimeError, match="incomplete"):
        run_monitor(watchlist, store, list_fn=listings, download_fn=downloader([]), today="2026-10-16")
    assert any("monitor/reports/0000000002/" in key for key in store.cache.client.objects)


def test_future_filings_and_amendments_are_ignored(store):
    future = replace(NEXT, filing_date="2027-01-01", accession_number="0000000001-27-000001")
    entries = records(OLD, NEW, future) + [{**records(NEW)[0], "filing_type": "10-K/A"}]
    summary = run_monitor(WATCHLIST, store, list_fn=lambda *args: entries,
                          download_fn=downloader([]), today="2026-10-16")
    assert summary["processed"] == 1


def test_time_budget_stops_without_baselining_pending_work(store):
    class Context:
        def get_remaining_time_in_millis(self):
            return 1000
    with pytest.raises(RuntimeError, match="Not enough Lambda time"):
        run(store, [OLD, NEW], context=Context())
    assert STATE_KEY not in store.cache.client.objects


def test_corrupt_state_is_not_silently_reset(store):
    store.cache.client.objects[STATE_KEY] = b'{"schema_version": 9}'
    with pytest.raises(RuntimeError, match="Invalid checkpoint"):
        run(store, [OLD, NEW])


@pytest.mark.parametrize("watchlist", [{}, {"companies": []}, {"companies": [{"cik": "bad"}]},
    {"companies": [{"cik": "1"}, {"cik": "0000000001"}]}])
def test_invalid_watchlist(watchlist):
    with pytest.raises(ValueError):
        validate_watchlist(watchlist)


def test_s3_forbidden_is_not_missing_state(store):
    def forbidden(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    store.cache.client.get_object = forbidden
    with pytest.raises(ClientError):
        store.read_json("monitor/state/1.json")


def test_handler_loads_config_from_s3_not_event(store, monkeypatch):
    monkeypatch.setattr(S3Cache, "from_env", lambda: store.cache)
    store.write_json("monitor/watchlist.json", WATCHLIST)
    def monitor(watchlist, monitor_store, **kwargs):
        assert watchlist == WATCHLIST
        return {"processed": 1}
    monkeypatch.setattr("src.aws.filing_monitor.run_monitor", monitor)
    assert handler({"companies": []}, None) == {"processed": 1}


def test_handler_requires_uploaded_watchlist(store, monkeypatch):
    monkeypatch.setattr(S3Cache, "from_env", lambda: store.cache)
    with pytest.raises(ValueError, match="Upload your watchlist"):
        handler({}, None)


def test_real_ingestion_reuses_s3_between_invocations(store, monkeypatch):
    from src.data_layer import edgar_ingest
    requests = []
    class Response:
        text = "Raw SEC document without parseable sections"
        def raise_for_status(self):
            pass
    def get(url, **kwargs):
        requests.append(url)
        assert kwargs["timeout"] == 60
        return Response()
    monkeypatch.setattr(edgar_ingest, "EDGAR_USER_AGENT", "Monitor test@research.org")
    monkeypatch.setattr(edgar_ingest.requests, "get", get)
    monkeypatch.setattr(edgar_ingest.limiter, "wait", lambda: None)
    run_monitor(WATCHLIST, store, list_fn=lambda *args: records(OLD, NEW), today="2026-10-16")
    assert len(requests) == 2
    assert any("filings/0000000001/" in key for key in store.cache.client.objects)
    run_monitor(WATCHLIST, store, list_fn=lambda *args: records(OLD, NEW, NEXT), today="2026-10-16")
    assert len(requests) == 3  # Prior filing restored from S3 in a fresh temp directory.


def test_handler_records_failure_then_recovery_health(store, monkeypatch):
    monkeypatch.setattr(S3Cache, 'from_env', lambda: store.cache)
    with pytest.raises(ValueError, match='Upload your watchlist'):
        handler({}, None)
    assert store.read_json('monitor/health.json')['state'] == 'failure'
    store.write_json('monitor/watchlist.json', WATCHLIST)
    monkeypatch.setattr('src.aws.filing_monitor.run_monitor', lambda *a, **kw: {'processed': 1, 'companies_checked': 1})
    handler({}, None)
    assert store.read_json('monitor/health.json')['state'] == 'success'
