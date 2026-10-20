"""Scheduled, sequential SEC watchlist ingestion with durable S3 checkpoints.

The deployment reserves one Lambda execution at a time. Do not run another writer
against the same state prefix. Checkpoints follow report uploads, so failed work
is retried; report keys are deterministic for safe at-least-once execution.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from botocore.exceptions import ClientError

from src.aws.s3_cache import S3Cache
from src.data_layer.filing_changes import Filing, compare_filings, render_markdown

logger = logging.getLogger(__name__)
FORMS = ("10-K", "10-Q")


class MonitorStore:
    """Small JSON state objects and reports share the existing encrypted S3 cache."""
    def __init__(self, cache):
        self.cache = cache

    def read_json(self, key):
        try:
            response = self.cache.client.get_object(Bucket=self.cache.bucket, Key=self.cache.key(key))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None
            raise
        body = response["Body"]
        try:
            return json.loads(body.read())
        finally:
            body.close()

    def write_text(self, key, text, content_type):
        self.cache.client.put_object(Bucket=self.cache.bucket, Key=self.cache.key(key),
            Body=text.encode("utf-8"), ContentType=content_type, ServerSideEncryption="AES256")

    def write_json(self, key, value):
        self.write_text(key, json.dumps(value, indent=2), "application/json")


def validate_watchlist(value):
    """Explicit CIKs avoid ambiguous ticker lookup and ticker reuse."""
    if not isinstance(value, dict) or not isinstance(value.get("companies"), list):
        raise ValueError("Watchlist must contain a companies list")
    if not 1 <= len(value["companies"]) <= 30:
        raise ValueError("Use 1-30 companies per monitor deployment")
    companies, seen = [], set()
    for item in value["companies"]:
        if not isinstance(item, dict):
            raise ValueError("Each watchlist company must be an object with a CIK")
        cik = str(item.get("cik", ""))
        if not cik.isdigit() or not 0 < int(cik) or len(cik) > 10:
            raise ValueError("Each company needs a positive SEC CIK of at most 10 digits")
        cik = cik.zfill(10)
        if cik in seen:
            raise ValueError(f"Duplicate watchlist CIK: {cik}")
        seen.add(cik)
        companies.append({"cik": cik, "ticker": str(item.get("ticker", cik))})
    return companies


def _filings(cik, records, today):
    filings = {}
    for record in records:
        if record["filing_type"] not in FORMS or record["filing_date"] > today:
            continue
        filing = Filing(cik, **{key: record[key] for key in
                         ("filing_type", "filing_date", "accession_number", "primary_document")})
        filings[filing.accession_number] = filing
    return sorted(filings.values(), key=lambda filing: (filing.filing_date, filing.accession_number))


def run_monitor(watchlist, store, *, list_fn=None, download_fn=None, context=None, today=None):
    """Process new original 10-K/10-Q filings, checkpointing each successful one.

    First observation processes only the newest filing per form; older accessions
    are baselined, avoiding a surprise bulk backfill. Subsequent observations process
    every unseen accession in the recent submissions list, including late arrivals.
    A company failure does not block others, but the run raises to trigger retries.
    """
    from src.data_layer.edgar_ingest import list_filings, download_filing
    list_fn = list_fn or list_filings
    download_fn = download_fn or download_filing
    today = today or datetime.now(timezone.utc).date().isoformat()
    companies = validate_watchlist(watchlist)
    summary = {"companies_checked": 0, "processed": 0, "reports": [], "errors": []}
    with TemporaryDirectory(prefix="filing-monitor-") as temporary:
        for company in companies:
            cik = company["cik"]
            try:
                if context and context.get_remaining_time_in_millis() < 60000:
                    raise TimeoutError("Not enough Lambda time remaining; successful checkpoints are preserved")
                filings = _filings(cik, list_fn(cik, list(FORMS)), today)
                state_key = f"monitor/state/{cik}.json"
                state = store.read_json(state_key)
                if state is None:
                    latest = {form: next((f.accession_number for f in reversed(filings)
                                          if f.filing_type == form), None) for form in FORMS}
                    state = {"schema_version": 1, "cik": cik,
                             "known_accessions": [f.accession_number for f in filings
                                                  if f.accession_number not in latest.values()]}
                    # Persist baseline before work. Unprocessed latest filings remain pending on retry.
                    store.write_json(state_key, state)
                if (not isinstance(state, dict) or state.get("schema_version") != 1
                        or state.get("cik") != cik or not isinstance(state.get("known_accessions"), list)
                        or any(not isinstance(accession, str) for accession in state["known_accessions"])):
                    raise ValueError(f"Invalid checkpoint for {cik}")
                known = set(state["known_accessions"])
                for current in filings:
                    if current.accession_number in known:
                        continue
                    if context and context.get_remaining_time_in_millis() < 60000:
                        raise TimeoutError("Time budget exhausted; remaining filings will be retried")
                    previous = next((f for f in reversed(filings)
                                     if f.filing_type == current.filing_type and f.filing_date < current.filing_date), None)
                    # A new TemporaryDirectory per invocation prevents reliance on Lambda warm-disk state.
                    def download(filing):
                        return download_fn(filing.cik, filing.accession_number, filing.primary_document,
                                           cache=store.cache, dest_dir=Path(temporary) / "filings")
                    current_text = download(current)
                    prefix = f"monitor/reports/{cik}/{current.accession_number}"
                    if previous is None:
                        report = {"schema_version": 1, "status": "no_comparable_predecessor",
                                  "ticker": company["ticker"], "current": {**asdict(current), "source_url": current.url},
                                  "note": "Raw filing saved; no earlier same-form filing exists in recent submissions."}
                        markdown = f"# {company['ticker']}: {current.filing_type}\n\n{report['note']}\n\nSource: {current.url}\n"
                    else:
                        report = compare_filings(previous, current, download(previous), current_text)
                        report["ticker"] = company["ticker"]
                        markdown = render_markdown(report)
                    store.write_json(prefix + ".json", report)
                    store.write_text(prefix + ".md", markdown, "text/markdown; charset=utf-8")
                    # Only mark processed once raw data and BOTH report formats have reached S3.
                    known.add(current.accession_number)
                    state["known_accessions"] = sorted(known)
                    state["updated_at"] = datetime.now(timezone.utc).isoformat()
                    store.write_json(state_key, state)
                    summary["processed"] += 1
                    summary["reports"].append(store.cache.key(prefix + ".md"))
                summary["companies_checked"] += 1
            except Exception as exc:
                logger.exception("Filing monitor failed for CIK %s", cik)
                summary["errors"].append({"cik": cik, "error": str(exc)})
    logger.info("Monitor summary: %s", json.dumps(summary))
    if summary["errors"]:
        raise RuntimeError("Filing monitor incomplete: " + json.dumps(summary))
    return summary


def handler(event, context):
    """EventBridge Lambda entry point. Configuration comes from env/S3, not event input."""
    logging.getLogger().setLevel(logging.INFO)
    cache = S3Cache.from_env()
    if cache is None:
        raise ValueError("AWS_S3_BUCKET is required for the scheduled monitor")
    store = MonitorStore(cache)
    key = os.getenv("MONITOR_WATCHLIST_KEY", "monitor/watchlist.json")
    try:
        watchlist = store.read_json(key)
        if watchlist is None:
            raise ValueError(f"Upload your watchlist to s3://{cache.bucket}/{cache.key(key)}")
        result = run_monitor(watchlist, store, context=context)
    except Exception:
        try:
            store.write_json("monitor/health.json", {"state": "failure", "completed_at": datetime.now(timezone.utc).isoformat()})
        except Exception:
            logger.exception("Could not persist failed monitor health")
        raise
    store.write_json("monitor/health.json", {"state": "success", "completed_at": datetime.now(timezone.utc).isoformat(),
        "companies_checked": result.get("companies_checked"), "processed": result.get("processed")})
    return result
