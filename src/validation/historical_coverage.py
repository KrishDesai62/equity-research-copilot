"""Source-reviewed negatives restricted to the BRD's large-public-company scope.

An arbitrary absent CIK is never sufficient. Each negative needs a complete
registry interval and verified, sufficiently large annual filings covering its
entire horizon. Evidence used for outcome review is never a model feature.
"""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.data_layer.lopucki import parse_archive

START, END = pd.Timestamp("2010-01-01"), pd.Timestamp("2022-12-31")
# The largest threshold in the actual 2010–2022 protocol table is $355,163,835.
# A fixed $400m floor conservatively exceeds every year-specific threshold.
ASSET_FLOOR = 400_000_000


def apply_historical_coverage(frame, cases, filings, *, observation_end=None, positive_followup="event"):
    if positive_followup not in {"event", "full_horizon"}:
        raise ValueError("Unknown positive follow-up policy")
    source_end = min(pd.Timestamp(observation_end), END) if observation_end else END
    if pd.isna(source_end):
        raise ValueError("Source observation endpoint must be a valid date")
    dates = pd.to_datetime(cases.filing_date)
    covered_cases = cases.loc[dates.between(START, END)].copy()
    if covered_cases.cik.isna().any():
        raise ValueError("Registry identities incomplete in the proposed coverage interval")
    if covered_cases.empty:
        raise ValueError("No registry cases in the coverage interval")
    result = frame.copy()
    result["label_available_date_1yr"] = pd.to_datetime(result.label_available_date_1yr, errors="raise")
    evidence = []
    result["historical_coverage_eligible"] = False
    result["coverage_review_method"] = "unreviewed_or_outside_scope"
    for cik, group in result.groupby("cik"):
        annual = group.loc[group.filing_type.eq("10-K")].sort_values("filing_date")
        annual_index = filings.loc[filings.cik.eq(cik) & filings.filing_type.eq("10-K")]
        company_events = pd.to_datetime(covered_cases.loc[covered_cases.cik.eq(cik), "filing_date"])
        for i, row in group.iterrows():
            anchor = pd.Timestamp(row.filing_date)
            horizon = anchor + pd.DateOffset(years=1)
            if anchor < START or horizon > END or bool(row.get("bankruptcy_label_conflict", False)):
                continue
            # A registry-confirmed positive is observed at its event date. Do
            # not demand post-bankruptcy annual filings to verify that event.
            # Negatives still require coverage through the entire horizon.
            review_end = horizon
            positive = pd.notna(row.target_distressed_1yr) and row.target_distressed_1yr == 1
            if positive and positive_followup == "event":
                event_date = pd.Timestamp(row.label_available_date_1yr)
                if pd.isna(event_date) or not anchor < event_date <= horizon or not company_events.eq(event_date).any():
                    continue
                review_end = event_date
            if review_end > source_end:
                continue
            annual_dates = pd.to_datetime(annual.filing_date)
            prior = annual.loc[annual_dates.le(anchor)]
            if prior.empty:
                continue
            last = prior.iloc[-1]
            # The last pre-anchor annual must establish public-company status
            # through the horizon, and every replacement annual must stay large.
            if pd.Timestamp(last.fiscal_period_end) < review_end - pd.DateOffset(years=3):
                continue
            stop_at_event = positive and positive_followup == "event"
            upper = annual_dates.lt(review_end) if stop_at_event else annual_dates.le(review_end)
            relevant = pd.concat([prior.tail(1), annual.loc[annual_dates.gt(anchor) & upper]])
            index_dates = pd.to_datetime(annual_index.filing_date)
            if stop_at_event and index_dates.eq(review_end).any():
                continue  # Day-only dates cannot order a filing and bankruptcy on the same day.
            index_upper = index_dates.lt(review_end) if stop_at_event else index_dates.le(review_end)
            expected = annual_index.loc[index_dates.ge(pd.Timestamp(last.filing_date)) & index_upper]
            if not set(expected.accession_number) <= set(relevant.accession_number):
                continue
            if relevant.Assets.isna().any() or not relevant.Assets.ge(ASSET_FLOOR).all():
                continue
            in_window = company_events.gt(anchor) & company_events.le(horizon)
            before = company_events.le(anchor).any()
            if before:
                continue
            result.loc[i, "historical_coverage_eligible"] = True
            result.loc[i, "coverage_review_method"] = "BRD complete-identity interval plus annual size/public-status evidence"
            # Any Chapter 7/other registry event prevents a 'healthy' label too.
            if not in_window.any() and pd.isna(row.target_distressed_1yr):
                result.loc[i, "target_distressed_1yr"] = 0
                result.loc[i, "label_available_date_1yr"] = horizon
            evidence.append({"cik": cik, "accession_number": row.accession_number,
                "filing_date": row.filing_date, "horizon_end": horizon.date().isoformat(),
                "review_end": review_end.date().isoformat(),
                "coverage_policy_version": 2 if positive_followup == "event" else 1,
                "annual_accessions": json.dumps(relevant.accession_number.tolist()),
                "minimum_annual_assets_usd": float(relevant.Assets.min()),
                "registry_events_in_window": int(in_window.sum()),
                "source_url": "https://lopucki.law.ufl.edu/documentation/BRD_protocols.pdf",
                "reviewed_by": "source_and_scope_review_2026-10-17"})
    # Readiness uses one-year targets; prevent old flags from hiding new reviewed
    # negatives. Conflicts/post-event exclusions remain unlabeled above.
    result["valid_for_training"] = result.target_distressed_1yr.notna()
    return result, pd.DataFrame(evidence)


def load_reviewed_registry():
    archive = Path("data/raw/lopucki/cases.download")
    report = json.loads(Path("data/processed/lopucki/report.json").read_text())
    if sha256(archive.read_bytes()).hexdigest() != report["archive_sha256"]:
        raise ValueError("BRD source changed; repeat source review before generating negatives")
    cases, _ = parse_archive(archive)
    return cases
