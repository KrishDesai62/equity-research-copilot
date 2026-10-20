"""Quarantine and audit a pinned public dataset before permitting model use.

Only the upstream TRAIN file is acquired. Validation/test/live outcomes are not
opened, and upstream zeros never become independently reviewed healthy labels.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.download import fetch
from src.validation.cohort_v3_data import digest
from src.validation.historical_coverage import load_reviewed_registry
from src.validation.local_workflow import local_only

REVISION = 'a12f20990b4540ecc3db0e0294e985b7c81f58e9'
RAW = Path('data/raw/public_dataset_review/openfundex')
OUTPUT = Path('data/processed/validation/openfundex_audit')


def audit_frame(frame, registry):
    required = ['cik', 'adsh', 'filing_date', 'period_end_date', 'bankruptcy_date', 'target_bankrupt_1y']
    if not set(required) <= set(frame):
        raise ValueError('Public dataset lacks required identity/date/target fields')
    frame = frame.copy()
    frame['cik'] = frame.cik.astype('Int64').astype(str).str.zfill(10)
    if frame.cik.str.len().ne(10).any() or not frame.cik.str.isdigit().all():
        raise ValueError('Invalid public dataset company identities')
    filed, period, event = [pd.to_datetime(frame[c], errors='raise') for c in ['filing_date', 'period_end_date', 'bankruptcy_date']]
    target = frame.target_bankrupt_1y
    if not set(target.dropna().unique()) <= {0, 1, False, True}:
        raise ValueError('Invalid public dataset outcome encoding')
    # Upstream defines a 365-day horizon, not this project's calendar-year target.
    expected = event.gt(filed) & event.le(filed + pd.Timedelta(days=365))
    prior = event.notna() & event.le(filed)
    first = registry.dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date
    known = pd.to_datetime(frame.cik.map(first))
    brd_expected = known.gt(filed) & known.le(filed + pd.Timedelta(days=365))
    conflicts = brd_expected & ~target.eq(True).fillna(False)
    result = {'rows': len(frame), 'companies': int(frame.cik.nunique()),
        'filing_start': str(filed.min().date()), 'filing_end': str(filed.max().date()),
        'duplicate_accession_rows': int(frame.duplicated(['cik', 'adsh']).sum()),
        'future_fiscal_period_rows': int(period.gt(filed).sum()),
        'rows_filed_after_stated_training_cutoff': int(filed.gt('2019-12-31').sum()),
        'upstream_positive_rows': int(target.eq(True).sum()), 'upstream_positive_companies': int(frame.loc[target.eq(True), 'cik'].nunique()),
        'upstream_negative_rows': int(target.eq(False).sum()), 'upstream_unknown_rows': int(target.isna().sum()),
        'known_event_inside_filing_horizon_but_label_not_positive': int((expected & ~target.eq(True).fillna(False)).sum()),
        'positive_label_outside_known_filing_horizon': int((target.eq(True) & ~expected).sum()),
        'rows_at_or_after_known_bankruptcy': int(prior.sum()),
        'brd_positive_windows_without_upstream_positive': int(conflicts.sum()),
        'brd_positive_companies_with_conflicts': int(frame.loc[conflicts, 'cik'].nunique()),
        'new_reviewed_negative_labels': 0, 'eligible_for_training': False, 'production_ready': False,
        'outcome_columns_forbidden_as_features': ['is_bankrupt', 'bankruptcy_date', 'bankruptcy_chapter', 'target_*', '*_rank'],
        'limits': ['Publisher labels are not independently reviewed project labels.',
            'An SEC Item 1.03 search does not prove exhaustive court outcome coverage.',
            '365-day upstream horizon differs from the project calendar-year horizon.',
            'Cross-sectional, future-outcome and post-event fields require explicit exclusion.',
            'Temporal file splits do not establish company-disjoint evaluation.',
            'Linked publisher code repository returned HTTP 404 during audit.']}
    return result


def run():
    metadata = json.loads((RAW / 'file_metadata.json').read_text())
    entry = next(e for e in metadata if e['path'] == 'train_clean.parquet')
    path = RAW / 'train_clean.parquet'
    source = fetch(f'https://huggingface.co/datasets/ttchopper/openfundex/resolve/{REVISION}/train_clean.parquet?download=true', path)
    expected = entry.get('lfs', {}).get('oid')
    if not expected or digest(path) != expected:
        raise ValueError('Downloaded public dataset differs from pinned upstream LFS hash')
    frame = pd.read_parquet(path)
    report = audit_frame(frame, load_reviewed_registry())
    report.update(source=source, revision=REVISION, publisher='Luke Danielson / ttchopper', license='CC-BY-4.0',
        source_card='https://huggingface.co/datasets/ttchopper/openfundex',
        dataset_card_sha256=digest(RAW / 'README.md'),
        unopened_upstream_files=['validation_clean.parquet', 'test_clean.parquet', 'recent_clean.parquet'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = frame[['cik', 'company_name', 'bankruptcy_date', 'bankruptcy_chapter']].dropna(subset=['bankruptcy_date']).drop_duplicates()
    candidates['cik'] = candidates.cik.astype('Int64').astype(str).str.zfill(10)
    candidate_dates = pd.to_datetime(candidates.bankruptcy_date)
    candidates = candidates.loc[candidate_dates.between('2023-01-01', '2025-12-31') & candidates.bankruptcy_chapter.isin(['7', '11'])].copy()
    candidates['review_status'] = 'publisher_event_candidate_requires_primary_source_and_first_case_review'
    candidates['eligible_for_training'] = False
    candidates['eligible_for_final_test'] = False
    candidates.to_csv(OUTPUT / 'recent_event_review_queue.csv', index=False)
    report['recent_event_candidate_rows'] = len(candidates)
    report['recent_event_candidate_companies'] = int(candidates.cik.nunique())
    report['event_queue_sha256'] = digest(OUTPUT / 'recent_event_review_queue.csv')
    write_json(OUTPUT / 'report.json', report)
    return report


if __name__ == '__main__':
    with local_only():
        print(json.dumps(run(), indent=2))
