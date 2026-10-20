"""Cross-check publisher event candidates against primary SEC Item 1.03 filings.

Machine-extracted petition dates remain review candidates. No negatives, reviewed
first-case labels, model fitting or held-out scoring are created by this command.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import load_submissions, write_json
from src.validation.download import fetch
from src.validation.local_workflow import local_only
from src.validation.cohort_v3_data import digest
from src.validation.historical_coverage import load_reviewed_registry
from src.validation.future_cohort import exposed_companies

ROOT = Path('data/processed/validation/public_event_checks')
RAW = Path('data/raw/public_event_checks')
QUEUE = Path('data/processed/validation/openfundex_audit/recent_event_review_queue.csv')
def petition_dates(text, filed):
    # Use the same date/action binding as the conservative source adjudicator.
    from src.validation.adjudicate_public_events import corroborate
    return sorted({a['petition_date'] for a in corroborate(text, filed=filed, names=[])})


def refresh_candidates(record):
    for filing in record['primary_item_103_filings']:
        filing['petition_date_candidates'] = petition_dates(
            Path(filing['source']['path']).read_text(errors='replace'), filing['filing_date'])
    extracted = sorted({day for filing in record['primary_item_103_filings'] for day in filing['petition_date_candidates']})
    record.update(petition_date_candidates=extracted,
        date_disagrees_with_publisher=len(extracted) == 1 and extracted[0] != record['publisher_date'],
        status='primary_petition_date_candidate' if len(extracted) == 1 else 'manual_review_required')
    return record


def run():
    queue_manifest = json.loads(QUEUE.with_name('report.json').read_text())
    if digest(QUEUE) != queue_manifest['event_queue_sha256']:
        raise ValueError('Publisher event queue changed after audit')
    queue = pd.read_csv(QUEUE, dtype={'cik': str})
    input_alias_rows = len(queue)
    queue = queue.drop_duplicates(['cik', 'bankruptcy_date', 'bankruptcy_chapter'])
    ROOT.mkdir(parents=True, exist_ok=True)
    checkpoints = ROOT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    records, errors = [], []
    for index, row in enumerate(queue.to_dict('records'), 1):
        cik, supplied = row['cik'], str(row['bankruptcy_date'])
        marker = checkpoints / f'{cik}_{supplied}.json'
        print(f"Primary event check {index}/{len(queue)} {cik}", flush=True)
        if marker.exists():
            record = json.loads(marker.read_text())
            for source in record['source_metadata'] + [m['source'] for m in record['primary_item_103_filings']]:
                if digest(source['path']) != source['sha256']:
                    raise ValueError('Cached primary event evidence changed')
            record = refresh_candidates(record)
            write_json(marker, record)
            records.append(record)
            continue
        try:
            center = pd.Timestamp(supplied)
            start, end = [(center + pd.Timedelta(days=d)).date().isoformat() for d in (-7, 7)]
            sources = []
            filings, sec_name = load_submissions({'cik': cik, 'ticker': f'CIK{cik}'}, start, end, RAW, sources)
            matches = []
            for filing in filings:
                if filing['filing_type'] != '8-K' or '1.03' not in filing['items']:
                    continue
                path = RAW / 'documents' / f"{cik}_{filing['accession_number']}.html"
                source = fetch(filing['source_url'], path)
                candidates = petition_dates(path.read_text(), filing['filing_date'])
                matches.append({'source_url': filing['source_url'], 'accession_number': filing['accession_number'],
                    'filing_date': filing['filing_date'], 'source': source, 'petition_date_candidates': candidates})
            extracted = sorted({day for match in matches for day in match['petition_date_candidates']})
            record = {'cik': cik, 'publisher_company_name': row['company_name'], 'sec_company_name': sec_name,
                'publisher_date': supplied, 'primary_item_103_filings': matches, 'petition_date_candidates': extracted,
                'date_disagrees_with_publisher': len(extracted) == 1 and extracted[0] != supplied,
                'status': 'primary_petition_date_candidate' if len(extracted) == 1 else 'manual_review_required',
                'eligible_for_training': False, 'eligible_for_final_test': False, 'source_metadata': sources}
            write_json(marker, record)
            records.append(record)
        except Exception as exc:
            errors.append({'cik': cik, 'publisher_date': supplied, 'error': str(exc)})
    registry = load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    exposed = exposed_companies()
    index = []
    for record in records:
        candidates = record['petition_date_candidates']
        day = candidates[0] if len(candidates) == 1 else None
        prior = registry.get(record['cik'])
        index.append({'cik': record['cik'], 'publisher_company_name': record['publisher_company_name'],
            'sec_company_name': record['sec_company_name'], 'publisher_date': record['publisher_date'],
            'petition_date_candidate': day, 'status': record['status'],
            'primary_urls': ';'.join(m['source_url'] for m in record['primary_item_103_filings']),
            'earlier_brd_case': prior if prior and prior < (day or record['publisher_date']) else None,
            'previously_exposed_company': record['cik'] in exposed,
            'eligible_for_training': False, 'eligible_for_final_test': False})
    pd.DataFrame(index).to_csv(ROOT / 'review_index.csv', index=False)
    report = {'input_alias_rows': input_alias_rows, 'requested_event_rows': len(queue), 'completed_event_rows': len(records),
        'events_with_primary_item_103': sum(bool(r['primary_item_103_filings']) for r in records),
        'events_with_single_petition_date_candidate': sum(r['status'] == 'primary_petition_date_candidate' for r in records),
        'events_with_publisher_date_disagreement': sum(r['date_disagrees_with_publisher'] for r in records),
        'errors': errors, 'new_reviewed_negative_labels': 0, 'production_ready': False,
        'queue_sha256': digest(QUEUE), 'review_index_sha256': digest(ROOT / 'review_index.csv'),
        'review_index_path': str(ROOT / 'review_index.csv'),
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'note': 'Primary source matching and date extraction are review aids, not completed legal-entity/first-case adjudication. Known-positive selection is development evidence only.'}
    write_json(ROOT / 'report.json', report)
    return report


if __name__ == '__main__':
    with local_only():
        result = run()
        print(json.dumps(result, indent=2))
        if result['errors']:
            raise SystemExit(2)
