"""Acquire primary SEC evidence for known-positive development cases only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from src.data_layer.filing_changes import normalize_filing
from src.data_layer.research_dataset import write_json
from src.validation.download import fetch
from src.validation.local_workflow import local_only


def run(config=Path('config/evidence/recent_events.json'), output=Path('data/processed/validation/recent_event_evidence.json')):
    entries = json.loads(Path(config).read_text())['companies']
    records = []
    for entry in entries:
        cik = entry['cik']
        if f'/data/{int(cik)}/' not in entry['source_url'] or not entry['source_url'].startswith('https://www.sec.gov/Archives/'):
            raise ValueError('Event evidence must reference the exact issuer SEC archive')
        path = Path('data/raw/recent_event_evidence') / f'{cik}.html'
        source = fetch(entry['source_url'], path)
        text = re.sub(r'\s+', ' ', normalize_filing(path.read_text()).replace('\xa0', ' ')).lower()
        if entry['date_text'].lower() not in text or entry['company_name'].lower() not in text or not re.search(r'chapter\s+11', text):
            raise ValueError(f'Primary evidence checks failed for {cik}; do not create a label')
        records.append({**entry, 'source': source, 'source_event_checks_passed': True,
            'retrieved_on': datetime.now(timezone.utc).date().isoformat(),
            'review_status': 'Primary filing confirms event; first-case/legal-group review and prediction-unit pairing remain pending',
            'eligible_for_training': False, 'eligible_for_final_test': False})
    report = {'known_positive_event_records': records, 'new_negative_labels': 0, 'production_ready': False,
              'note': 'Case-enriched development evidence only. No first-case labels or untouched evaluation claimed.'}
    write_json(Path(output), report)
    return report


if __name__ == '__main__':
    with local_only():
        print(json.dumps(run(), indent=2))
