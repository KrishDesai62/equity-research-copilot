"""Resolve expected financial-data absence without changing outcome labels.

The initial acquisition log is preserved. Only an exact, independently checked
lack of original USD Assets facts qualifies; network/parser errors stay blocking.
"""
import json
from pathlib import Path

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest


def resolve_missing_financials(report, raw, root):
    exclusions = []
    for error in report['errors']:
        cik = error['cik']
        if error['error'] != f'No original 10-K/10-Q USD Assets facts for CIK{cik}':
            raise ValueError('Resolve acquisition errors before fitting; failures cannot be negatives')
        path = Path(raw) / 'companyfacts' / f'CIK{cik}.json'
        payload = json.loads(path.read_text())
        if str(payload['cik']).zfill(10) != cik:
            raise ValueError('Availability evidence identity mismatch')
        values = payload.get('facts', {}).get('us-gaap', {}).get('Assets', {}).get('units', {}).get('USD', [])
        if any(f.get('form') in {'10-K', '10-Q'} for f in values):
            raise ValueError('Financial availability exclusion contradicted by source')
        exclusions.append({'cik': cik, 'classification': 'ineligible_missing_original_usd_assets',
            'no_outcome_label_assigned': True, 'source': {'path': str(path), 'sha256': digest(path),
                'url': f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'}})
    resolution = {'acquisition_report_sha256': digest(Path(root) / 'report.json'),
        'implementation_sha256': digest(__file__), 'resolved_financial_exclusions': exclusions,
        'unresolved_acquisition_errors': 0, 'outcome_labels_changed': 0}
    path = Path(root) / 'availability_resolution.json'
    if path.exists() and json.loads(path.read_text()) != resolution:
        raise ValueError('Financial availability resolution changed')
    if not path.exists():
        write_json(path, resolution)
    return resolution
