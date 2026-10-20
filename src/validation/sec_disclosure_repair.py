"""Recover an outcome acquisition blocked only by out-of-scope Form 4 metadata."""
import json
from pathlib import Path
from unittest.mock import patch

from src.data_layer import research_dataset
from src.validation import sec_disclosure_extension as extension
from src.validation import sec_disclosure_data as original
from src.validation.cohort_v3_data import digest
from src.validation.historical_coverage import load_reviewed_registry
from src.data_layer.research_dataset import write_json
from src.validation.local_workflow import local_only

NORMALIZE = research_dataset.submission_rows
LOAD = research_dataset.load_submissions


def outcome_submission_rows(block, cik, ticker, start, end):
    """Validate required forms normally; Form 4 is unrelated to this target."""
    filtered = dict(block, form=['__outcome_out_of_scope__' if f == '4' else f for f in block.get('form', [])])
    return NORMALIZE(filtered, cik, ticker, start, end)


def outcome_load_submissions(*args, **kwargs):
    with patch.object(research_dataset, 'submission_rows', outcome_submission_rows):
        return LOAD(*args, **kwargs)


def run():
    report = json.loads((extension.ROOT / 'report.json').read_text())
    plan = extension.commit()
    config = json.loads(original.CONFIG.read_text())
    cases = load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    changes = []
    for error in report['errors']:
        if error['error'] != 'Invalid SEC document path':
            continue
        company = next(c for c in plan['companies'] if c['cik'] == error['cik'])
        marker = extension.ROOT / 'checkpoints' / (company['cik'] + '.json')
        if marker.exists():
            raise ValueError('Repair never replaces an acquired checkpoint')
        with patch.object(original, 'load_submissions', outcome_load_submissions):
            record = extension.acquire_one(company, config, cases)
        record.update(plan_sha256=digest(extension.PLAN), repair={
            'implementation_sha256': digest(__file__), 'original_error': error,
            'scope': 'Skip Form 4 normalization only; 10-K, 10-Q, 8-K, 8-K/A validation and evidence admission unchanged.'})
        write_json(marker, record)
        changes.append({'cik': company['cik'], 'checkpoint_sha256': digest(marker)})
    write_json(extension.ROOT / 'metadata_repair.json', {'repairs': changes, 'implementation_sha256': digest(__file__),
        'unrelated_form4_is_not_a_model_input': True})
    return extension.prepare()


if __name__ == '__main__':
    with local_only():
        print(json.dumps(run(), indent=2))
