"""Point-in-time filing-event features from sealed public submissions metadata."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.sec_disclosure_admission_v3 import ROOT as ADMISSION, run as verify_admission

CONFIG = Path('config/sec_disclosure_expansion/filing_signals_v4.json')
ROOT = Path('data/processed/validation/sec_filing_signals_v4')
FEATURES = ['late_report_notices_365d', 'material_default_items_365d', 'material_default_items_90d',
    'nonreliance_items_365d', 'executive_change_items_90d', 'current_reports_90d', 'days_since_annual_report']


def read_metadata(cik, marker):
    record = json.loads(Path(marker).read_text())
    rows, proofs = {}, []
    for source in record['sources']:
        if '/submissions/' not in source['path']:
            continue
        if digest(source['path']) != source['sha256']:
            raise ValueError('Filing-signal metadata changed')
        payload = json.loads(Path(source['path']).read_text())
        if 'cik' in payload and str(payload['cik']).zfill(10) != cik:
            raise ValueError('Filing-signal issuer mismatch')
        block = payload.get('filings', {}).get('recent', payload)
        n = len(block.get('accessionNumber', []))
        if any(len(block.get(k, [])) != n for k in ['form', 'filingDate']):
            raise ValueError('Malformed filing-signal metadata')
        items = block.get('items', [''] * n)
        if len(items) != n:
            items = [''] * n
        for i in range(n):
            accession = block['accessionNumber'][i]
            if not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accession):
                raise ValueError('Invalid filing-signal accession')
            row = {'accession': accession, 'form': block['form'][i], 'date': block['filingDate'][i], 'items': items[i] or ''}
            if accession in rows and rows[accession] != row:
                raise ValueError('Conflicting filing-signal metadata')
            rows[accession] = row
        proofs.append(source)
    if not proofs:
        raise ValueError('No public filing-signal coverage')
    return list(rows.values()), proofs


def signals_at(rows, anchor):
    public = [r for r in rows if r['date'] < anchor]
    anchor_date = pd.Timestamp(anchor)
    windows = {days: [r for r in public if pd.Timestamp(r['date']) >= anchor_date - pd.Timedelta(days=days)] for days in [90, 365]}
    reports = {days: [r for r in group if r['form'] in {'8-K', '8-K/A'}] for days, group in windows.items()}
    def item_count(item, days):
        group = reports[days]
        if any(not r['items'].strip() for r in group):
            return np.nan
        return sum(item in set(re.findall(r'\d+\.\d{2}', r['items'])) for r in group)
    annuals = [r['date'] for r in public if r['form'] == '10-K']
    return {'late_report_notices_365d': sum(r['form'] in {'NT 10-K', 'NT 10-Q'} for r in windows[365]),
        'material_default_items_365d': item_count('2.04', 365), 'material_default_items_90d': item_count('2.04', 90),
        'nonreliance_items_365d': item_count('4.02', 365), 'executive_change_items_90d': item_count('5.02', 90),
        'current_reports_90d': len(reports[90]),
        'days_since_annual_report': (anchor_date - pd.Timestamp(max(annuals))).days if annuals else np.nan,
        'filing_signals_latest_public_date': max((r['date'] for r in public), default=None)}


def run():
    verify_admission()
    ROOT.mkdir(parents=True, exist_ok=True)
    source_paths = {r: ADMISSION / (r + '.csv') for r in ['training', 'calibration']}
    source_hashes = {r: digest(p) for r, p in source_paths.items()}
    saved_path = ROOT / 'report.json'
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        if saved['source_hashes'] != source_hashes or saved['config_sha256'] != digest(CONFIG) or saved['implementation_sha256'] != digest(__file__):
            raise ValueError('Frozen filing-signal extraction changed')
        for role, expected in saved['output_hashes'].items():
            if digest(ROOT / (role + '.csv')) != expected:
                raise ValueError('Frozen filing-signal features changed')
        for source in saved['sources']:
            if digest(source['path']) != source['sha256']:
                raise ValueError('Frozen filing-signal source changed')
        return saved
    checkpoints = json.loads(Path('data/processed/validation/sec_disclosure_admission_v2/report.json').read_text())['checkpoint_inputs']
    cached, all_sources, counts, outputs = {}, {}, {}, {}
    for role, path in source_paths.items():
        frame = pd.read_csv(path, dtype={'cik': str})
        additions = []
        for row in frame.itertuples():
            if row.cik not in cached:
                cached[row.cik], sources = read_metadata(row.cik, checkpoints[row.cik]['path'])
                for source in sources:
                    all_sources[source['path']] = source
            values = signals_at(cached[row.cik], row.prediction_date)
            if values['filing_signals_latest_public_date'] and values['filing_signals_latest_public_date'] >= row.prediction_date:
                raise ValueError('Future filing signal leaked into prediction')
            additions.append(values)
        result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(additions)], axis=1)
        result.to_csv(ROOT / (role + '.csv'), index=False)
        counts[role] = {feature: int(result[feature].notna().sum()) for feature in FEATURES}
        outputs[role] = digest(ROOT / (role + '.csv'))
    report = {'config_sha256': digest(CONFIG), 'implementation_sha256': digest(__file__),
        'source_hashes': source_hashes, 'output_hashes': outputs, 'coverage': counts,
        'sources': list(all_sources.values()), 'features': FEATURES, 'final_feature_rows_read': False,
        'final_predictions_created': False, 'production_ready': False}
    write_json(saved_path, report)
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
