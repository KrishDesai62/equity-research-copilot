"""Point-in-time disclosure-change and financial features for future development."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json

CHANGE_FEATURES = ['risk_text_change', 'mda_text_change', 'risk_added_distress_mentions', 'mda_added_distress_mentions']
FINANCIAL_FEATURES = ['accruals_to_assets', 'net_debt_proxy_to_assets', 'operating_cash_burn_to_cash']
TERMS = ['going concern', 'substantial doubt', 'covenant', 'default', 'restructuring']


def report_features(report):
    previous, current = report['previous'], report['current']
    if (previous['cik'] != current['cik'] or current['cik'] != report['cik'] or
            previous['filing_type'] != current['filing_type'] or current['filing_type'] not in {'10-K', '10-Q'} or
            pd.Timestamp(previous['filing_date']) >= pd.Timestamp(current['filing_date'])):
        raise ValueError('Disclosure pair identity, form or dates do not match')
    values = {key: np.nan for key in CHANGE_FEATURES}
    for section in report['sections']:
        prefix = {'risk_factors': 'risk', 'management_discussion': 'mda'}.get(section['section'])
        if prefix is None or section['status'] == 'unavailable':
            continue
        similarity = float(section['text_similarity'])
        if not np.isfinite(similarity) or not 0 <= similarity <= 1:
            raise ValueError('Invalid disclosure similarity')
        values[prefix + '_text_change'] = 1 - similarity
        delta = 0
        for block in section['changes']:
            before, after = (str(block.get(k) or '').lower() for k in ['previous_text', 'current_text'])
            delta += sum(after.count(term) - before.count(term) for term in TERMS)
        values[prefix + '_added_distress_mentions'] = max(0, delta)
    return {'cik': current['cik'], 'available_date': current['filing_date'],
            'change_source_url': current['source_url'], 'change_accession': current['accession_number'], **values}


def add_features(frame, reports):
    frame = frame.copy()
    asset = pd.to_numeric(frame.Assets, errors='raise').where(lambda s: s > 0)
    cash = pd.to_numeric(frame.Cash, errors='raise')
    frame['accruals_to_assets'] = (frame.NetIncomeLoss_ttm - frame.OperatingCashFlow_ttm) / asset
    frame['net_debt_proxy_to_assets'] = (frame.Liabilities - cash) / asset
    frame['operating_cash_burn_to_cash'] = (-frame.OperatingCashFlow_ttm).clip(lower=0) / cash.where(cash > 0)
    # Liabilities minus cash is explicitly a proxy, not debt reconstructed from securities.
    features = [report_features(r) for r in reports]
    joined = []
    for row in frame.to_dict('records'):
        prediction = row.get('prediction_date', row['filing_date'])
        feature_date = row.get('feature_filing_date', row['filing_date'])
        if pd.Timestamp(row['latest_input_filing_date']) > pd.Timestamp(feature_date) or pd.Timestamp(feature_date) > pd.Timestamp(prediction):
            raise ValueError('Financial features include future public inputs')
        candidates = [r for r in features if r['cik'] == row['cik'] and pd.Timestamp(r['available_date']) < pd.Timestamp(prediction)
                      and (pd.Timestamp(prediction) - pd.Timestamp(r['available_date'])).days <= 550]
        current = max(candidates, key=lambda r: (r['available_date'], r['change_accession']), default=None)
        row.update({key: np.nan for key in CHANGE_FEATURES})
        row.update(change_available_date=None, change_source_url=None, change_accession=None)
        if current:
            row.update({key: current[key] for key in CHANGE_FEATURES})
            row.update(change_available_date=current['available_date'], change_source_url=current['change_source_url'], change_accession=current['change_accession'])
        joined.append(row)
    return pd.DataFrame(joined).replace([np.inf, -np.inf], np.nan)


def build(root=Path('data/processed/research'), output=Path('data/processed/validation/disclosure_development_v2')):
    root, output = Path(root), Path(output)
    if output.exists():
        saved = json.loads((output / 'manifest.json').read_text())
        if sha256((output / 'features.csv').read_bytes()).hexdigest() != saved['features_sha256']:
            raise ValueError('Saved development features changed')
        for name, expected in saved['source_sha256'].items():
            if sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Development sources changed; choose a new version output')
        return saved
    pairs = [json.loads(line) for line in (root / 'filing_pairs.jsonl').read_text().splitlines() if line.strip()]
    paths = [Path(pair['report_path']) for pair in pairs]
    frame = pd.read_csv(root / 'financial_features.csv', dtype={'cik': str})
    result = add_features(frame, [json.loads(path.read_text()) for path in paths])
    # This is deliberately a separate development artifact; sealed V3 is untouched.
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / 'features.csv', index=False)
    sources = [root / 'financial_features.csv', *paths]
    report = {'rows': len(result), 'rows_with_disclosure_features': int(result[CHANGE_FEATURES].notna().any(axis=1).sum()),
              'new_features': FINANCIAL_FEATURES + CHANGE_FEATURES,
              'source_sha256': {str(p): sha256(p.read_bytes()).hexdigest() for p in sources},
              'features_sha256': sha256((output / 'features.csv').read_bytes()).hexdigest(),
              'purpose': 'Development feature extraction only; no new labels, final scoring, model fitting or measured improvement.',
              'production_ready': False}
    write_json(output / 'manifest.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/processed/validation/disclosure_development_v2'))
    print(json.dumps(build(output=parser.parse_args().output), indent=2))
