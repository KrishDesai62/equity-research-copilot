"""Expand historical training with the same SEC-disclosure outcome definition.

Old consumed historical test companies are explicitly retired into development
for this new study. All recent calibration/final company IDs remain excluded.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import submission_rows, write_json
from src.ml.financial_history import extract_history
from src.validation.adjudicate_public_events import corroborate
from src.validation.cohort_v3_data import digest, make_snapshots
from src.validation.download import fetch
from src.validation.local_workflow import local_only
from src.validation.sec_disclosure_data import amendments, decide_label, followup_complete, FEATURES, TARGET
from src.validation.historical_coverage import load_reviewed_registry

CONFIG = Path('config/sec_disclosure_expansion/training_v2.json')
PLAN = CONFIG.with_name('training_acquisition_v2.json')
ROOT = Path('data/processed/validation/sec_disclosure_training_v2')
RAW = Path('data/raw/sec_disclosure_training_v2')
OLD = Path('data/processed/validation/cohort_v3')


def commit():
    oldplan = Path('config/cohort_v3_acquisition.json')
    companies = json.loads(oldplan.read_text())['companies']
    reserved = {c['cik'] for path in ['config/sec_disclosure/acquisition_v1.json',
        'config/sec_disclosure_expansion/acquisition_v2.json'] for c in json.loads(Path(path).read_text())['companies']}
    if reserved & {c['cik'] for c in companies}:
        raise ValueError('Historical training overlaps recent reservations')
    manifest = {'config_sha256': digest(CONFIG), 'old_plan_sha256': digest(oldplan),
        'old_source_seal_sha256': digest(OLD / 'source_seal.json'),
        'old_annual_frame_sha256': digest(OLD / 'annual_frame.csv'),
        'reserved_companies': sorted(reserved), 'companies': companies,
        'implementation_sha256': digest(__file__), 'dependencies': {p: digest(p) for p in [
            'src/validation/sec_disclosure_data.py', 'src/validation/adjudicate_public_events.py',
            'src/ml/financial_history.py', 'src/validation/cohort_v3_data.py',
            'src/validation/sec_disclosure_coverage.py']},
        'old_final_retired_into_development': True, 'old_final_predictions_read': False,
        'recent_final_predictions_created': False}
    ROOT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != manifest:
            raise ValueError('Historical training acquisition plan changed')
    else:
        with PLAN.open('x') as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


def history_sources(company, seal):
    marker = OLD / 'company_checkpoints' / (company['cik'] + '.json')
    if digest(marker) != seal['checkpoints'][company['cik']]:
        raise ValueError('Original source checkpoint changed')
    previous = json.loads(marker.read_text())
    if previous['status'] == 'unavailable':
        return None, [], [], {'status': 'excluded', 'reason': 'original_source_unavailable',
            'sources': [{'path': str(marker), 'sha256': digest(marker)}]}
    sources = previous['sources'].copy()
    metadata, name, facts = {}, '', None
    for source in sources:
        path = Path(source['path'])
        if digest(path) != source['sha256'] or seal['raw_sources'].get(str(path)) != source['sha256']:
            raise ValueError('Original raw source changed')
        payload = json.loads(path.read_text())
        if '/companyfacts/' in str(path):
            if str(payload['cik']).zfill(10) != company['cik']:
                raise ValueError('Financial source identity mismatch')
            facts = payload
        elif '/submissions/' in str(path):
            if 'cik' in payload and str(payload['cik']).zfill(10) != company['cik']:
                raise ValueError('Submission identity mismatch')
            name = payload.get('name', name)
            block = payload.get('filings', {}).get('recent', payload)
            for row in submission_rows(block, company['cik'], company['ticker'], '2010-01-01', '2022-07-19'):
                old = metadata.get(row['accession_number'])
                if old is not None and old != row:
                    raise ValueError('Conflicting original submission records')
                metadata[row['accession_number']] = row
    if facts is None:
        raise ValueError('Missing sealed financial source')
    rows = list(metadata.values()) + amendments(sources, company, '2010-01-01', '2022-07-19')
    return (facts, name), rows, sources, None


def acquire_company(company, config, seal, annual, prior):
    payload, metadata, sources, unavailable = history_sources(company, seal)
    if unavailable:
        return unavailable
    facts, name = payload
    history, _, _ = extract_history(facts, company['ticker'], public_filings=metadata)
    history['source_url'] = history.accession_number.map({f['accession_number']: f['source_url'] for f in metadata})
    # Do not infer an unknown or unindexed amendment's absence from metadata.
    from src.validation.sec_disclosure_coverage import review_events
    events, unresolved, reviews, documents = review_events(metadata, company, name,
        earliest_anchor=min(config['anchors']), raw=RAW)
    sources += documents
    company = {**company, 'role': 'training'}
    snapshots, skipped = make_snapshots(history, annual.loc[annual.cik.eq(company['cik'])], company,
        {'train_anchors': config['anchors'], 'max_feature_age_days': config['max_feature_age_days']})
    records = []
    for row in snapshots:
        anchor = row['prediction_date']
        if prior.get(company['cik'], '9999-12-31') <= anchor:
            skipped.append({'cik': company['cik'], 'prediction_date': anchor, 'reason': 'known_prior_event'})
            continue
        assets, cash = row.get('Assets', np.nan), row.get('Cash', np.nan)
        row['accruals_to_assets'] = (row.get('NetIncomeLoss_ttm', np.nan) - row.get('OperatingCashFlow_ttm', np.nan)) / assets if assets > 0 else np.nan
        row['net_debt_proxy_to_assets'] = (row.get('Liabilities', np.nan) - cash) / assets if assets > 0 else np.nan
        row['operating_cash_burn_to_cash'] = max(-row.get('OperatingCashFlow_ttm', np.nan), 0) / cash if cash > 0 else np.nan
        values = pd.Series({f: row.get(f, np.nan) for f in FEATURES}).replace([np.inf, -np.inf], np.nan)
        if values.isna().mean() > config['max_feature_missing_fraction']:
            skipped.append({'cik': company['cik'], 'prediction_date': anchor, 'reason': 'missing_features'})
            continue
        horizon = (pd.Timestamp(anchor) + pd.DateOffset(years=1)).date().isoformat()
        cutoff = (pd.Timestamp(horizon) + pd.Timedelta(days=config['reporting_grace_days'])).date().isoformat()
        assertions = [a for a in events if a['filing_date'] <= cutoff]
        pending = [a for a in unresolved if a['filing_date'] <= cutoff]
        complete = followup_complete(history, row['feature_filing_date'], anchor, horizon, cutoff)
        label, reason = decide_label(assertions, pending, followup=complete, anchor=anchor, horizon=horizon)
        row.update(values.to_dict())
        row.update(**{TARGET: label}, label_reason=reason, horizon_end=horizon,
            label_available_date=cutoff, role='training', original_role_retired=True)
        records.append(row)
    result = {'status': 'complete', 'rows': records, 'exclusions': skipped,
        'sources': sources, 'event_reviews': reviews}
    return json.loads(pd.Series([result]).to_json(orient='values'))[0]


def prepare():
    plan, config = commit(), json.loads(CONFIG.read_text())
    seal = json.loads((OLD / 'source_seal.json').read_text())
    annual = pd.read_csv(OLD / 'annual_frame.csv', dtype={'cik': str, 'filed': str})
    cases = load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    checkpoints = ROOT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    rows, exclusions, errors, proofs = [], [], [], []
    for index, company in enumerate(plan['companies'], 1):
        print(f'Expanded historical SEC training {index}/{len(plan["companies"])} {company["cik"]}', flush=True)
        marker = checkpoints / (company['cik'] + '.json')
        try:
            if marker.exists():
                record = json.loads(marker.read_text())
                if record['plan_sha256'] != digest(PLAN):
                    raise ValueError('Training checkpoint plan changed')
                for source in record['sources']:
                    if digest(source['path']) != source['sha256']:
                        raise ValueError('Training checkpoint source changed')
            else:
                record = acquire_company(company, config, seal, annual, cases)
                record['plan_sha256'] = digest(PLAN)
                write_json(marker, record)
            proofs.append({'cik': company['cik'], 'sha256': digest(marker)})
            if record['status'] == 'complete':
                rows.extend(record['rows'])
                exclusions.extend(record['exclusions'])
            else:
                exclusions.append({'cik': company['cik'], 'reason': record['reason']})
        except Exception as exc:
            errors.append({'cik': company['cik'], 'error': str(exc)})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'dataset.csv', index=False)
    report = {'plan_sha256': digest(PLAN), 'dataset_sha256': digest(ROOT / 'dataset.csv'),
        'rows': len(frame), 'errors': errors, 'exclusions': exclusions, 'checkpoint_proofs': proofs,
        'positive_companies': int(frame.loc[frame[TARGET].eq(1), 'cik'].nunique()) if len(frame) else 0,
        'negative_rows': int(frame[TARGET].eq(0).sum()) if len(frame) else 0,
        'unknown_rows': int(frame[TARGET].isna().sum()) if len(frame) else 0,
        'old_final_retired_into_development': True, 'recent_final_used': False, 'production_ready': False}
    write_json(ROOT / 'report.json', report)
    return report


if __name__ == '__main__':
    with local_only():
        print(json.dumps(prepare(), indent=2))
