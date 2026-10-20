"""Precommitted complementary SEC-disclosure acquisition with union weights.

Preserves V1 source files and its unscored final cohort. Does not fit or predict.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest, uniform
from src.validation.historical_coverage import load_reviewed_registry
from src.validation.local_workflow import local_only
from src.validation.sec_event_inventory import run as inventory
from src.validation import sec_disclosure_data as original

CONFIG = Path('config/sec_disclosure_expansion/cohort_v2.json')
PLAN = CONFIG.with_name('acquisition_v2.json')
ROOT = Path('data/processed/validation/sec_disclosure_v2')
RAW = Path('data/raw/sec_disclosure_v2')
ORIGINAL_ROOT = original.ROOT
ORIGINAL_PLAN = original.PLAN
ORIGINAL_CONFIG = original.CONFIG
TARGET = original.TARGET


def union_probability(*, original_proxy, original_sampling, final_role, extension_sampling):
    if not all(0 < x <= 1 for x in [original_sampling, final_role, extension_sampling]):
        raise ValueError('Invalid inclusion probability')
    if original_proxy:
        if original_sampling != 1:
            raise ValueError('Original proxy must be a certainty sample')
        return final_role
    return original_sampling * final_role + (1 - original_sampling) * extension_sampling


def commit():
    source_inventory = inventory()
    config, oldconfig, oldplan = [json.loads(p.read_text()) for p in (CONFIG, ORIGINAL_CONFIG, ORIGINAL_PLAN)]
    rows = pd.read_csv(ORIGINAL_ROOT / 'annual_frame.csv', dtype={'cik': str})
    frame = rows.loc[rows.frame_year.eq(2024)].sort_values('filed').drop_duplicates('cik')
    original_ids = {c['cik'] for c in oldplan['companies']}
    candidates_path = Path('data/processed/validation/sec_event_inventory_v1/candidates.csv')
    candidates = set(pd.read_csv(candidates_path, dtype={'cik': str}).cik)
    records = []
    for row in frame.loc[~frame.cik.isin(original_ids)].sort_values('cik').itertuples():
        pi = 1. if row.cik in candidates else config['extension_background_probability']
        if uniform(config['seed'] + ':sample', row.cik) >= pi:
            continue
        total = union_probability(original_proxy=False, original_sampling=oldconfig['background_probability'],
            final_role=1 - oldconfig['calibration_probability_other'], extension_sampling=pi)
        records.append({'cik': row.cik, 'ticker': 'CIK' + row.cik, 'company_name': row.name,
            'role': 'final', 'frame_member': True, 'extension_sampling_probability': pi,
            'union_inclusion_probability': total, 'design_weight': 1 / total,
            'sec_inventory_proxy': row.cik in candidates, 'acquisition_phase': 'complementary_v2'})
    manifest = {'config_sha256': digest(CONFIG), 'original_plan_sha256': digest(ORIGINAL_PLAN),
        'original_config_sha256': digest(ORIGINAL_CONFIG), 'annual_frame_sha256': digest(ORIGINAL_ROOT / 'annual_frame.csv'),
        'original_dataset_sha256': digest(ORIGINAL_ROOT / 'dataset.csv'),
        'inventory_report_sha256': digest(candidates_path.with_name('report.json')),
        'inventory_candidates_sha256': source_inventory['candidates_sha256'],
        'implementation_sha256': digest(__file__),
        'dependencies': {str(p): digest(p) for p in [Path(original.__file__), original.PARSER,
            Path('src/ml/financial_history.py'), Path('src/validation/sec_event_inventory.py')]},
        'original_2024_frame_companies': len(frame), 'complement_frame_companies': int((~frame.cik.isin(original_ids)).sum()),
        'companies': records, 'original_selected_companies_excluded': sorted(original_ids),
        'original_final_used_for_selection': False, 'source_review_is_not_independent_human_adjudication': True}
    ROOT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != manifest:
            raise ValueError('Committed complementary cohort changed')
    else:
        with PLAN.open('x') as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


@contextmanager
def acquisition_directory():
    # Reuse the sealed admission logic without editing its source or old outputs.
    previous = original.RAW
    original.RAW = RAW
    try:
        yield
    finally:
        original.RAW = previous


def acquire_one(company, config, cases):
    with acquisition_directory():
        try:
            return original.acquire(company, config, cases)
        except ValueError as exc:
            expected = f'No original 10-K/10-Q USD Assets facts for CIK{company["cik"]}'
            if str(exc) != expected:
                raise
            # Confirm the domain-specific source absence. Other source failures
            # remain errors, and no missing-data exclusion gets an outcome label.
            path = RAW / 'companyfacts' / f'CIK{company["cik"]}.json'
            payload = json.loads(path.read_text())
            facts = payload.get('facts', {}).get('us-gaap', {}).get('Assets', {}).get('units', {}).get('USD', [])
            if str(payload['cik']).zfill(10) != company['cik'] or any(f.get('form') in {'10-K', '10-Q'} for f in facts):
                raise ValueError('Missing-assets exclusion contradicted by source') from exc
            return {'status': 'excluded', 'reason': 'missing_original_usd_assets', 'sources': [
                {'path': str(path), 'sha256': digest(path), 'url': f'https://data.sec.gov/api/xbrl/companyfacts/CIK{company["cik"]}.json'}]}


def combined_rows(extension, inventory_candidates, oldconfig):
    old = pd.read_csv(ORIGINAL_ROOT / 'dataset.csv', dtype={'cik': str})
    old = old.loc[old.role.eq('final')].copy()
    probabilities = []
    for row in old.itertuples():
        proxy = pd.notna(row.proxy_year)
        newpi = 1. if row.cik in inventory_candidates else .1
        probabilities.append(union_probability(original_proxy=proxy,
            original_sampling=row.sampling_probability, final_role=row.role_probability, extension_sampling=newpi))
    old['union_inclusion_probability'] = probabilities
    old['design_weight'] = 1 / old.union_inclusion_probability
    old['acquisition_phase'] = 'original_v1_final'
    result = pd.concat([old, pd.DataFrame(extension)], ignore_index=True)
    if result.cik.duplicated().any():
        raise ValueError('Company overlap across acquisition phases')
    return result


def prepare():
    plan = commit()
    config = json.loads(ORIGINAL_CONFIG.read_text())
    # Admission scope, feature staleness, missingness and follow-up rules stay
    # identical. Extension enrolment is different and explicitly frozen above.
    cases = load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    checkpoints = ROOT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    records, errors, exclusions, proofs = [], [], [], []
    for index, company in enumerate(plan['companies'], 1):
        print(f'SEC complementary acquisition {index}/{len(plan["companies"])} {company["cik"]}', flush=True)
        marker = checkpoints / (company['cik'] + '.json')
        try:
            if marker.exists():
                record = json.loads(marker.read_text())
                if record['plan_sha256'] != digest(PLAN):
                    raise ValueError('Checkpoint plan changed')
                for source in record['sources']:
                    if digest(source['path']) != source['sha256']:
                        raise ValueError('Checkpoint source changed')
            else:
                record = acquire_one(company, config, cases)
                record['plan_sha256'] = digest(PLAN)
                write_json(marker, record)
            proofs.append({'cik': company['cik'], 'sha256': digest(marker)})
            if record['status'] == 'complete':
                records.append(record['row'])
            else:
                exclusions.append({'cik': company['cik'], 'reason': record['reason']})
        except Exception as exc:
            errors.append({'cik': company['cik'], 'error': str(exc), 'label_assigned': False})
    pd.DataFrame(records).to_csv(ROOT / 'extension_dataset.csv', index=False)
    candidates = set(pd.read_csv('data/processed/validation/sec_event_inventory_v1/candidates.csv', dtype={'cik': str}).cik)
    frame = combined_rows(records, candidates, config)
    frame.to_csv(ROOT / 'combined_final_dataset.csv', index=False)
    counts = lambda f: {'rows': len(f), 'positive_companies': int(f.loc[f[TARGET].eq(1), 'cik'].nunique()),
        'negative_companies': int(f.loc[f[TARGET].eq(0), 'cik'].nunique()), 'unknown': int(f[TARGET].isna().sum())}
    report = {'plan_sha256': digest(PLAN), 'selected_extension_companies': len(plan['companies']),
        'extension': counts(pd.DataFrame(records)) if records else {}, 'combined_final': counts(frame),
        'combined_final_sha256': digest(ROOT / 'combined_final_dataset.csv'),
        'extension_dataset_sha256': digest(ROOT / 'extension_dataset.csv'), 'checkpoint_proofs': proofs,
        'exclusions': exclusions, 'errors': errors, 'final_predictions_created': False, 'production_ready': False,
        'limits': 'Known sequential sampling is weighted; source nonresponse/follow-up censoring are not corrected. Automated disclosure outcomes are not independently human-reviewed court outcomes.'}
    write_json(ROOT / 'report.json', report)
    return report


if __name__ == '__main__':
    with local_only():
        print(json.dumps(prepare(), indent=2))
