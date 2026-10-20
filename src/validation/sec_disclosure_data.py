"""Auditable SEC-disclosure outcome cohort; never asserts exhaustive court coverage.

Run locally with ``python -m src.validation.sec_disclosure_data``. Acquisition is
resumable. Unknown outcomes and failed acquisitions never become negatives.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import load_submissions, submission_rows, write_json
from src.ml.financial_history import extract_history, BASE_FEATURES, EXTRA_FEATURES
from src.validation.adjudicate_public_events import corroborate
from src.validation.cohort_v3_data import digest, prior_exposures, uniform
from src.validation.download import companyfacts, fetch
from src.validation.historical_coverage import load_reviewed_registry
from src.validation.local_workflow import local_only

CONFIG = Path('config/sec_disclosure/cohort_v1.json')
PLAN = CONFIG.with_name('acquisition_v1.json')
RAW = Path('data/raw/sec_disclosure_v1')
ROOT = Path('data/processed/validation/sec_disclosure_v1')
QUEUE = Path('data/processed/validation/openfundex_audit/recent_event_review_queue.csv')
PARSER = Path('src/validation/adjudicate_public_events.py')
TARGET = 'target_sec_disclosed_petition_1yr'
FEATURES = BASE_FEATURES + EXTRA_FEATURES + ['log_total_assets', 'feature_age_days',
    'accruals_to_assets', 'net_debt_proxy_to_assets', 'operating_cash_burn_to_cash']


def sampling_record(cik, proxy_year, config):
    pi = 1. if proxy_year else config['background_probability']
    if uniform(config['seed'] + ':sample', cik) >= pi:
        return None
    pcal = config.get(f'calibration_probability_for_{proxy_year}_proxy', config['calibration_probability_other'])
    role = 'calibration' if uniform(config['seed'] + ':role', cik) < pcal else 'final'
    role_pi = pcal if role == 'calibration' else 1. - pcal
    if not 0 < pi <= 1 or not 0 < pcal < 1:
        raise ValueError('Every company must have positive sampling and role probabilities')
    return {'role': role, 'sampling_probability': pi, 'role_probability': role_pi,
        'design_weight': 1. / (pi * role_pi), 'proxy_year': proxy_year}


def commit():
    config = json.loads(CONFIG.read_text())
    queue = pd.read_csv(QUEUE, dtype={'cik': str})
    audit = json.loads(QUEUE.with_name('report.json').read_text())
    if digest(QUEUE) != audit['event_queue_sha256']:
        raise ValueError('Publisher queue integrity failure')
    proxy = queue.assign(year=pd.to_datetime(queue.bankruptcy_date).dt.year).groupby('cik').year.min().to_dict()
    excluded = prior_exposures() | {c['cik'] for c in json.loads(Path('config/cohort_v3_acquisition.json').read_text())['companies']}
    blocks, sources = [], []
    for year in (2022, 2024):
        for quarter in range(1, 5):
            path = RAW / 'bulk' / f'{year}q{quarter}.zip'
            with zipfile.ZipFile(path) as archive:
                frame = pd.read_csv(archive.open('sub.txt'), sep='\t', dtype={'cik': str, 'filed': str}, low_memory=False)
            frame = frame.loc[frame.form.eq('10-K'), ['cik', 'name', 'filed', 'sic']].copy()
            frame['cik'] = frame.cik.str.zfill(10)
            frame['filed'] = pd.to_datetime(frame.filed, format='%Y%m%d').dt.strftime('%Y-%m-%d')
            frame['frame_year'] = year
            blocks.append(frame.loc[frame.filed.str.startswith(str(year))])
            sources.append({'path': str(path), 'sha256': digest(path)})
    annual = pd.concat(blocks, ignore_index=True).drop_duplicates()
    sic = pd.to_numeric(annual.sic, errors='coerce')
    annual = annual.loc[sic.notna() & ~sic.between(6000, 6999) & ~annual.cik.isin(excluded)]
    names = annual.sort_values('filed').drop_duplicates('cik').set_index('cik')['name'].to_dict()
    companies = []
    for cik in sorted(names):
        record = sampling_record(cik, int(proxy[cik]) if cik in proxy else None, config)
        if record:
            year = config[record['role'] + '_frame_year']
            companies.append({'cik': cik, 'ticker': 'CIK' + cik, 'company_name': names[cik], **record,
                'frame_member': bool((annual.cik.eq(cik) & annual.frame_year.eq(year)).any())})
    manifest = {'config_sha256': digest(CONFIG), 'proxy_queue_sha256': digest(QUEUE),
        'universe_archives': sources, 'excluded_model_study_companies': sorted(excluded),
        'frame_companies': int(annual.cik.nunique()), 'companies': companies,
        'limits': 'Evidence-review exposure is recorded but is not model-fitting exposure. Proxy-assisted sampling is not outcome-blind; inverse joint weights do not correct coverage censoring/nonresponse.'}
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != manifest:
            raise ValueError('Committed SEC disclosure acquisition plan changed')
    else:
        with PLAN.open('x') as handle:
            json.dump(manifest, handle, indent=2)
    ROOT.mkdir(parents=True, exist_ok=True)
    annual.to_csv(ROOT / 'annual_frame.csv', index=False)
    return manifest


def amendments(sources, company, start, end):
    result = []
    for source in sources:
        if '/submissions/' not in source['url']:
            continue
        payload = json.loads(Path(source['path']).read_text())
        block = payload.get('filings', {}).get('recent', payload)
        forms = block.get('form', [])
        copied = dict(block, form=['8-K' if f == '8-K/A' else '__ignore__' for f in forms])
        for row in submission_rows(copied, company['cik'], company['ticker'], start, end):
            row['filing_type'] = '8-K/A'
            result.append(row)
    return result


def followup_complete(history, baseline_date, anchor, horizon, cutoff):
    """Observable continued reporting; no assertion about unreported court events."""
    required = {'filing_date', 'fiscal_period_end'}
    if not required.issubset(history.columns):
        return False
    future = history.loc[history.filing_date.gt(anchor) & history.filing_date.le(cutoff)].sort_values('filing_date')
    if future.empty or future.fiscal_period_end.max() < horizon or future.fiscal_period_end.nunique() < 4:
        return False
    # Stop at the first report covering the horizon; late later reports don't
    # manufacture a reporting gap after the label was already observable.
    end = future.loc[future.fiscal_period_end.ge(horizon), 'filing_date'].min()
    days = pd.to_datetime([baseline_date] + sorted(set(future.loc[future.filing_date.le(end), 'filing_date'])))
    return bool((days[1:] - days[:-1] <= pd.Timedelta(days=180)).all())


def decide_label(assertions, unresolved, *, followup, anchor, horizon):
    issuer = [a for a in assertions if a['issuer_explicitly_included']]
    if any(a['petition_date'] <= anchor for a in issuer):
        return None, 'prior_issuer_event_censored'
    positives = [a for a in issuer if anchor < a['petition_date'] <= horizon]
    if positives:
        return 1, 'explicit_issuer_petition_statement'
    if unresolved:
        return None, 'unresolved_item_103'
    if not followup:
        return None, 'incomplete_reporting_followup'
    return 0, 'no_qualifying_disclosure_with_complete_reporting_followup'


def acquire(company, config, prior_cases):
    cik = company['cik']
    if not company['frame_member']:
        return {'status': 'excluded', 'reason': 'not_member_of_assigned_year_frame', 'sources': []}
    anchor = config[company['role'] + '_anchor']
    horizon = (pd.Timestamp(anchor) + pd.DateOffset(years=1)).date().isoformat()
    cutoff = (pd.Timestamp(horizon) + pd.Timedelta(days=config['reporting_grace_days'])).date().isoformat()
    if pd.Timestamp(cutoff) >= pd.Timestamp.now().normalize():
        raise ValueError('Outcome horizon and reporting grace have not matured')
    if cik in prior_cases and prior_cases[cik] <= anchor:
        return {'status': 'excluded', 'reason': 'known_prior_registry_bankruptcy', 'sources': []}
    sources = []
    metadata, sec_name = load_submissions(company, '2010-01-01', cutoff, RAW, sources)
    metadata += amendments(sources, company, '2010-01-01', cutoff)
    facts, source = companyfacts(cik, RAW / 'companyfacts')
    sources.append(source)
    if str(facts['cik']).zfill(10) != cik:
        raise ValueError('Companyfacts identity mismatch')
    history, _, diagnostics = extract_history(facts, company['ticker'], public_filings=metadata)
    if history.empty:
        return {'status': 'excluded', 'reason': 'no_reconciled_original_financials', 'sources': sources}
    public = history.loc[history.filing_date.lt(anchor)].sort_values(['filing_date', 'accession_number'])
    if public.empty:
        return {'status': 'excluded', 'reason': 'no_pre_anchor_financials', 'sources': sources}
    row = public.iloc[-1].to_dict()
    age = (pd.Timestamp(anchor) - pd.Timestamp(row['filing_date'])).days
    row['feature_age_days'] = age
    row['log_total_assets'] = float(np.log1p(row['Assets'])) if row['Assets'] > 0 else np.nan
    assets, cash = row.get('Assets', np.nan), row.get('Cash', np.nan)
    row['accruals_to_assets'] = (row.get('NetIncomeLoss_ttm', np.nan) - row.get('OperatingCashFlow_ttm', np.nan)) / assets if assets > 0 else np.nan
    row['net_debt_proxy_to_assets'] = (row.get('Liabilities', np.nan) - cash) / assets if assets > 0 else np.nan
    row['operating_cash_burn_to_cash'] = max(-row.get('OperatingCashFlow_ttm', np.nan), 0) / cash if cash > 0 else np.nan
    values = pd.to_numeric(pd.Series({f: row.get(f, np.nan) for f in FEATURES})).replace([np.inf, -np.inf], np.nan)
    if age > config['max_feature_age_days'] or values.isna().mean() > config['max_feature_missing_fraction']:
        return {'status': 'excluded', 'reason': 'stale_or_missing_financial_features', 'sources': sources}
    assertions, reviews, unresolved = [], [], []
    for filing in sorted(metadata, key=lambda f: (f['filing_date'], f['accession_number'])):
        if filing['filing_type'] not in {'8-K', '8-K/A'}:
            continue
        items = set(re.findall(r'\d+\.\d{2}', filing.get('items') or ''))
        # Review all historical indexed 1.03s for prior events. Unindexed and
        # amendment bodies in the observation window cannot silently be skipped.
        if '1.03' not in items and not (filing['filing_date'] > anchor and (not items or filing['filing_type'] == '8-K/A')):
            continue
        path = RAW / 'documents' / f"{cik}_{filing['accession_number']}.html"
        proof = fetch(filing['source_url'], path)
        sources.append(proof)
        text = path.read_text(errors='replace')
        is_103 = '1.03' in items or bool(re.search(r'Item\s*(?:<[^>]+>|\s|&nbsp;)*1\s*\.\s*03', text, re.I))
        if not is_103 and re.search(r'bankrupt|chapter\s*(?:<[^>]+>|\s)*(?:7|11)\b|receivership', text, re.I):
            unresolved.append(filing['accession_number'])
        if not is_103:
            reviews.append({'accession': filing['accession_number'], 'classification': 'unindexed_body_not_auto_admitted_as_103', 'source': proof})
            continue
        found = corroborate(text, filed=filing['filing_date'], names=[sec_name, company['company_name']])
        for event in found:
            event.update(filing_date=filing['filing_date'], source_url=filing['source_url'], source_sha256=proof['sha256'])
        assertions.extend(found)
        if not found or not all(a['issuer_explicitly_included'] for a in found):
            unresolved.append(filing['accession_number'])
        reviews.append({'accession': filing['accession_number'], 'assertions': found, 'source': proof})
    complete = followup_complete(history, row['filing_date'], anchor, horizon, cutoff)
    label, reason = decide_label(assertions, unresolved, followup=complete, anchor=anchor, horizon=horizon)
    row.update(company, **values.to_dict(), prediction_date=anchor, horizon_end=horizon,
        label_available_date=cutoff, target_definition=config['target_definition'],
        **{TARGET: label}, label_reason=reason, source_filing_date=row['filing_date'],
        unit_id=f'{cik}@{anchor}', reporting_followup_complete=complete)
    record = {'status': 'complete', 'row': row, 'sources': sources, 'reviews': reviews,
        'unresolved_accessions': unresolved, 'diagnostics': diagnostics,
        'metadata_filings': len(metadata), 'reporting_followup_complete': complete}
    # Convert NaN to JSON null; numeric features are imputed using training only.
    return json.loads(pd.Series([record]).to_json(orient='values'))[0]


def prepare():
    plan, config = commit(), json.loads(CONFIG.read_text())
    checkpoints = ROOT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    prior = load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    rows, errors, exclusions, proofs = [], [], [], []
    for index, company in enumerate(plan['companies'], 1):
        cik = company['cik']
        print(f'SEC disclosure acquisition {index}/{len(plan["companies"])} {cik}', flush=True)
        marker = checkpoints / f'{cik}.json'
        try:
            if marker.exists():
                record = json.loads(marker.read_text())
                for source in record['sources']:
                    if digest(source['path']) != source['sha256']:
                        raise ValueError('Acquired source changed')
            else:
                record = acquire(company, config, prior)
                record.update(config_sha256=digest(CONFIG), implementation_sha256=digest(__file__), parser_sha256=digest(PARSER))
                write_json(marker, record)
            if record['config_sha256'] != digest(CONFIG) or record['implementation_sha256'] != digest(__file__) or record['parser_sha256'] != digest(PARSER):
                raise ValueError('Checkpoint implementation/config changed; explicit version migration required')
            proofs.append({'cik': cik, 'checkpoint_sha256': digest(marker)})
            if record['status'] == 'complete':
                rows.append(record['row'])
            else:
                exclusions.append({'cik': cik, 'reason': record['reason']})
        except Exception as exc:
            # A 404 is missing public data, never a negative or a replacement.
            errors.append({'cik': cik, 'error': str(exc), 'not_a_negative': True})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(ROOT / 'dataset.csv', index=False)
    report = {'plan_sha256': digest(PLAN), 'config_sha256': digest(CONFIG),
        'implementation_sha256': digest(__file__), 'selected_companies': len(plan['companies']),
        'completed_rows': len(frame), 'errors': errors, 'exclusions': exclusions,
        'checkpoint_proofs': proofs, 'target': TARGET, 'production_ready': False,
        'scope': 'Automated SEC-disclosure coverage only; not exhaustive court outcomes or human-reviewed negatives.',
        'limits': 'Proxy-assisted sampling and follow-up exclusions can cause spectrum/selection bias. Design weights do not repair censoring. No model accuracy claim follows from acquisition.'}
    if not frame.empty:
        report['dataset_sha256'] = digest(ROOT / 'dataset.csv')
        report['cohorts'] = {role: {'rows': len(f), 'positive_companies': int(f.loc[f[TARGET].eq(1), 'cik'].nunique()),
            'negative_companies': int(f.loc[f[TARGET].eq(0), 'cik'].nunique()), 'unknown': int(f[TARGET].isna().sum())}
            for role, f in frame.groupby('role')}
    write_json(ROOT / 'report.json', report)
    return report


if __name__ == '__main__':
    with local_only():
        print(json.dumps(prepare(), indent=2))
