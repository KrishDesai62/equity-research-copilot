"""Acquire precommitted historical SEC training complement, preserving all tests."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from src.data_layer import research_dataset
from src.data_layer.research_dataset import write_json
from src.ml.financial_history import extract_history
from src.validation.cohort_v3_data import digest, uniform
from src.validation.historical_sec_inventory import run as inventory
from src.validation.historical_coverage import load_reviewed_registry
from src.validation.local_workflow import local_only
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.sec_disclosure_data import amendments, TARGET
from src.validation.sec_disclosure_repair import outcome_load_submissions
from src.validation.sec_disclosure_admission_v3 import corroborate_v3
from src.validation import sec_disclosure_training as training
from src.validation import sec_disclosure_coverage as coverage

CONFIG = Path('config/development_20261019/training_expansion.json')
PLAN = CONFIG.with_name('training_acquisition.json')
RAW = Path('data/raw/historical_training_20261019')
ROOT = Path('data/processed/validation/historical_training_20261019')
BASE = Path('data/processed/validation/sec_disclosure_admission_v3/training.csv')


def union_probability(original_pi, extension_pi):
    if not 0 < original_pi <= 1 or not 0 <= extension_pi <= 1:
        raise ValueError('Invalid historical sampling probabilities')
    return original_pi + (1 - original_pi) * extension_pi


def commit():
    discovery = inventory()
    config = json.loads(CONFIG.read_text())
    old = json.loads(Path('config/cohort_v3_acquisition.json').read_text())
    recent = {c['cik'] for p in ['config/sec_disclosure/acquisition_v1.json', 'config/sec_disclosure_expansion/acquisition_v2.json']
        for c in json.loads(Path(p).read_text())['companies']}
    annual_path = Path('data/processed/validation/cohort_v3/annual_frame.csv')
    annual = pd.read_csv(annual_path, dtype={'cik': str})
    sic = pd.to_numeric(annual.sic, errors='coerce')
    frame = annual.loc[annual.filed.between('2014-01-01','2020-12-30') & sic.notna() & ~sic.between(6000,6999)].copy()
    frame = frame.loc[~frame.cik.isin(recent | set(old['excluded_companies']))]
    frame = frame.sort_values('filed').drop_duplicates('cik', keep='last')
    original_ids = {c['cik'] for c in old['companies']}
    proxies = set(pd.read_csv('data/processed/validation/historical_sec_inventory_20261019/candidates.csv', dtype={'cik':str}).cik)
    records = []
    for row in frame.loc[~frame.cik.isin(original_ids)].sort_values('cik').itertuples():
        probability = 1. if row.cik in proxies else config['background_probability']
        if uniform(config['seed'] + ':sample', row.cik) >= probability:
            continue
        total = union_probability(.1, probability)
        records.append({'cik': row.cik, 'ticker':'CIK'+row.cik, 'company_name':row.name,
            'role':'training', 'sampling_stratum':'sec_inventory_proxy' if row.cik in proxies else 'background',
            'extension_probability':probability, 'inclusion_probability':total, 'design_weight':1/total})
    result = {'config_sha256':digest(CONFIG), 'inventory_candidates_sha256':discovery['candidates_sha256'],
        'old_plan_sha256':digest('config/cohort_v3_acquisition.json'), 'annual_frame_sha256':digest(annual_path),
        'old_training_sha256':digest(BASE), 'implementation_sha256':digest(__file__),
        'dependencies':{p:digest(p) for p in ['src/validation/sec_disclosure_training.py',
            'src/validation/sec_disclosure_coverage.py','src/validation/sec_disclosure_admission_v3.py',
            'src/validation/public_source_fetch.py','src/ml/financial_history.py']},
        'recent_reserved_companies':sorted(recent), 'extension_frame':sorted(set(frame.cik)),
        'sec_inventory_proxies':sorted(proxies), 'companies':records,
        'old_company_probabilities':{c['cik']:c['inclusion_probability'] for c in old['companies']},
        'final_rows_used':False, 'limits':config['scope']}
    ROOT.mkdir(parents=True,exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != result:raise ValueError('Committed historical expansion changed')
    else:
        with PLAN.open('x') as handle:json.dump(result,handle,indent=2)
    return result


def acquire(company, config, annual, cases):
    sources=[]
    with patch.object(research_dataset,'fetch',coordinated_fetch):
        metadata,name=outcome_load_submissions(company,'2010-01-01','2022-07-19',RAW,sources)
    metadata += amendments(sources,company,'2010-01-01','2022-07-19')
    path=RAW/'companyfacts'/('CIK'+company['cik']+'.json')
    sources.append(coordinated_fetch('https://data.sec.gov/api/xbrl/companyfacts/CIK'+company['cik']+'.json',path))
    payload=json.loads(path.read_text())
    if str(payload['cik']).zfill(10)!=company['cik']:raise ValueError('Financial issuer mismatch')
    try:extract_history(payload,company['ticker'],public_filings=metadata)
    except ValueError as exc:
        if not str(exc).startswith('No original 10-K/10-Q USD Assets facts'):raise
        return {'status':'excluded','reason':'no_original_usd_assets','sources':sources}
    # Reuse the frozen snapshot/coverage implementation with actual newly
    # acquired public sources; no historical labels or predictions are copied.
    with patch.object(training,'history_sources',lambda c,s:((payload,name),metadata,sources,None)),\
         patch.object(training,'RAW',RAW),patch.object(coverage,'fetch',coordinated_fetch),\
         patch.object(coverage,'corroborate',corroborate_v3):
        return training.acquire_company(company,config,{},annual,cases)


def prepare():
    plan,config=commit(),json.loads(CONFIG.read_text())
    annual=pd.read_csv('data/processed/validation/cohort_v3/annual_frame.csv',dtype={'cik':str,'filed':str})
    cases=load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    folder=ROOT/'checkpoints';folder.mkdir(exist_ok=True)
    rows,errors,exclusions,proofs=[],[],[],[]
    for index,company in enumerate(plan['companies'],1):
        print(f'Historical complement {index}/{len(plan["companies"])} {company["cik"]}',flush=True)
        marker=folder/(company['cik']+'.json')
        try:
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['plan_sha256']!=digest(PLAN):raise ValueError('Checkpoint plan changed')
                for source in record['sources']:
                    if digest(source['path'])!=source['sha256']:raise ValueError('Source bytes changed')
            else:
                try:record=acquire(company,config,annual,cases)
                except requests.HTTPError as exc:
                    if exc.response is None or exc.response.status_code!=404:raise
                    record={'status':'excluded','reason':'public_source_404','sources':[],
                        'unavailable_url':exc.response.url,'http_status':404,'label_assigned':False}
                record['plan_sha256']=digest(PLAN);write_json(marker,record)
            proofs.append({'cik':company['cik'],'sha256':digest(marker)})
            if record['status']=='complete':rows.extend(record['rows']);exclusions.extend(record['exclusions'])
            else:exclusions.append({'cik':company['cik'],'reason':record['reason']})
        except Exception as exc:errors.append({'cik':company['cik'],'error':str(exc)})
    extra=pd.DataFrame(rows)
    extra.to_csv(ROOT/'extension.csv',index=False)
    old=pd.read_csv(BASE,dtype={'cik':str})
    frame=set(plan['extension_frame']);proxies=set(plan['sec_inventory_proxies'])
    def probability(cik):
        extension=(1. if cik in proxies else config['background_probability']) if cik in frame else 0.
        return union_probability(plan['old_company_probabilities'][cik],extension)
    old['inclusion_probability']=old.cik.map(probability);old['design_weight']=1/old.inclusion_probability
    combined=pd.concat([old,extra],ignore_index=True)
    if combined.unit_id.duplicated().any() or set(combined.cik)&set(plan['recent_reserved_companies']):raise ValueError('Training/evaluation leakage')
    combined.to_csv(ROOT/'training.csv',index=False)
    result={'plan_sha256':digest(PLAN),'training_sha256':digest(ROOT/'training.csv'),
        'extension_sha256':digest(ROOT/'extension.csv'),'selected_companies':len(plan['companies']),
        'rows':len(combined),'eligible_rows':int(combined[TARGET].notna().sum()),
        'positive_companies':int(combined.loc[combined[TARGET].eq(1),'cik'].nunique()),
        'errors':errors,'exclusions':exclusions,'checkpoint_proofs':proofs,'final_rows_used':False,'production_ready':False}
    write_json(ROOT/'report.json',result);return result


if __name__=='__main__':
    with local_only():print(json.dumps(prepare(),indent=2))
