"""Precommitted recent DEVELOPMENT complement; never opens final feature rows."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from src.data_layer import research_dataset
from src.data_layer.research_dataset import write_json
from src.validation import sec_disclosure_data as original
from src.validation.calibration_sec_inventory import run as inventory
from src.validation.cohort_v3_data import digest, uniform
from src.validation.historical_coverage import load_reviewed_registry
from src.validation.local_workflow import local_only
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.sec_disclosure_admission_v3 import corroborate_v3
from src.validation.sec_disclosure_repair import outcome_load_submissions

CONFIG = Path('config/development_20261019/calibration_expansion.json')
PLAN = CONFIG.with_name('calibration_acquisition.json')
ROOT = Path('data/processed/validation/recent_development_20261019')
RAW = Path('data/raw/recent_development_20261019')
BASE = Path('data/processed/validation/sec_disclosure_admission_v3/calibration.csv')


def calibration_union_probability(p0, role_pi, final_extension_pi, history_extension_pi, recent_extension_pi):
    values = [p0, role_pi, final_extension_pi, history_extension_pi, recent_extension_pi]
    if any(not 0 <= x <= 1 for x in values) or p0 <= 0 or role_pi <= 0:
        raise ValueError('Invalid development sampling probabilities')
    return p0 * role_pi + (1-p0) * (1-final_extension_pi) * (1-history_extension_pi) * recent_extension_pi


def commit():
    inventory()
    config = json.loads(CONFIG.read_text())
    old = json.loads(original.PLAN.read_text())
    oldconfig = json.loads(original.CONFIG.read_text())
    history = json.loads(Path('config/development_20261019/training_acquisition.json').read_text())
    oldhistory = json.loads(Path('config/cohort_v3_acquisition.json').read_text())
    finalplan = json.loads(Path('config/sec_disclosure_expansion/acquisition_v2.json').read_text())
    annual = pd.read_csv(original.ROOT/'annual_frame.csv', dtype={'cik': str})
    historical = pd.read_csv('data/processed/validation/cohort_v3/annual_frame.csv', dtype={'cik': str})
    sic = pd.to_numeric(historical.sic, errors='coerce')
    history_population = set(historical.loc[historical.filed.between('2014-01-01','2020-12-30') & sic.notna() & ~sic.between(6000,6999), 'cik']) - set(oldhistory['excluded_companies'])
    final_population = set(annual.loc[annual.frame_year.eq(2024), 'cik'])
    final_proxies = set(pd.read_csv('data/processed/validation/sec_event_inventory_v1/candidates.csv', dtype={'cik': str}).cik)
    recent_proxies = set(pd.read_csv('data/processed/validation/calibration_sec_inventory_20261019/candidates.csv', dtype={'cik': str}).cik)
    queue = pd.read_csv(original.QUEUE, dtype={'cik': str})
    original_proxy = queue.assign(year=pd.to_datetime(queue.bankruptcy_date).dt.year).groupby('cik').year.min().to_dict()
    reserved = {c['cik'] for plan in [old, finalplan, history] for c in plan['companies']}
    frame = annual.loc[annual.frame_year.eq(2022)].sort_values('filed').drop_duplicates('cik',keep='last')
    def probability(cik):
        proxy = original_proxy.get(cik)
        p0 = 1. if proxy else oldconfig['background_probability']
        role = oldconfig.get(f'calibration_probability_for_{int(proxy)}_proxy', oldconfig['calibration_probability_other']) if proxy else oldconfig['calibration_probability_other']
        qfinal = (1. if cik in final_proxies else .1) if cik in final_population else 0.
        qhistory = (1. if cik in history['sec_inventory_proxies'] else .1) if cik in history_population else 0.
        qrecent = 1. if cik in recent_proxies else config['background_probability']
        return calibration_union_probability(p0,role,qfinal,qhistory,qrecent)
    records = []
    for row in frame.loc[~frame.cik.isin(reserved)].sort_values('cik').itertuples():
        q = 1. if row.cik in recent_proxies else config['background_probability']
        if uniform(config['seed']+':sample', row.cik) >= q:continue
        pi = probability(row.cik)
        if pi <= 0:raise ValueError('Selected company has zero marginal probability')
        records.append({'cik':row.cik, 'ticker':'CIK'+row.cik, 'company_name':row.name,
                        'role':'calibration', 'frame_member':True, 'inclusion_probability':pi,
                        'design_weight':1/pi, 'acquisition_phase':'recent_development_20261019'})
    paths = [CONFIG, original.CONFIG, original.PLAN, original.QUEUE, BASE,
             Path('config/development_20261019/training_acquisition.json'),
             Path('config/cohort_v3_acquisition.json'), Path('config/sec_disclosure_expansion/acquisition_v2.json'),
             original.ROOT/'annual_frame.csv', Path('data/processed/validation/cohort_v3/annual_frame.csv'),
             Path('data/processed/validation/sec_event_inventory_v1/candidates.csv'),
             Path('data/processed/validation/calibration_sec_inventory_20261019/candidates.csv'),
             Path(__file__), Path(original.__file__), Path('src/validation/sec_disclosure_admission_v3.py'),
             Path('src/validation/public_source_fetch.py')]
    result = {'source_hashes':{str(p):digest(p) for p in paths}, 'companies':records,
              'all_reserved_ids_excluded':sorted(reserved),
              'calibration_probabilities':{c:probability(c) for c in frame.cik},
              'final_rows_loaded':False, 'scope':config['scope']}
    ROOT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != result:raise ValueError('Frozen development acquisition changed')
    else:
        with PLAN.open('x') as handle:json.dump(result,handle,indent=2)
    return result


def acquire(company, config, prior):
    def facts(cik, directory):
        path = Path(directory)/('CIK'+cik+'.json')
        proof = coordinated_fetch('https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',path)
        return json.loads(path.read_text()), proof
    with patch.object(original,'RAW',RAW), patch.object(original,'fetch',coordinated_fetch), \
         patch.object(research_dataset,'fetch',coordinated_fetch), \
         patch.object(original,'companyfacts',facts), patch.object(original,'load_submissions',outcome_load_submissions), \
         patch.object(original,'corroborate',corroborate_v3):
        return original.acquire(company,config,prior)


def prepare():
    plan = commit(); config = json.loads(original.CONFIG.read_text())
    prior = load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    folder = ROOT/'checkpoints'; folder.mkdir(exist_ok=True)
    rows, errors, exclusions, proofs = [], [], [], []
    for index, company in enumerate(plan['companies'],1):
        print(f'Recent development {index}/{len(plan["companies"])} {company["cik"]}',flush=True)
        marker = folder/(company['cik']+'.json')
        try:
            if marker.exists():
                record = json.loads(marker.read_text())
                if record['plan_sha256'] != digest(PLAN):raise ValueError('Checkpoint plan changed')
                for source in record['sources']:
                    if digest(source['path']) != source['sha256']:raise ValueError('Cached source changed')
            else:
                try:record = acquire(company,config,prior)
                except requests.HTTPError as exc:
                    if exc.response is None or exc.response.status_code != 404:raise
                    record = {'status':'excluded','reason':'public_source_404','sources':[], 'unavailable_url':exc.response.url}
                except ValueError as exc:
                    if not str(exc).startswith('No original 10-K/10-Q USD Assets facts'):raise
                    # Reacquire cached facts to preserve a proof even for exclusions.
                    path=RAW/'companyfacts'/('CIK'+company['cik']+'.json')
                    record={'status':'excluded','reason':'no_original_usd_assets', 'sources':[{'path':str(path),'sha256':digest(path)}]}
                record['plan_sha256']=digest(PLAN);write_json(marker,record)
            proofs.append({'cik':company['cik'],'sha256':digest(marker)})
            if record['status']=='complete':rows.append(record['row'])
            else:exclusions.append({'cik':company['cik'],'reason':record['reason']})
        except Exception as exc:errors.append({'cik':company['cik'],'error':str(exc),'label_assigned':False})
    extra = pd.DataFrame(rows); extra.to_csv(ROOT/'extension.csv',index=False)
    old = pd.read_csv(BASE,dtype={'cik':str})
    old['inclusion_probability'] = old.cik.map(plan['calibration_probabilities'])
    if old.inclusion_probability.isna().any() or (old.inclusion_probability<=0).any():raise ValueError('Missing inclusion probability')
    old['design_weight'] = 1/old.inclusion_probability
    combined = pd.concat([old,extra],ignore_index=True)
    if combined.unit_id.duplicated().any():raise ValueError('Duplicate calibration company')
    combined.to_csv(ROOT/'calibration.csv',index=False)
    report = {'plan_sha256':digest(PLAN),'calibration_sha256':digest(ROOT/'calibration.csv'),
              'selected_companies':len(plan['companies']),'rows':len(combined),
              'positive_companies':int(combined.loc[combined[original.TARGET].eq(1),'cik'].nunique()),
              'negative_companies':int(combined.loc[combined[original.TARGET].eq(0),'cik'].nunique()),
              'unknown':int(combined[original.TARGET].isna().sum()),'errors':errors,
              'exclusions':exclusions,'checkpoint_proofs':proofs,'final_rows_loaded':False,'production_ready':False}
    write_json(ROOT/'report.json',report);return report


if __name__=='__main__':
    with local_only():print(json.dumps(prepare(),indent=2))
