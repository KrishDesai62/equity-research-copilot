"""Frozen newer training extension and 2023-05 development cohort, public only."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from src.data_layer import research_dataset
from src.data_layer.research_dataset import write_json
from src.ml.financial_history import extract_history
from src.validation import sec_disclosure_data as original
from src.validation import sec_disclosure_training as training
from src.validation import sec_disclosure_coverage as coverage
from src.validation.cohort_v3_data import digest,uniform
from src.validation.historical_coverage import load_reviewed_registry
from src.validation.issuer_identity_admission_v3 import corroborate_identity
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.recent_study_sources import ROOT as SOURCE,RAW,CONFIG,chronology,prepare as prepare_sources
from src.validation.sec_disclosure_repair import outcome_load_submissions

ROOT=Path('data/processed/validation/recent_study_data_20261019')
PLAN=Path('config/recent_study_20261019/acquisition.json')
BASE=Path('data/processed/validation/issuer_identity_admission_v3_20261019/expanded_training.csv')
CHECKPOINT_ROOTS=['cohort_v3/company_checkpoints','sec_disclosure_v1/checkpoints','sec_disclosure_v2/checkpoints',
                 'historical_training_20261019/checkpoints','recent_development_20261019/checkpoints']


def load(path):return json.loads(Path(path).read_text())


def catalog():
    pool={};inputs={}
    for directory in CHECKPOINT_ROOTS:
        for path in sorted((Path('data/processed/validation')/directory).glob('*.json')):
            record=load(path);inputs[str(path)]=digest(path)
            for source in record.get('sources',[]):
                if source.get('url') and source.get('sha256'):
                    pool[source['url']]={k:source[k] for k in ['url','path','sha256']}
    return pool,inputs


def inclusion_probability(cik,old,history,new_frame,proxies,background):
    p0=next((c['inclusion_probability'] for c in old['companies'] if c['cik']==cik),.1)
    p1=(1. if cik in history['sec_inventory_proxies'] else .1) if cik in history['extension_frame'] else 0.
    previous=p0+(1-p0)*p1
    extra=(1. if cik in proxies else background) if cik in new_frame else 0.
    return previous+(1-previous)*extra


def commit():
    prepare_sources();config=load(CONFIG);chronology(config)
    old=load('config/cohort_v3_acquisition.json');history=load('config/development_20261019/training_acquisition.json')
    v1=load('config/sec_disclosure/acquisition_v1.json');v2=load('config/sec_disclosure_expansion/acquisition_v2.json')
    old_cal=load('config/development_20261019/calibration_acquisition.json')
    final={c['cik'] for plan in [v1,v2] for c in plan['companies'] if c['role']=='final'}
    development={c['cik'] for c in v1['companies'] if c['role']=='calibration'}|{c['cik'] for c in old_cal['companies']}
    old_training={c['cik']:c for plan in [old,history] for c in plan['companies']}
    if set(old_training)&(final|development):raise ValueError('Prior study company split violated')
    annual_path=Path('data/processed/validation/cohort_v3/annual_frame.csv')
    annual=pd.read_csv(annual_path,dtype={'cik':str})
    sic=pd.to_numeric(annual.sic,errors='coerce')
    frame=annual.loc[annual.filed.between('2014-01-01',config['new_training_anchor'],inclusive='left') & sic.notna() & ~sic.between(6000,6999)]
    frame=frame.loc[~frame.cik.isin(final|development|set(old['excluded_companies']))].sort_values('filed').drop_duplicates('cik',keep='last')
    new_frame=set(frame.cik)
    proxies=set(pd.read_csv('data/processed/validation/historical_sec_inventory_20261019/candidates.csv',dtype={'cik':str}).cik)
    proxies|=set(pd.read_csv(SOURCE/'training_extension/candidates.csv',dtype={'cik':str}).cik)
    selected=dict(old_training);new_records=[]
    for row in frame.loc[~frame.cik.isin(selected)].sort_values('cik').itertuples():
        q=1. if row.cik in proxies else config['training_extension_background_probability']
        if uniform(config['seed']+':training',row.cik)>=q:continue
        record={'cik':row.cik,'ticker':'CIK'+row.cik,'company_name':row.name}
        selected[row.cik]=record;new_records.append(row.cik)
    probabilities={c:inclusion_probability(c,old,history,new_frame,proxies,config['training_extension_background_probability']) for c in selected}
    # Existing training rows are retained and reweighted uniformly; only their
    # new date is acquired. Newly selected companies receive every train anchor.
    records=[]
    for cik,c in sorted(selected.items()):
        records.append({'cik':cik,'ticker':c['ticker'],'company_name':c['company_name'],'role':'training',
                        'sampling_stratum':'recent_extension','inclusion_probability':probabilities[cik],
                        'design_weight':1/probabilities[cik],
                        'anchors':config['training_anchors'] if cik in new_records else [config['new_training_anchor']]})
    calannual=pd.read_csv(SOURCE/'annual_2023.csv',dtype={'cik':str})
    calframe=calannual.loc[~calannual.cik.isin(set(selected)|final|set(old['excluded_companies']))].sort_values('filed').drop_duplicates('cik',keep='last')
    calproxies=set(pd.read_csv(SOURCE/'new_calibration/candidates.csv',dtype={'cik':str}).cik)
    calibration=[]
    for row in calframe.sort_values('cik').itertuples():
        pi=1. if row.cik in calproxies else config['calibration_background_probability']
        if uniform(config['seed']+':calibration',row.cik)>=pi:continue
        calibration.append({'cik':row.cik,'ticker':'CIK'+row.cik,'company_name':row.name,'role':'calibration',
                            'frame_member':True,'inclusion_probability':pi,'design_weight':1/pi})
    pool,pool_inputs=catalog()
    paths=[CONFIG,BASE,annual_path,SOURCE/'report.json',SOURCE/'annual_2023.csv',Path(__file__),
           Path('src/validation/issuer_identity_admission_v3.py'),Path('src/validation/sec_disclosure_training.py'),
           Path('src/validation/sec_disclosure_data.py'),Path('src/validation/sec_disclosure_coverage.py'),
           Path('src/validation/public_source_fetch.py'),Path('src/ml/financial_history.py')]
    paths += [Path(p) for p in ['config/cohort_v3_acquisition.json','config/development_20261019/training_acquisition.json',
                              'config/sec_disclosure/acquisition_v1.json','config/sec_disclosure_expansion/acquisition_v2.json',
                              'config/development_20261019/calibration_acquisition.json']]
    result={'source_hashes':{str(p):digest(p) for p in paths},'training':records,'calibration':calibration,
            'old_training_probabilities':probabilities,'new_training_companies':new_records,
            'reserved_final_ids':sorted(final),'prior_development_excluded_from_training':sorted(development),
            'conditional_calibration_population':int(calframe.cik.nunique()),'cache_pool':pool,
            'cache_checkpoint_inputs':pool_inputs,'chronology':dict(zip(['training_cutoff','calibration_cutoff'],chronology(config))),
            'limits':config['scope']+' Conditional sampling weights do not repair population exclusions or outcome censoring.'}
    ROOT.mkdir(parents=True,exist_ok=True)
    if PLAN.exists():
        if load(PLAN)!=result:raise ValueError('Frozen newer acquisition changed')
    else:
        with PLAN.open('x') as handle:json.dump(result,handle,indent=2)
    return result


def cached_fetch(url,path,pool):
    path=Path(path)
    if url in pool:
        cached=pool[url]
        if digest(cached['path'])!=cached['sha256']:raise ValueError('Frozen cached source changed')
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(Path(cached['path']).resolve())
        if digest(path)!=cached['sha256']:raise ValueError('Conflicting source for frozen cached URL')
        return {'url':url,'path':str(path),'sha256':cached['sha256'],'bytes':path.stat().st_size}
    return coordinated_fetch(url,path)


def acquire_training(company,config,annual,prior,fetch):
    # Skip dates with no annual membership or already-known prior petitions
    # without assigning an outcome. The original reserved IDs stay reserved.
    membership=annual.loc[annual.cik.eq(company['cik']) & annual.filed.lt(max(company['anchors']))]
    if membership.empty:return {'status':'excluded','reason':'no_prior_annual_membership','sources':[]}
    anchors=[a for a in company['anchors'] if prior.get(company['cik'],'9999-12-31')>a]
    if not anchors:return {'status':'excluded','reason':'known_prior_registry_petition','sources':[]}
    sources=[];raw=RAW/'training'
    cutoff=chronology(config)[0]
    with patch.object(research_dataset,'fetch',fetch):
        metadata,name=outcome_load_submissions(company,'2010-01-01',cutoff,raw,sources)
    metadata+=original.amendments(sources,company,'2010-01-01',cutoff)
    path=raw/'companyfacts'/('CIK'+company['cik']+'.json')
    sources.append(fetch('https://data.sec.gov/api/xbrl/companyfacts/CIK'+company['cik']+'.json',path))
    facts=load(path)
    if str(facts['cik']).zfill(10)!=company['cik']:raise ValueError('Training issuer mismatch')
    extract_history(facts,company['ticker'],public_filings=metadata)
    options={**config,'anchors':anchors}
    with patch.object(training,'history_sources',lambda c,s:((facts,name),metadata,sources,None)),\
         patch.object(training,'RAW',raw),patch.object(coverage,'fetch',fetch),patch.object(coverage,'corroborate',corroborate_identity):
        return training.acquire_company(company,options,{},annual,prior)


def acquire_calibration(company,config,prior,fetch):
    raw=RAW/'calibration'
    def facts(cik,directory):
        path=Path(directory)/('CIK'+cik+'.json');proof=fetch('https://data.sec.gov/api/xbrl/companyfacts/CIK'+cik+'.json',path)
        return load(path),proof
    with patch.object(original,'RAW',raw),patch.object(original,'fetch',fetch),patch.object(original,'companyfacts',facts),\
         patch.object(original,'load_submissions',outcome_load_submissions),patch.object(research_dataset,'fetch',fetch),\
         patch.object(original,'corroborate',corroborate_identity):
        return original.acquire(company,config,prior)


def prepare(role):
    if role not in {'training','calibration'}:raise ValueError('Invalid acquisition role')
    plan=commit();config=load(CONFIG);folder=ROOT/(role+'_checkpoints');folder.mkdir(exist_ok=True)
    annual=pd.read_csv('data/processed/validation/cohort_v3/annual_frame.csv',dtype={'cik':str})
    prior=load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    fetch=lambda url,path:cached_fetch(url,path,plan['cache_pool'])
    rows=[];errors=[];exclusions=[];proofs=[]
    for index,company in enumerate(plan[role],1):
        print(f'Newer {role} {index}/{len(plan[role])} {company["cik"]}',flush=True)
        marker=folder/(company['cik']+'.json')
        try:
            if marker.exists():
                record=load(marker)
                if record['plan_sha256']!=digest(PLAN):raise ValueError('Newer checkpoint plan changed')
                for source in record['sources']:
                    if digest(source['path'])!=source['sha256']:raise ValueError('Newer source bytes changed')
            else:
                try:
                    record=acquire_training(company,config,annual,prior,fetch) if role=='training' else acquire_calibration(company,config,prior,fetch)
                except requests.HTTPError as exc:
                    if exc.response is None or exc.response.status_code!=404:raise
                    record={'status':'excluded','reason':'public_source_404','sources':[],'unavailable_url':exc.response.url}
                except ValueError as exc:
                    if not str(exc).startswith('No original 10-K/10-Q USD Assets facts'):raise
                    path=RAW/role/'companyfacts'/('CIK'+company['cik']+'.json')
                    record={'status':'excluded','reason':'no_original_usd_assets','sources':[{'url':'https://data.sec.gov/api/xbrl/companyfacts/CIK'+company['cik']+'.json','path':str(path),'sha256':digest(path)}]}
                record['plan_sha256']=digest(PLAN);write_json(marker,record)
            proofs.append({'cik':company['cik'],'sha256':digest(marker)})
            if record['status']=='complete':rows.extend(record['rows'] if role=='training' else [record['row']])
            else:exclusions.append({'cik':company['cik'],'reason':record['reason']})
        except Exception as exc:errors.append({'cik':company['cik'],'error':str(exc),'label_assigned':False})
    frame=pd.DataFrame(rows)
    if role=='training':
        old=pd.read_csv(BASE,dtype={'cik':str});old['inclusion_probability']=old.cik.map(plan['old_training_probabilities']);old['design_weight']=1/old.inclusion_probability
        frame=pd.concat([old,frame],ignore_index=True)
    if not frame.empty:
        if frame.unit_id.duplicated().any() or set(frame.cik)&set(plan['reserved_final_ids']):raise ValueError('Duplicate/new-final company leakage')
        frame.to_csv(ROOT/(role+'.csv'),index=False)
    report={'plan_sha256':digest(PLAN),'selected_companies':len(plan[role]),'rows':len(frame),
            'positive_companies':int(frame.loc[frame[original.TARGET].eq(1),'cik'].nunique()) if not frame.empty else 0,
            'negative_rows':int(frame[original.TARGET].eq(0).sum()) if not frame.empty else 0,
            'unknown_rows':int(frame[original.TARGET].isna().sum()) if not frame.empty else 0,
            'errors':errors,'exclusions':exclusions,'checkpoint_proofs':proofs,'final_rows_used':False,'production_approval':False}
    if not frame.empty:report['dataset_sha256']=digest(ROOT/(role+'.csv'))
    write_json(ROOT/(role+'_report.json'),report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['training','calibration'],required=True)
    print(json.dumps(prepare(parser.parse_args().role),indent=2))
