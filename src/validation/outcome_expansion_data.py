"""Expand admissible historical outcomes and reserve disjoint later companies."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import requests
from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest,uniform
from src.validation.historical_coverage import load_reviewed_registry
from src.validation import recent_study_data as kernel
from src.validation import outcome_expansion_sources as sources
from src.validation.sec_disclosure_data import TARGET

ROOT=Path('data/processed/validation/outcome_expansion_20261019')
PLAN=Path('config/outcome_expansion_20261019/acquisition.json')
BASE=Path('data/processed/validation/recent_study_data_20261019/training.csv')


def load(path):return json.loads(Path(path).read_text())


def prior_selected():
    ids=set();proofs={}
    for p in sorted(Path('config').rglob('*.json')):
        if 'outcome_expansion_20261019' in str(p):continue
        record=load(p)
        if not isinstance(record,dict):continue
        used=False
        for key in ['companies','training','calibration']:
            rows=record.get(key)
            if isinstance(rows,list):
                for row in rows:
                    if isinstance(row,dict) and isinstance(row.get('cik'),str):ids.add(row['cik'].zfill(10));used=True
        if used:proofs[str(p)]=digest(p)
    return ids,proofs


def commit():
    sources.prepare();config=load(sources.CONFIG);sources.chronology(config)
    previous=load('config/recent_study_20261019/acquisition.json')
    old=load('config/cohort_v3_acquisition.json');history=load('config/development_20261019/training_acquisition.json')
    all_prior,plans=prior_selected();old_training={c['cik']:c for c in previous['training']}
    forbidden=(all_prior-set(old_training))|set(old['excluded_companies'])
    annual_path=Path('data/processed/validation/cohort_v3/annual_frame.csv')
    annual=pd.read_csv(annual_path,dtype={'cik':str});sic=pd.to_numeric(annual.sic,errors='coerce')
    base=annual.loc[annual.filed.ge('2014-01-01')&sic.notna()&~sic.between(6000,6999)]
    # Reconstruct the prior sampling frame exactly for the union probabilities.
    prior_forbidden=set(previous['reserved_final_ids'])|set(previous['prior_development_excluded_from_training'])|set(old['excluded_companies'])
    prior_frame=set(base.loc[base.filed.lt('2021-05-31')&~base.cik.isin(prior_forbidden),'cik'])
    frame=base.loc[base.filed.lt(config['new_training_anchor'])&~base.cik.isin(forbidden)].sort_values('filed').drop_duplicates('cik',keep='last')
    proxy_paths=[Path('data/processed/validation/historical_sec_inventory_20261019/candidates.csv'),Path('data/processed/validation/recent_study_sources_20261019/training_extension/candidates.csv')]
    prior_proxies=set().union(*(set(pd.read_csv(p,dtype={'cik':str}).cik) for p in proxy_paths))
    proxies=prior_proxies|set(pd.read_csv(sources.ROOT/'training_extension/candidates.csv',dtype={'cik':str}).cik)
    selected=dict(old_training);new=[]
    for row in frame.loc[~frame.cik.isin(selected)].sort_values('cik').itertuples():
        q=1. if row.cik in proxies else .05
        if uniform(config['seed']+':training',row.cik)>=q:continue
        selected[row.cik]={'cik':row.cik,'ticker':'CIK'+row.cik,'company_name':row.name};new.append(row.cik)
    probabilities={}
    for cik in selected:
        p=previous['old_training_probabilities'].get(cik)
        if p is None:p=kernel.inclusion_probability(cik,old,history,prior_frame,prior_proxies,.05)
        q=(1. if cik in proxies else .05) if cik in set(frame.cik) else 0.
        probabilities[cik]=p+(1-p)*q
    training=[]
    for cik,c in sorted(selected.items()):
        training.append({'cik':cik,'ticker':c['ticker'],'company_name':c['company_name'],'role':'training','inclusion_probability':probabilities[cik],
            'design_weight':1/probabilities[cik],'sampling_stratum':'october_2021_extension',
            'anchors':config['training_anchors'] if cik in new else [config['new_training_anchor']]})
    freshannual=pd.read_csv(sources.ROOT/'annual_2025.csv',dtype={'cik':str})
    freshframe=freshannual.loc[~freshannual.cik.isin(all_prior|set(selected)|set(old['excluded_companies']))].sort_values('filed').drop_duplicates('cik',keep='last')
    freshproxies=set(pd.read_csv(sources.ROOT/'fresh_reservation/candidates.csv',dtype={'cik':str}).cik)
    reservation=[]
    for row in freshframe.sort_values('cik').itertuples():
        pi=1. if row.cik in freshproxies else .1
        if uniform(config['seed']+':fresh',row.cik)>=pi:continue
        reservation.append({'cik':row.cik,'ticker':'CIK'+row.cik,'company_name':row.name,'role':'calibration','frame_member':True,'inclusion_probability':pi,'design_weight':1/pi})
    pool,pool_inputs=kernel.catalog()
    for role in ['training','calibration']:
        for p in sorted((kernel.ROOT/(role+'_checkpoints')).glob('*.json')):
            record=load(p);pool_inputs[str(p)]=digest(p)
            for source in record.get('sources',[]):
                if source.get('url'):pool[source['url']]={k:source[k] for k in ['url','path','sha256']}
    inputs=[sources.CONFIG,BASE,annual_path,sources.ROOT/'report.json',Path(__file__),Path(kernel.__file__),Path('config/recent_study_20261019/acquisition.json')]
    record={'inputs':{str(p):digest(p) for p in inputs},'prior_plan_inputs':plans,'prior_selected_ids':sorted(all_prior),'training':training,'fresh_reservation':reservation,
       'new_training_ids':new,'training_probabilities':probabilities,'fresh_eligible_population':int(freshframe.cik.nunique()),
       'cache_pool':pool,'cache_checkpoint_inputs':pool_inputs,'training_evidence_cutoff':sources.chronology(config)[0],
       'fresh_evidence_cutoff':(pd.Timestamp(config['fresh_evaluation_anchor'])+pd.DateOffset(years=1)+pd.Timedelta(days=config['reporting_grace_days'])).date().isoformat(),
       'limits':'Conditional fresh population excludes all previously selected companies. This is a reservation, not a new passed final evaluation. No model predictions permitted by this module.'}
    ROOT.mkdir(parents=True,exist_ok=True)
    if PLAN.exists():
        if load(PLAN)!=record:raise ValueError('Frozen outcome expansion changed')
    else:
        with PLAN.open('x') as handle:json.dump(record,handle,indent=2)
    return record


def run(role):
    if role not in {'training','fresh_reservation'}:raise ValueError('Unknown role')
    plan=commit();config=load(sources.CONFIG);folder=ROOT/(role+'_checkpoints');folder.mkdir(exist_ok=True)
    prior=load_reviewed_registry().dropna(subset=['cik']).sort_values('filing_date').drop_duplicates('cik').set_index('cik').filing_date.to_dict()
    annual=pd.read_csv('data/processed/validation/cohort_v3/annual_frame.csv',dtype={'cik':str})
    fetch=lambda url,path:kernel.cached_fetch(url,path,plan['cache_pool'])
    rows=[];errors=[];exclusions=[];proofs=[]
    for i,company in enumerate(plan[role],1):
        print('Expansion',role,i,len(plan[role]),company['cik'],flush=True)
        marker=folder/(company['cik']+'.json')
        try:
            if marker.exists():
                record=load(marker)
                if record['expansion_plan_sha256']!=digest(PLAN):raise ValueError('Expansion checkpoint changed')
                for source in record['sources']:
                    if digest(source['path'])!=source['sha256']:raise ValueError('Expansion source changed')
            else:
                try:
                    with patch.object(kernel,'RAW',sources.RAW):
                        if role=='training':record=kernel.acquire_training(company,config,annual,prior,fetch)
                        else:record=kernel.acquire_calibration(company,{**config,'calibration_anchor':config['fresh_evaluation_anchor']},prior,fetch)
                except requests.HTTPError as exc:
                    if exc.response is None or exc.response.status_code!=404:raise
                    record={'status':'excluded','reason':'public_source_404','sources':[]}
                except ValueError as exc:
                    if not str(exc).startswith('No original 10-K/10-Q USD Assets facts'):raise
                    rawrole='training' if role=='training' else 'calibration'
                    p=sources.RAW/rawrole/'companyfacts'/('CIK'+company['cik']+'.json')
                    record={'status':'excluded','reason':'no_original_usd_assets','sources':[{'path':str(p),'sha256':digest(p)}]}
                record['expansion_plan_sha256']=digest(PLAN);write_json(marker,record)
            proofs.append({'path':str(marker),'sha256':digest(marker)})
            if record['status']=='complete':
                collected=record['rows'] if role=='training' else [record['row']]
                for row in collected:
                    if role=='fresh_reservation':row={**row,'role':'reserved_fresh','reserved_no_model_predictions':True}
                    rows.append(row)
            else:exclusions.append({'cik':company['cik'],'reason':record['reason']})
        except Exception as exc:errors.append({'cik':company['cik'],'error':str(exc),'label_assigned':False})
    frame=pd.DataFrame(rows)
    if role=='training':
        old=pd.read_csv(BASE,dtype={'cik':str});old['inclusion_probability']=old.cik.map(plan['training_probabilities']);old['design_weight']=1/old.inclusion_probability
        frame=pd.concat([old,frame],ignore_index=True)
    if not frame.empty:
        if frame.unit_id.duplicated().any():raise ValueError('Duplicate expansion units')
        if role=='fresh_reservation' and set(frame.cik)&set(plan['prior_selected_ids']):raise ValueError('Fresh reservation overlaps past companies')
        if role=='training' and set(frame.cik)&{c['cik'] for c in plan['fresh_reservation']}:raise ValueError('Train/fresh overlap')
        frame.to_csv(ROOT/(role+'.csv'),index=False)
    report={'plan_sha256':digest(PLAN),'rows':len(frame),'errors':errors,'exclusions':exclusions,'checkpoints':proofs,'production_approval':False,'model_predictions_generated':False,
      'positive_companies':int(frame.loc[frame[TARGET].eq(1),'cik'].nunique()) if not frame.empty else 0,
      'negative_rows':int(frame[TARGET].eq(0).sum()) if not frame.empty else 0,'unknown_rows':int(frame[TARGET].isna().sum()) if not frame.empty else 0}
    if not frame.empty:report['dataset_sha256']=digest(ROOT/(role+'.csv'))
    write_json(ROOT/(role+'_report.json'),report);return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['training','fresh_reservation'],required=True);args=parser.parse_args();print(json.dumps(run(args.role),indent=2))
