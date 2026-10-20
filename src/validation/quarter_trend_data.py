"""Build quarter trends and a source-linked false-alert audit without network access."""
import json
from pathlib import Path
import pandas as pd
from src.ml.quarter_trends import balances,at_snapshot,FEATURES
from src.validation.cohort_v3_data import digest
from src.validation import filing_text_signals as text
from src.validation.matched_development_text import checkpoint_map
from src.validation.recent_study_features import ROOT as SOURCE
from src.validation.recent_study_data import ROOT as DATA
from src.data_layer.research_dataset import write_json
from src.validation.sec_disclosure_data import TARGET

ROOT=Path('data/processed/validation/quarter_trends_20261019')


def run():
    paths=[SOURCE/(r+'.csv') for r in ['training','calibration']]
    reference=json.loads(Path('data/processed/validation/recent_study_models_20261019/selection.json').read_text())
    for p in paths:
        if digest(p)!=reference['inputs'][str(p)]:raise ValueError('Frozen source table changed')
    mapping=checkpoint_map()
    for role in ['training','calibration']:
        for p in (DATA/(role+'_checkpoints')).glob('*.json'):
            record=json.loads(p.read_text())
            if any('/submissions/' in x['path'] for x in record.get('sources',[])):mapping[p.stem]=str(p)
    inputs={str(p):digest(p) for p in [*paths,Path(__file__),Path('src/ml/quarter_trends.py')]}
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if report['inputs']!=inputs:raise ValueError('Frozen trend inputs changed')
        for p,sha in {**report['outputs'],**report['source_hashes']}.items():
            if digest(p)!=sha:raise ValueError('Frozen trend evidence changed')
        return report
    frames={r:pd.read_csv(p,dtype={'cik':str}) for r,p in zip(['training','calibration'],paths)}
    cache={};proofs={};outputs={};links=[];counts={}
    for role,frame in frames.items():
        additions=[]
        for i,row in enumerate(frame.to_dict('records'),1):
            cik=row['cik']
            if cik not in cache:
                marker=Path(mapping[cik]);record=json.loads(marker.read_text());proofs[str(marker)]=digest(marker)
                for source in record['sources']:
                    if digest(source['path'])!=source['sha256']:raise ValueError('Cached public source changed')
                    proofs[source['path']]=source['sha256']
                facts_sources=[s for s in record['sources'] if '/companyfacts/' in s['path']]
                if len(facts_sources)!=1:raise ValueError('Ambiguous company facts source')
                facts=json.loads(Path(facts_sources[0]['path']).read_text())
                if str(facts['cik']).zfill(10)!=cik:raise ValueError('Issuer mismatch')
                metadata=text.metadata_for(cik,marker)
                cache[cik]=balances(facts,metadata)
                print('Quarter histories',len(cache),cik,flush=True)
            values,prior=at_snapshot(row,cache[cik]);additions.append(values)
            links.append({'unit_id':row['unit_id'],'role':role,'prior':prior})
        extra=pd.DataFrame(additions);result=pd.concat([frame.reset_index(drop=True),extra],axis=1)
        path=ROOT/(role+'.csv');result.to_csv(path,index=False);outputs[str(path)]=digest(path)
        counts[role]={'rows':len(frame),'rows_with_quarter_cash_change':int(extra.q_cash_change_to_prior_assets.notna().sum())}
    write_json(ROOT/'provenance.json',links);outputs[str(ROOT/'provenance.json')]=digest(ROOT/'provenance.json')
    cal=frames['calibration'];alerts=pd.read_csv('data/processed/validation/recent_operating_policy_20261019/alerts.csv')
    merged=cal.merge(alerts[['unit_id','alerts_3']],on='unit_id',validate='one_to_one')
    false=merged.loc[merged[TARGET].eq(0)&merged.alerts_3.eq(1)].copy();audit=[]
    for row in false.to_dict('records'):
        marker=DATA/'calibration_checkpoints'/(row['cik']+'.json');record=json.loads(marker.read_text())
        audit.append({**{k:row[k] for k in ['cik','company_name','unit_id','design_weight','source_filing_date','horizon_end','label_available_date','label_reason']},
           'checkpoint':str(marker),'checkpoint_sha256':digest(marker),'reporting_followup_complete':record['reporting_followup_complete'],
           'unresolved_accessions':record['unresolved_accessions'],'reviewed_disclosures':record['reviews'],
           'assessment':'Distressed SEC-disclosure negative under frozen evidence protocol; no contrary issuer petition evidence established by this cache audit. Not a court-record clearance or independent human label review.'})
    write_json(ROOT/'false_positive_audit.json',audit);outputs[str(ROOT/'false_positive_audit.json')]=digest(ROOT/'false_positive_audit.json')
    report={'inputs':inputs,'outputs':outputs,'source_hashes':proofs,'counts':counts,'features':FEATURES,'false_alert_companies':len(audit),'production_approval':False,'final_rows_used':False}
    write_json(saved,report);return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
