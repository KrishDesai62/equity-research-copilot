"""Build unchanged 63-feature inputs for expanded training; never loads fresh reservation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_layer.research_dataset import write_json
from src.ml.detection_models import FEATURES
from src.validation import filing_text_signals as text
from src.validation import contextual_filing_signals as context
from src.validation.market_filing_signals import values_at,corrected_liquidity
from src.validation.sec_filing_signals import read_metadata,signals_at
from src.validation.matched_development_text import checkpoint_map
from src.validation.cohort_v3_data import digest
from src.validation.outcome_expansion_data import ROOT as DATA,PLAN
from src.validation.sec_disclosure_data import TARGET,FEATURES as FINANCIAL

ROOT=Path('data/processed/validation/outcome_expansion_features_20261019')
PREVIOUS=Path('data/processed/validation/recent_study_features_20261019')


def run():
    acquired=json.loads((DATA/'training_report.json').read_text())
    source=DATA/'training.csv'
    if acquired['errors'] or digest(source)!=acquired['dataset_sha256']:raise ValueError('Training evidence incomplete or changed')
    previous=json.loads(Path('data/processed/validation/recent_study_models_20261019/selection.json').read_text())
    old_paths=[PREVIOUS/(r+'.csv') for r in ['training','calibration']]
    for p in old_paths:
        if digest(p)!=previous['inputs'][str(p)]:raise ValueError('Previous development table changed')
    annual_paths=[Path('data/processed/validation/cohort_v3/annual_frame.csv'),Path('data/processed/validation/recent_study_sources_20261019/annual_2023.csv')]
    paths=[source,DATA/'training_report.json',PLAN,*old_paths,*annual_paths,Path(__file__),Path(text.__file__),Path(context.__file__),Path('src/validation/sec_filing_signals.py'),Path('src/validation/market_filing_signals.py')]
    inputs={str(p):digest(p) for p in paths};ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if report['inputs']!=inputs:raise ValueError('Frozen expanded feature inputs changed')
        for p,sha in {**report['outputs'],**report['source_hashes']}.items():
            if digest(p)!=sha:raise ValueError('Frozen feature bytes changed')
        return report
    train=pd.read_csv(source,dtype={'cik':str});train=train.loc[train[TARGET].notna()].copy()
    old=pd.read_csv(old_paths[0],dtype={'cik':str}).set_index('unit_id')
    cal=pd.read_csv(old_paths[1],dtype={'cik':str})
    if set(train.cik)&set(cal.cik):raise ValueError('Train/development company overlap')
    annual=pd.concat([pd.read_csv(p,dtype={'cik':str}) for p in annual_paths]).drop_duplicates(['cik','filed','sic']);annuals=dict(tuple(annual.groupby('cik')))
    mapping=checkpoint_map()
    for parent in [Path('data/processed/validation/recent_study_data_20261019'),DATA]:
        for p in (parent/'training_checkpoints').glob('*.json'):
            r=json.loads(p.read_text())
            if any('/submissions/' in s['path'] for s in r.get('sources',[])):mapping[p.stem]=str(p)
    metadata={};filings={};proofs={};additions=[];errors=[];reused=0
    for i,row in enumerate(train.to_dict('records'),1):
        print('Expanded features',i,len(train),row['unit_id'],flush=True)
        try:
            if row['unit_id'] in old.index:
                previous_row=old.loc[row['unit_id']]
                if not np.allclose(np.array([row[k] for k in FINANCIAL],float),previous_row[FINANCIAL].to_numpy(float),equal_nan=True):raise ValueError('Financial inputs for reused unit changed')
                if previous_row.feature_filing_date!=row['feature_filing_date'] or previous_row[TARGET]!=row[TARGET]:raise ValueError('Reused unit date/outcome changed')
                additions.append(previous_row[FEATURES].to_dict());reused+=1;continue
            cik=row['cik']
            if cik not in metadata:
                marker=Path(mapping[cik]);proofs[str(marker)]=digest(marker)
                metadata[cik],sources=read_metadata(cik,marker);filings[cik]=text.metadata_for(cik,marker)
                for p in sources:proofs[p['path']]=p['sha256']
            doc=text.document_for(row,filings[cik]);key=cik+'_'+doc['accession'];marker=ROOT/'documents'/(key+'.json')
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['inputs']!=inputs or record['document']!=doc or digest(record['source']['path'])!=record['source']['sha256']:raise ValueError('Expanded text cache changed')
            else:
                old_marker=text.ROOT/'documents'/(key+'.json')
                if old_marker.exists():
                    cached=json.loads(old_marker.read_text())
                    if cached['document']!=doc or cached['implementation_sha256']!=digest(text.__file__) or cached['config_sha256']!=digest(text.CONFIG):raise ValueError('Original text identity changed')
                    source_proof=cached['source']
                    if digest(source_proof['path'])!=source_proof['sha256']:raise ValueError('Original filing changed')
                else:source_proof=text.coordinated_fetch(doc['url'],ROOT/'raw'/(key+'.html'))
                payload=Path(source_proof['path']).read_bytes();base,_=text.extract_signals(payload);visible=text.visible_text(payload)
                extra,_=context.signals(visible);extra.update(corrected_liquidity(visible,extra))
                record={'inputs':inputs,'document':doc,'source':source_proof,'values':{**base,**extra}};write_json(marker,record)
            proofs[record['source']['path']]=record['source']['sha256']
            additions.append({**{k:row[k] for k in FINANCIAL},**signals_at(metadata[cik],row['prediction_date']),**record['values'],**values_at(metadata[cik],annuals.get(cik,annual.iloc[:0]),row['prediction_date'])})
        except Exception as exc:errors.append({'unit_id':row['unit_id'],'error':str(exc)});additions.append({})
    extra=pd.DataFrame(additions);train=train.drop(columns=list(set(extra)&set(train.columns)))
    result=pd.concat([train.reset_index(drop=True),extra],axis=1);outputs={}
    for role,frame in [('training',result),('calibration',cal)]:
        p=ROOT/(role+'.csv');frame.to_csv(p,index=False);outputs[str(p)]=digest(p)
    report={'inputs':inputs,'outputs':outputs,'source_hashes':proofs,'errors':errors,'reused_training_units':reused,'fresh_reservation_loaded':False,'final_rows_used':False,'production_approval':False}
    write_json(ROOT/('report.json' if not errors else 'incomplete_report.json'),report);return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
