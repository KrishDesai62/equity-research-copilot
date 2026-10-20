"""Exact pre-forecast filing features for the newer, reserved study."""
import json
from pathlib import Path
import pandas as pd
from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation import filing_text_signals as text
from src.validation import contextual_filing_signals as context
from src.validation.market_filing_signals import values_at, corrected_liquidity
from src.validation.sec_filing_signals import read_metadata, signals_at
from src.validation.matched_development_text import checkpoint_map
from src.validation.recent_study_data import ROOT as DATA, PLAN
from src.validation.recent_study_sources import ROOT as SOURCES
from src.validation.sec_disclosure_data import TARGET

ROOT=Path('data/processed/validation/recent_study_features_20261019')


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    reports={r:json.loads((DATA/(r+'_report.json')).read_text()) for r in ['training','calibration']}
    if any(r['errors'] for r in reports.values()):raise ValueError('Resolve acquisition errors before feature fitting')
    paths=[DATA/(r+'.csv') for r in reports]
    for role,path in zip(reports,paths):
        if digest(path)!=reports[role]['dataset_sha256']:raise ValueError('Acquired dataset changed')
    annual_paths=[Path('data/processed/validation/cohort_v3/annual_frame.csv'),SOURCES/'annual_2023.csv']
    annual=pd.concat([pd.read_csv(p,dtype={'cik':str}) for p in annual_paths]).drop_duplicates(['cik','filed','sic'])
    annuals=dict(tuple(annual.groupby('cik')))
    mapping=checkpoint_map()
    for role in reports:
        for marker in (DATA/(role+'_checkpoints')).glob('*.json'):
            record=json.loads(marker.read_text())
            if any('/submissions/' in source['path'] for source in record.get('sources',[])):
                mapping[marker.stem]=str(marker)
    dependencies=[Path(__file__),PLAN,*paths,*annual_paths,Path(text.__file__),Path(context.__file__),Path('src/validation/market_filing_signals.py'),Path('src/validation/sec_filing_signals.py')]
    inputs={str(p):digest(p) for p in dependencies}
    saved=ROOT/'report.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if report['inputs']!=inputs:raise ValueError('Frozen feature inputs changed')
        for p,sha in report['outputs'].items():
            if digest(p)!=sha:raise ValueError('Frozen feature output changed')
        for proof in report['source_proofs']:
            if digest(proof['path'])!=proof['sha256']:raise ValueError('Frozen feature source changed')
        return report
    metadata={};filings={};proofs={};outputs={};errors=[]
    for role,path in zip(reports,paths):
        frame=pd.read_csv(path,dtype={'cik':str});frame=frame.loc[frame[TARGET].notna()].copy()
        additions=[]
        for i,row in enumerate(frame.to_dict('records'),1):
            print('Newer features',role,i,len(frame),row['unit_id'],flush=True)
            try:
                cik=row['cik']
                if cik not in metadata:
                    metadata[cik],source=read_metadata(cik,mapping[cik]);filings[cik]=text.metadata_for(cik,mapping[cik])
                    for p in source:proofs[p['path']]=p
                doc=text.document_for(row,filings[cik]);key=cik+'_'+doc['accession']
                marker=ROOT/'documents'/(key+'.json')
                if marker.exists():
                    record=json.loads(marker.read_text())
                    if record['document']!=doc or record['inputs']!=inputs:raise ValueError('Newer text cache changed')
                    if digest(record['source']['path'])!=record['source']['sha256']:raise ValueError('Newer document changed')
                else:
                    old=text.ROOT/'documents'/(key+'.json')
                    if old.exists():
                        cached=json.loads(old.read_text())
                        if cached['document']!=doc or cached['implementation_sha256']!=digest(text.__file__) or cached['config_sha256']!=digest(text.CONFIG):raise ValueError('Original text cache identity changed')
                        source=cached['source']
                        if digest(source['path'])!=source['sha256']:raise ValueError('Original filing changed')
                    else:source=text.coordinated_fetch(doc['url'],ROOT/'raw'/(key+'.html'))
                    payload=Path(source['path']).read_bytes();base,_=text.extract_signals(payload)
                    visible=text.visible_text(payload);extra,_=context.signals(visible);extra.update(corrected_liquidity(visible,extra))
                    record={'inputs':inputs,'document':doc,'source':source,'values':{**base,**extra}}
                    write_json(marker,record)
                proofs[record['source']['path']]=record['source']
                values={**signals_at(metadata[cik],row['prediction_date']),**record['values'],**values_at(metadata[cik],annuals.get(cik,annual.iloc[:0]),row['prediction_date'])}
                additions.append(values)
            except Exception as exc:
                errors.append({'role':role,'unit_id':row['unit_id'],'error':str(exc)});additions.append({})
        extra=pd.DataFrame(additions);frame=frame.drop(columns=list(set(extra.columns)&set(frame.columns)))
        result=pd.concat([frame.reset_index(drop=True),extra],axis=1);out=ROOT/(role+'.csv');result.to_csv(out,index=False);outputs[str(out)]=digest(out)
    report={'inputs':inputs,'outputs':outputs,'source_proofs':list(proofs.values()),'errors':errors,'final_rows_used':False,'production_approval':False}
    # Failed documents remain resumable; a completed seal is only written after all succeed.
    write_json(ROOT/('report.json' if not errors else 'incomplete_report.json'),report)
    return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
