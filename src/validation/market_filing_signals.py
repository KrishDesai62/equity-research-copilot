"""Pre-prediction exchange/auditor/restructuring notices and industry context."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.matched_development_text import checkpoint_map
from src.validation.sec_filing_signals import read_metadata
from src.validation.contextual_filing_signals import prepare as context_prepare, ROOT as CONTEXT_ROOT, TERMS, BOILERPLATE, NEGATION
from src.validation.filing_text_signals import visible_text

ROOT=Path('data/processed/validation/market_context_signals_20261019')
ITEMS={'market_delisting_90d':('3.01',90),'market_delisting_365d':('3.01',365),
       'market_equity_sales_90d':('3.02',90),'market_equity_sales_365d':('3.02',365),
       'market_auditor_change_365d':('4.01',365),'market_agreement_termination_90d':('1.02',90),
       'market_exit_costs_365d':('2.05',365),'market_impairments_365d':('2.06',365),
       'market_security_rights_365d':('5.03',365)}
FEATURES=list(ITEMS)+['industry_division_'+str(i) for i in range(10)]


def values_at(rows,annual,anchor):
    values={}
    for feature,(item,days) in ITEMS.items():
        eligible=[r for r in rows if r['form'] in {'8-K','8-K/A'} and r['date']<anchor
                  and pd.Timestamp(r['date'])>=pd.Timestamp(anchor)-pd.Timedelta(days=days)]
        values[feature]=np.nan if any(not r['items'].strip() for r in eligible) else sum(item in set(re.findall(r'\d+\.\d{2}',r['items'])) for r in eligible)
    prior=annual.loc[annual.filed.lt(anchor)].sort_values('filed')
    if prior.empty:division=None
    else:
        latest=prior.loc[prior.filed.eq(prior.filed.max())]
        sic=pd.to_numeric(latest.sic,errors='coerce')
        if sic.isna().any() or sic.nunique()!=1:raise ValueError('Ambiguous prior annual SIC')
        division=int(sic.iloc[0])//1000
    values.update({'industry_division_'+str(i):float(division==i) if division is not None else np.nan for i in range(10)})
    return values


def corrected_liquidity(text,original):
    # "Do not have sufficient cash" is an affirmative shortfall, not a
    # negation of distress. Preserve the archived experiment and correct this
    # feature and its negated-match counter in this versioned input stage.
    narrow_negation=re.compile(r'\bno\s+(?:material\s+)?(?:substantial doubt|default|violation|breach)|\bnot\s+(?:in\s+)?(?:default|violation)|\b(?:do|does|did) not (?:raise|believe)',re.I)
    words=len(re.findall(r'\b\w+\b',text));included=0;delta=0
    for sentence in re.finditer(r'[^.!?]+(?:[.!?]+|$)',text):
        content=sentence.group()
        for match in re.finditer(TERMS['context_liquidity_shortfall_rate'],content,re.I):
            nearby=content[max(0,match.start()-180):min(len(content),match.end()+180)]
            old=bool(NEGATION.search(nearby));new=bool(narrow_negation.search(nearby))
            delta+=int(new)-int(old)
            included+=int(not BOILERPLATE.search(nearby) and not new)
    return {'context_liquidity_shortfall_rate':included*10000/words,
            'context_negated_distress_rate':max(0.,original['context_negated_distress_rate']+delta*10000/words)}


def prepare():
    context_prepare()
    paths=[Path('data/processed/validation/cohort_v3/annual_frame.csv'),
           Path('data/processed/validation/sec_disclosure_v1/annual_frame.csv')]
    annual=pd.concat([pd.read_csv(p,dtype={'cik':str}) for p in paths],ignore_index=True).drop_duplicates(['cik','filed','sic'])
    frames={r:pd.read_csv(CONTEXT_ROOT/(r+'.csv'),dtype={'cik':str}) for r in ['training','calibration']}
    mapping=checkpoint_map();rows={};sources={}
    for cik in sorted(set().union(*(set(f.cik) for f in frames.values()))):
        rows[cik],proofs=read_metadata(cik,mapping[cik]);sources.update({p['path']:p for p in proofs})
    inputs={str(p):digest(p) for p in paths+[CONTEXT_ROOT/'training.csv',CONTEXT_ROOT/'calibration.csv',Path(__file__),Path('src/validation/sec_filing_signals.py')]}
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if report['inputs']!=inputs:raise ValueError('Frozen market-context inputs changed')
        for name,expected in report['outputs'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen market-context output changed')
        return report
    text_plan=json.loads(Path('data/processed/validation/matched_text_20261019/expanded_acquisition.json').read_text())
    corrections={};correction_proofs=[]
    for marker in sorted((CONTEXT_ROOT/'documents').glob('*.json')):
        record=json.loads(marker.read_text())
        if not any(m['feature']=='context_liquidity_shortfall_rate' for m in record['evidence']['matches']):continue
        text=visible_text(Path(record['source']['path']).read_bytes())
        corrected=corrected_liquidity(text,record['values'])
        corrections[marker.stem]=corrected
        if any(abs(corrected[k]-record['values'][k])>1e-10 for k in corrected):
            correction_proofs.append({'document':marker.stem,'source':record['source'],
                                      'old':{k:record['values'][k] for k in corrected},'new':corrected})
    outputs={}
    annuals={c:f for c,f in annual.groupby('cik')}
    for role,frame in frames.items():
        for index,row in frame.iterrows():
            key=text_plan['unit_document_map'][row.unit_id]
            for name,value in corrections.get(key,{}).items():frame.at[index,name]=value
        additions=[values_at(rows[r.cik],annuals.get(r.cik,annual.iloc[:0]),r.prediction_date) for r in frame.itertuples()]
        result=pd.concat([frame.reset_index(drop=True),pd.DataFrame(additions)],axis=1)
        path=ROOT/(role+'.csv');result.to_csv(path,index=False);outputs[path.name]=digest(path)
    write_json(ROOT/'liquidity_corrections.json',correction_proofs);outputs['liquidity_corrections.json']=digest(ROOT/'liquidity_corrections.json')
    report={'corrected_liquidity_documents':len(correction_proofs),'inputs':inputs,'outputs':outputs,'sources':list(sources.values()),'features':FEATURES,
            'errors':[],'final_model_rows_used':False,'production_approval':False,
            'limits':'Filing item categories are risk proxies, not a claim that every notice means distress. Missing item metadata stays unknown.'}
    write_json(saved,report);return report


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
