"""Auditable contextual signals from cached, original pre-prediction filings."""
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.filing_text_signals import visible_text

ROOT=Path('data/processed/validation/contextual_signals_20261019')
BASE=Path('data/processed/validation/matched_models_20261019/expanded')
FEATURES=['context_going_concern_rate','context_liquidity_shortfall_rate',
          'context_covenant_default_rate','context_forbearance_rate',
          'context_restructuring_advisor_rate','context_financing_need_rate',
          'context_negated_distress_rate','context_audit_boilerplate_rate']
TERMS={
    'context_going_concern_rate':r'substantial doubt|ability to continue as a going concern',
    'context_liquidity_shortfall_rate':r'insufficient (?:cash|liquidity|capital)|(?:do|does|will) not have sufficient (?:cash|liquidity|capital)|cash (?:resources |balances )?(?:will|may) be exhausted',
    'context_covenant_default_rate':r'not in compliance with.{0,100}covenant|(?:breach|violation|default).{0,80}(?:covenant|credit agreement)|(?:covenant|credit agreement).{0,80}(?:breach|violation|default)',
    'context_forbearance_rate':r'entered into.{0,90}forbearance|forbearance agreement|waiver of.{0,70}(?:default|covenant)',
    'context_restructuring_advisor_rate':r'(?:engaged|retained|appointed).{0,90}(?:restructuring|financial) advis[oe]r|(?:exploring|evaluating|pursuing).{0,60}restructuring',
    'context_financing_need_rate':r'(?:need|require).{0,60}(?:additional|further) (?:financing|capital|funding)|(?:dependent|depend).{0,80}(?:additional|further) (?:financing|capital|funding)',
}
BOILERPLATE=re.compile(r'evaluate whether|conclude whether|our responsibilit|auditor.s responsibilit|as part of an audit|in performing our audit|reasonable assurance|could cause.{0,60}substantial doubt',re.I)
NEGATION=re.compile(r'\bno\s+(?:material\s+)?(?:substantial doubt|default|violation|breach)|\bnot\s+(?:in\s+)?(?:default|violation)|\b(?:do|does|did) not (?:raise|have|believe)',re.I)


def signals(text):
    words=len(re.findall(r'\b\w+\b',text))
    if words<200:raise ValueError('Insufficient original filing text')
    counts={f:0 for f in FEATURES};proofs=[]
    # Split at actual sentence boundaries without splitting every "U.S." or
    # decimal. Bounded contexts keep distant boilerplate from masking a warning.
    sentences=list(re.finditer(r'[^.!?]+(?:[.!?]+|$)',text))
    for sentence in sentences:
        content=sentence.group()
        for feature,pattern in TERMS.items():
            for match in re.finditer(pattern,content,re.I):
                nearby=content[max(0,match.start()-180):min(len(content),match.end()+180)]
                boilerplate=bool(BOILERPLATE.search(nearby));negated=bool(NEGATION.search(nearby))
                if boilerplate:counts['context_audit_boilerplate_rate']+=1
                if negated:counts['context_negated_distress_rate']+=1
                if not boilerplate and not negated:counts[feature]+=1
                proofs.append({'feature':feature,'start':sentence.start()+match.start(),
                    'end':sentence.start()+match.end(),'included':not boilerplate and not negated,
                    'context_sha256':sha256(nearby.encode()).hexdigest()})
    return {f:counts[f]*10000/words for f in FEATURES},{'word_count':words,'matches':proofs,'visible_text_sha256':sha256(text.encode()).hexdigest()}


def document_result(args):
    key,doc=args
    marker=Path('data/processed/validation/filing_text_20261019/documents')/(key+'.json')
    record=json.loads(marker.read_text())
    if record['document']!=doc or digest(record['source']['path'])!=record['source']['sha256']:
        raise ValueError('Original filing source changed')
    checkpoint=ROOT/'documents'/(key+'.json')
    inputs={'implementation_sha256':digest(__file__),'source_checkpoint_sha256':digest(marker),
            'text_kernel_sha256':digest('src/validation/filing_text_signals.py')}
    if checkpoint.exists():
        result=json.loads(checkpoint.read_text())
        if result['inputs']!=inputs:raise ValueError('Context document checkpoint changed')
    else:
        text=visible_text(Path(record['source']['path']).read_bytes())
        if sha256(text.encode()).hexdigest()!=record['evidence']['visible_text_sha256']:
            raise ValueError('Visible text source changed')
        values,detail=signals(text)
        result={'inputs':inputs,'values':values,'evidence':{'key':key,'document':doc,**detail},'source':record['source']}
        write_json(checkpoint,result)
    return key,result


def prepare():
    source_plan=Path('data/processed/validation/matched_text_20261019/expanded_acquisition.json')
    plan=json.loads(source_plan.read_text());frames={r:pd.read_csv(BASE/(r+'_inputs.csv'),dtype={'cik':str}) for r in ['training','calibration']}
    inputs={str(p):digest(p) for p in [BASE/'selection.json',BASE/'training_inputs.csv',BASE/'calibration_inputs.csv',source_plan,Path(__file__),Path('src/validation/filing_text_signals.py')]}
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if report['inputs']!=inputs:raise ValueError('Frozen contextual extraction changed')
        for name,expected in report['outputs'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen contextual output changed')
        for proof in report['source_proofs']:
            if digest(proof['path'])!=proof['sha256']:raise ValueError('Original contextual source changed')
        return report
    values={};proofs=[];evidence=[]
    keys={plan['unit_document_map'][unit] for frame in frames.values() for unit in frame.unit_id}
    (ROOT/'documents').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as workers:
        results=workers.map(document_result,[(key,plan['documents'][key]) for key in sorted(keys)],chunksize=2)
        for index,(key,result) in enumerate(results,1):
            print(f'Contextual filing {index}/{len(keys)} {key}',flush=True)
            values[key]=result['values'];proofs.append(result['source']);evidence.append(result['evidence'])
    outputs={}
    for role,frame in frames.items():
        additions=[]
        for row in frame.itertuples():
            key=plan['unit_document_map'][row.unit_id]
            if plan['documents'][key]['filing_date']>=row.prediction_date:raise ValueError('Future contextual filing')
            additions.append(values[key])
        joined=pd.concat([frame.reset_index(drop=True),pd.DataFrame(additions)],axis=1)
        dest=ROOT/(role+'.csv');joined.to_csv(dest,index=False);outputs[dest.name]=digest(dest)
    write_json(ROOT/'evidence.json',evidence);outputs['evidence.json']=digest(ROOT/'evidence.json')
    report={'inputs':inputs,'outputs':outputs,'source_proofs':proofs,'features':FEATURES,
            'documents':len(keys),'final_model_rows_used':False,'production_approval':False,
            'limits':'Context heuristics are features, not legal conclusions; absence of a phrase is not proof of financial health.'}
    write_json(saved,report);return report


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
