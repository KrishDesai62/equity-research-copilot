"""Versioned issuer-subject correction; preserves every earlier dataset/artifact."""
from hashlib import sha256
import json
from pathlib import Path
import re
from unittest.mock import patch

import pandas as pd

from src.data_layer.filing_changes import normalize_filing
from src.data_layer.research_dataset import write_json
from src.validation.adjudicate_public_events import canonical
from src.validation.cohort_v3_data import digest
from src.validation import sec_disclosure_admission as evidence
from src.validation.sec_disclosure_admission_v3 import corroborate_v3, LIQUIDATION, legal_name
from src.validation.filing_text_signals import checkpoint_map
from src.validation.sec_disclosure_data import TARGET

ROOT=Path('data/processed/validation/issuer_identity_admission_v3_20261019')


def direct_issuer_subject(subject,names):
    from src.validation.issuer_identity_admission_v2 import direct_issuer_subject as direct
    if direct(subject,names):return True
    initial=re.sub(r'^\s*(?:\([^)]{0,100}\)\s*)?[,;:]?\s*','',subject).strip()
    if not re.match(r'^after considering\b',initial,re.I):return False
    # A disclosed deliberation clause may precede the actual subject list.
    # Only complete legal-name subjects at clause boundaries can match; names
    # inside an ownership phrase still fail the unchanged direct-subject rule.
    for comma in re.finditer(',',initial):
        if direct(initial[comma.end():],names):return True
    return False


def corroborate_identity(text,*,filed,names,maximum_lag=200):
    plain=re.sub(r'\s+',' ',normalize_filing(text))
    result=[]
    for assertion in corroborate_v3(text,filed=filed,names=names,maximum_lag=maximum_lag):
        start=assertion['statement_start']
        match=evidence.STATEMENT.match(plain,start) or LIQUIDATION.match(plain,start)
        if match is None:raise ValueError('Unrecoverable source statement boundary')
        checked=dict(assertion)
        checked['issuer_explicitly_included']=direct_issuer_subject(match.group('subject'),names)
        # Do not mistake the date of a previously filed Current Report for a
        # petition date. Leave that statement unresolved unless another actual
        # petition-date statement establishes the event.
        prefix=plain[max(0,start-180):start]
        if re.search(r'(?:current\s+report|form\s+8\s*-\s*k)[^.]{0,140}\bfiled\s*$',prefix,re.I):
            checked['issuer_explicitly_included']=False
            checked['identity_review_reason']='report_filing_date_is_not_petition_date'
        checked['subject_sha256']=sha256(match.group('subject').encode()).hexdigest()
        result.append(checked)
    return result


def apply(frame,events):
    updated=frame.copy();changes=[]
    for index,row in frame.iterrows():
        eligible=[e for e in events.get(row.cik,[]) if e['issuer_explicitly_included'] and e['filing_date']<=row.label_available_date]
        prior=any(e['petition_date']<=row.prediction_date for e in eligible)
        positive=any(row.prediction_date<e['petition_date']<=row.horizon_end for e in eligible)
        original=row[TARGET];label=original;reason=None
        if prior:label,reason=None,'direct_issuer_prior_event_censored'
        elif pd.notna(original) and original==1 and not positive:label,reason=None,'issuer_identity_or_petition_date_requires_review'
        elif pd.notna(original) and original==0 and positive:label,reason=None,'contradictory_negative_requires_review'
        if reason and pd.notna(original):
            updated.at[index,TARGET]=label;updated.at[index,'label_reason']=reason
            changes.append({'unit_id':row.unit_id,'cik':row.cik,'old_label':int(original),'new_label':None,'reason':reason})
    return updated,changes


def run():
    sources={'base_training':Path('data/processed/validation/sec_filing_signals_v4/training.csv'),
             'expanded_training':Path('data/processed/validation/historical_training_20261019/training.csv'),
             'calibration':Path('data/processed/validation/recent_development_20261019/calibration.csv')}
    if json.loads(sources['calibration'].with_name('report.json').read_text())['errors']:raise ValueError('Recent acquisition incomplete')
    mapping=checkpoint_map();mapping.update({p.stem:str(p) for p in sources['calibration'].parent.joinpath('checkpoints').glob('*.json')})
    frames={r:pd.read_csv(p,dtype={'cik':str}) for r,p in sources.items()}
    companies=set().union(*(set(f.cik) for f in frames.values()))
    inputs={str(p):digest(p) for p in sources.values()}
    inputs.update({mapping[c]:digest(mapping[c]) for c in companies})
    inputs.update({str(p):digest(p) for p in [Path(__file__),Path(evidence.__file__),Path('src/validation/sec_disclosure_admission_v3.py'),Path('src/validation/adjudicate_public_events.py'),Path('src/validation/issuer_identity_admission_v2.py')]})
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists():
        record=json.loads(saved.read_text())
        if record['inputs']!=inputs:raise ValueError('Frozen issuer audit changed')
        for name,expected in record['output_hashes'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen issuer audit output changed')
        return record
    events={};proofs=[]
    with patch.object(evidence,'corroborate_v2',corroborate_identity):
        for index,cik in enumerate(sorted(companies),1):
            print(f'Issuer identity audit {index}/{len(companies)} {cik}',flush=True)
            events[cik],found=evidence.company_proofs(cik,Path(mapping[cik]),200);proofs.extend(found)
    counts={};changes={};outputs={}
    for role,frame in frames.items():
        updated,changes[role]=apply(frame,events)
        path=ROOT/(role+'.csv');updated.to_csv(path,index=False);outputs[path.name]=digest(path)
        counts[role]={'rows':len(updated),'eligible_rows':int(updated[TARGET].notna().sum()),
                      'positive_companies':int(updated.loc[updated[TARGET].eq(1),'cik'].nunique())}
    write_json(ROOT/'evidence.json',proofs);outputs['evidence.json']=digest(ROOT/'evidence.json')
    result={'inputs':inputs,'output_hashes':outputs,'counts':counts,'changes':changes,
            'new_negative_labels':0,'final_rows_loaded':False,'independent_human_review':False,'production_ready':False}
    write_json(saved,result);return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
