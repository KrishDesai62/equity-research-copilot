"""Uniform outcome-only audit of reserved companies; never fits or scores a model.

Uses the issuer rule already frozen on development. Model feature columns are
excluded when reading the final CSV. Old labels, artifacts and packets persist.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_layer.research_dataset import write_json, write_jsonl
from src.validation import sec_disclosure_admission as evidence
from src.validation.cohort_v3_data import digest
from src.validation.issuer_identity_admission_v2 import apply, corroborate_identity
from src.validation.sec_disclosure_data import TARGET

ROOT = Path('data/processed/validation/final_outcome_audit_20261019')
SOURCE = Path('data/processed/validation/sec_disclosure_admission_v3/final.csv')
COLUMNS = ['unit_id', 'cik', 'company_name', 'prediction_date', 'horizon_end',
           'label_available_date', TARGET, 'label_reason']


def outcome_rows(path):
    return pd.read_csv(path, usecols=COLUMNS, dtype={'cik': str})


def run():
    mapping_path = Path('data/processed/validation/sec_disclosure_admission_v2/report.json')
    mapping = json.loads(mapping_path.read_text())['checkpoint_inputs']
    frame = outcome_rows(SOURCE)
    if frame.unit_id.duplicated().any():raise ValueError('Duplicate reserved outcome units')
    inputs = {str(p):digest(p) for p in [SOURCE, mapping_path, Path(__file__),
              Path('src/validation/issuer_identity_admission_v2.py'),Path(evidence.__file__)]}
    inputs.update({mapping[c]['path']:digest(mapping[c]['path']) for c in frame.cik})
    ROOT.mkdir(parents=True,exist_ok=True);saved=ROOT/'report.json'
    if saved.exists():
        report=json.loads(saved.read_text())
        if report['inputs']!=inputs:raise ValueError('Frozen final outcome audit changed')
        for name,expected in report['outputs'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen outcome-only output changed')
        return report
    events={};proofs=[]
    with patch.object(evidence,'corroborate_v2',corroborate_identity):
        for index,cik in enumerate(sorted(frame.cik),1):
            print(f'Final outcome audit {index}/{len(frame)} {cik}',flush=True)
            if digest(mapping[cik]['path'])!=mapping[cik]['sha256']:raise ValueError('Reserved source checkpoint changed')
            events[cik],found=evidence.company_proofs(cik,Path(mapping[cik]['path']),200)
            proofs.extend(found)
    corrected,changes=apply(frame,events)
    corrected.to_csv(ROOT/'outcomes.csv',index=False)
    write_json(ROOT/'source_evidence.json',proofs)
    tasks=[]
    for row in corrected.loc[corrected[TARGET].notna()].to_dict('records'):
        primary=[{'source':p['source'],'assertion':a} for p in proofs if p['cik']==row['cik']
                 for a in p['assertions'] if a['issuer_explicitly_included']
                 and row['prediction_date']<a['petition_date']<=row['horizon_end']
                 and a['filing_date']<=row['label_available_date']]
        if row[TARGET]==1 and not primary:raise ValueError('Positive lacks updated issuer evidence')
        tasks.append({'unit_id':row['unit_id'],'cik':row['cik'],'company_name':row['company_name'],
            'prediction_date':row['prediction_date'],'horizon_end':row['horizon_end'],
            'evidence_cutoff':row['label_available_date'],'automated_proposed_label':int(row[TARGET]),
            'target':TARGET,'scope':'SEC-disclosed issuer Chapter 7/11; court coverage is not asserted',
            'positive_statement_evidence':primary,'source_checkpoint':mapping[row['cik']],
            'review_status':'pending_independent_review','model_predictions_included':False})
    write_jsonl(ROOT/'pending_final_reviews.jsonl',tasks)
    outputs={p.name:digest(p) for p in [ROOT/'outcomes.csv',ROOT/'source_evidence.json',ROOT/'pending_final_reviews.jsonl']}
    positives=sum(t['automated_proposed_label'] for t in tasks)
    manifest={'packet_sha256':outputs['pending_final_reviews.jsonl'],'review_units':len(tasks),
              'positive_units':positives,'negative_units':len(tasks)-positives,
              'independent_reviews_completed':0,'production_approval':False,
              'model_predictions_included':False,'issuer_rule_sha256':digest('src/validation/issuer_identity_admission_v2.py')}
    write_json(ROOT/'manifest.json',manifest);outputs['manifest.json']=digest(ROOT/'manifest.json')
    report={'inputs':inputs,'outputs':outputs,'changes':changes,'resolved_companies':len(tasks),
            'positive_companies':positives,'negative_companies':len(tasks)-positives,
            'unknown_companies':int(corrected[TARGET].isna().sum()),'new_negative_labels':0,
            'final_model_feature_columns_read':False,'model_predictions_created':False,
            'independent_human_review':False,'production_approval':False,
            'scope':'Source-admission audit, not final performance evaluation. The old final dataset and packet are unchanged.'}
    write_json(saved,report);return report


if __name__=='__main__':print(json.dumps(run(),indent=2))
