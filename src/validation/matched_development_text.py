"""Use the unchanged text parser/cache on expanded, matched development cohorts."""
import argparse
import json
from pathlib import Path

import lxml
import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation import filing_text_signals as kernel
from src.validation.sec_disclosure_data import TARGET

ROOT = Path('data/processed/validation/matched_text_20261019')


def checkpoint_map():
    result = kernel.checkpoint_map()
    result.update({p.stem:str(p) for p in Path('data/processed/validation/recent_development_20261019/checkpoints').glob('*.json')})
    return result


def prepare(phase='base'):
    if phase not in {'base','expanded'}:raise ValueError('Unknown training phase')
    audit=Path('data/processed/validation/issuer_identity_admission_v2_20261019')
    sources = {'training':audit/(phase+'_training.csv'), 'calibration':audit/'calibration.csv'}
    from src.validation.issuer_identity_admission_v2 import run as verify_audit
    report=verify_audit()
    for path in sources.values():
        if report['output_hashes'][path.name]!=digest(path):raise ValueError('Audited dataset changed')
    frames={r:pd.read_csv(p,dtype={'cik':str}) for r,p in sources.items()}
    frames={r:f.loc[f[TARGET].notna()].copy() for r,f in frames.items()}
    if set(frames['training'].cik)&set(frames['calibration'].cik):raise ValueError('Company leakage')
    checkpoints=checkpoint_map();metadata={};documents={};links={}
    for frame in frames.values():
        for row in frame.to_dict('records'):
            cik=row['cik']
            if cik not in metadata:metadata[cik]=kernel.metadata_for(cik,checkpoints[cik])
            doc=kernel.document_for(row,metadata[cik]);key=cik+'_'+doc['accession']
            documents[key]=doc;links[row['unit_id']]=key
    ROOT.mkdir(parents=True,exist_ok=True)
    plan={'implementation_sha256':digest(__file__),'kernel_sha256':digest(kernel.__file__),
          'config_sha256':digest(kernel.CONFIG),'rate_helper_sha256':digest('src/validation/public_source_fetch.py'),
          'html_parser_version':lxml.__version__,'requirements_sha256':digest('requirements-filing-text.lock'),
          'source_hashes':{r:digest(p) for r,p in sources.items()},
          'documents':documents,'unit_document_map':links,'final_rows_loaded':False}
    planpath=ROOT/(phase+'_acquisition.json')
    if planpath.exists():
        if json.loads(planpath.read_text())!=plan:raise ValueError('Frozen matched acquisition changed')
    else:
        with planpath.open('x') as handle:json.dump(plan,handle,indent=2)
    # Original parser hash and exact original document identity are mandatory.
    # Reuse already verified bytes; no new parser or silent migration is involved.
    folder=kernel.ROOT/'documents';folder.mkdir(parents=True,exist_ok=True)
    values,errors,proofs={},[],[]
    for index,(key,doc) in enumerate(sorted(documents.items()),1):
        print(f'Matched text {phase} {index}/{len(documents)} {key}',flush=True)
        marker=folder/(key+'.json')
        try:
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['config_sha256']!=digest(kernel.CONFIG) or record['implementation_sha256']!=digest(kernel.__file__) or record['document']!=doc or digest(record['source']['path'])!=record['source']['sha256']:raise ValueError('Original text cache changed')
            else:
                path=kernel.RAW/(key+'.html');proof=kernel.coordinated_fetch(doc['url'],path)
                signals,evidence=kernel.extract_signals(path.read_bytes())
                record={'document':doc,'values':signals,'evidence':evidence,'source':proof,
                        'config_sha256':digest(kernel.CONFIG),'implementation_sha256':digest(kernel.__file__)}
                write_json(marker,record)
            values[key]=record['values'];proofs.append({'key':key,'sha256':digest(marker)})
        except Exception as exc:errors.append({'key':key,'error':str(exc),'features_assigned':False})
    outputs={}
    for role,frame in frames.items():
        additions=[values.get(links[unit],{k:np.nan for k in kernel.FEATURES}) for unit in frame.unit_id]
        combined=pd.concat([frame.reset_index(drop=True),pd.DataFrame(additions)],axis=1)
        path=ROOT/(phase+'_'+role+'.csv');combined.to_csv(path,index=False);outputs[role]=digest(path)
    report={'plan_sha256':digest(planpath),'requested_documents':len(documents),'completed_documents':len(values),
            'errors':errors,'checkpoint_proofs':proofs,'output_hashes':outputs,'final_rows_loaded':False,'production_ready':False}
    write_json(ROOT/(phase+'_report.json'),report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['base','expanded'],default='base')
    print(json.dumps(prepare(parser.parse_args().phase),indent=2))
