"""Acquire exact historical input filings and extract auditable text signals."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from lxml import html
import lxml
import numpy as np
import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.sec_disclosure_repair import outcome_submission_rows
from src.validation.sec_disclosure_data import TARGET

CONFIG=Path('config/development_20261019/text_signals.json')
ROOT=Path('data/processed/validation/filing_text_20261019')
RAW=Path('data/raw/filing_text_20261019')
FEATURES=['text_going_concern_rate','text_substantial_doubt_rate','text_covenant_breach_rate',
    'text_liquidity_shortfall_rate','text_restructuring_rate','text_auditor_change_rate',
    'text_financing_dependency_rate','text_log_word_count']
PATTERNS={
    'text_going_concern_rate':r'\bgoing\s+concern\b',
    'text_substantial_doubt_rate':r'\bsubstantial\s+(?:doubt|uncertainty)\b',
    'text_covenant_breach_rate':r'\b(?:in\s+default|covenant\s+(?:breach|violation)|breach\s+of\s+(?:our\s+)?covenants?)\b',
    'text_liquidity_shortfall_rate':r'\b(?:insufficient\s+(?:cash|liquidity)|liquidity\s+shortfall|unable\s+to\s+meet\s+(?:our\s+)?(?:obligations|payments))\b',
    'text_restructuring_rate':r'\b(?:restructuring|reorganization)\b',
    'text_auditor_change_rate':r'\b(?:dismissed|resigned|change\s+in)\b.{0,80}\b(?:auditor|accountant)\b',
    'text_financing_dependency_rate':r'\b(?:additional\s+financing|raise\s+additional\s+capital|additional\s+funding)\b'}


def visible_text(payload):
    if not payload or not payload.strip():raise ValueError('Empty filing document')
    parser=html.HTMLParser(no_network=True)
    doc=html.fromstring(payload,parser=parser)
    if any('excessive depth' in e.message.lower() or 'resource limit' in e.message.lower() for e in parser.error_log):
        raise ValueError('HTML parser reported incomplete resource-limited content')
    # HTML parsing does not execute code or fetch external references.
    for node in list(doc.xpath('//*[self::script or self::style or self::head or self::noscript or local-name()="ix:hidden" or @hidden]')):
        if node is doc:return ''
        if node.getparent() is not None:node.drop_tree()
    for node in list(doc.xpath('//*[@style]')):
        style=re.sub(r'\s+','',node.get('style','')).lower()
        if 'display:none' in style or 'visibility:hidden' in style:
            if node is doc:return ''
            if node.getparent() is not None:node.drop_tree()
    return re.sub(r'\s+',' ',' '.join(doc.itertext())).strip()


def extract_signals(payload):
    text=visible_text(payload)
    if re.search(r'request rate threshold exceeded|your request originates from an undeclared automated tool',text,re.I):
        raise ValueError('SEC access response is not a filing')
    words=len(re.findall(r'\b[a-zA-Z][a-zA-Z\'-]*\b',text))
    if words<200:raise ValueError('Document too short to establish complete filing-text coverage')
    result={name:len(re.findall(pattern,text,re.I))*10000/words for name,pattern in PATTERNS.items()}
    result['text_log_word_count']=float(np.log1p(words))
    proofs=[]
    for name,pattern in PATTERNS.items():
        for match in list(re.finditer(pattern,text,re.I))[:3]:
            proofs.append({'feature':name,'start':match.start(),'end':match.end(),
                'matched_text':match.group(), 'context_sha256':sha256(text[max(0,match.start()-100):match.end()+100].encode()).hexdigest()})
    return result,{'word_count':words,'visible_text_sha256':sha256(text.encode()).hexdigest(),'matches':proofs}


def metadata_for(cik,marker):
    record=json.loads(Path(marker).read_text());rows={}
    for source in record['sources']:
        if '/submissions/' not in source['path']:continue
        if digest(source['path'])!=source['sha256']:raise ValueError('Text metadata source changed')
        payload=json.loads(Path(source['path']).read_text())
        if 'cik' in payload and str(payload['cik']).zfill(10)!=cik:raise ValueError('Text issuer metadata mismatch')
        block=payload.get('filings',{}).get('recent',payload)
        for row in outcome_submission_rows(block,cik,'CIK'+cik,'2010-01-01','2026-07-19'):
            if row['filing_type'] in {'10-K','10-Q'}:rows[row['accession_number']]=row
    return rows


def document_for(row,metadata):
    accession=row.get('source_accession_number')
    if pd.isna(accession) or not accession:accession=str(row['accession_number']).split('@')[0]
    item=metadata.get(accession)
    filed=row.get('feature_filing_date')
    if pd.isna(filed) or not filed:filed=row.get('source_filing_date')
    if item is None or item['filing_date']!=filed or item['filing_type'] not in {'10-K','10-Q'} or filed>=row['prediction_date']:
        raise ValueError('Exact original financial filing/date unavailable for text features')
    return {'cik':row['cik'],'accession':accession,'filing_date':filed,'form':item['filing_type'],'url':item['source_url']}


def checkpoint_map():
    base=json.loads(Path('data/processed/validation/sec_disclosure_admission_v2/report.json').read_text())['checkpoint_inputs']
    result={c:p['path'] for c,p in base.items()}
    folder=Path('data/processed/validation/historical_training_20261019/checkpoints')
    result.update({p.stem:str(p) for p in folder.glob('*.json')})
    return result


def prepare(phase='base'):
    if phase not in {'base','expanded'}:raise ValueError('Unknown text acquisition phase')
    sources={'training':Path('data/processed/validation/sec_filing_signals_v4/training.csv') if phase=='base' else Path('data/processed/validation/historical_training_20261019/training.csv'),
        'calibration':Path('data/processed/validation/sec_filing_signals_v4/calibration.csv')}
    if phase=='expanded':
        report=json.loads(sources['training'].with_name('report.json').read_text())
        if report['errors'] or report['training_sha256']!=digest(sources['training']):raise ValueError('Expanded source acquisition is not complete')
    frames={r:pd.read_csv(p,dtype={'cik':str}) for r,p in sources.items()}
    frames={r:f.loc[f[TARGET].notna()].copy() for r,f in frames.items()}
    mapping=checkpoint_map();cached={};documents={};links={}
    for role,frame in frames.items():
        for row in frame.to_dict('records'):
            cik=row['cik']
            if cik not in cached:cached[cik]=metadata_for(cik,mapping[cik])
            doc=document_for(row,cached[cik]);key=cik+'_'+doc['accession']
            documents[key]=doc;links[row['unit_id']]=key
    ROOT.mkdir(parents=True,exist_ok=True)
    plan={'config_sha256':digest(CONFIG),'implementation_sha256':digest(__file__),
        'rate_helper_sha256':digest('src/validation/public_source_fetch.py'),
        'html_parser_version':lxml.__version__, 'text_requirements_sha256':digest('requirements-filing-text.lock'),
        'source_hashes':{r:digest(p) for r,p in sources.items()},'documents':documents,'unit_document_map':links,
        'final_rows_loaded':False}
    planpath=ROOT/(phase+'_acquisition.json')
    if planpath.exists():
        if json.loads(planpath.read_text())!=plan:raise ValueError('Frozen text acquisition changed')
    else:
        with planpath.open('x') as handle:json.dump(plan,handle,indent=2)
    folder=ROOT/'documents';folder.mkdir(exist_ok=True)
    results,errors,proofs={},[],[]
    for index,(key,doc) in enumerate(sorted(documents.items()),1):
        print(f'Text filing {index}/{len(documents)} {key}',flush=True)
        marker=folder/(key+'.json')
        try:
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['config_sha256']!=digest(CONFIG) or record['implementation_sha256']!=digest(__file__) or record['document']!=doc or digest(record['source']['path'])!=record['source']['sha256']:raise ValueError('Text checkpoint changed')
            else:
                path=RAW/(key+'.html');source=coordinated_fetch(doc['url'],path)
                values,evidence=extract_signals(path.read_bytes())
                record={'document':doc,'values':values,'evidence':evidence,'source':source,
                    'config_sha256':digest(CONFIG),'implementation_sha256':digest(__file__)}
                write_json(marker,record)
            results[key]=record['values'];proofs.append({'key':key,'sha256':digest(marker)})
        except Exception as exc:errors.append({'key':key,'error':str(exc),'features_assigned':False})
    outputs={}
    for role,frame in frames.items():
        additions=[results.get(links[unit],{k:np.nan for k in FEATURES}) for unit in frame.unit_id]
        combined=pd.concat([frame.reset_index(drop=True),pd.DataFrame(additions)],axis=1)
        dest=ROOT/(phase+'_'+role+'.csv');combined.to_csv(dest,index=False);outputs[role]=digest(dest)
    result={'plan_sha256':digest(planpath),'requested_documents':len(documents),'completed_documents':len(results),
        'errors':errors,'checkpoint_proofs':proofs,'output_hashes':outputs,'final_rows_loaded':False,'production_ready':False}
    write_json(ROOT/(phase+'_report.json'),result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['base','expanded'],default='base')
    args=parser.parse_args();print(json.dumps(prepare(args.phase),indent=2))
