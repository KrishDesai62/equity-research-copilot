"""Resumable automated evidence screening, never a human negative sign-off."""
import argparse
import html
import json
from pathlib import Path
import re

from src.validation.cohort_v3_data import digest
from src.validation.filing_text_signals import visible_text
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.sec_disclosure_repair import outcome_submission_rows

ROOT = Path('data/processed/validation/sampled_negative_coverage_v1')
FORMS = {'8-K','8-K/A','10-K','10-K/A','10-Q','10-Q/A','6-K','6-K/A','20-F','20-F/A','40-F','40-F/A'}
TERMS = re.compile(r'\bbankrupt\w*\b|\bchapter\s*(?:7|11|seven|eleven)\b|\breceivership\b|\bpetition\w*\b', re.I)


def scoped_submission_rows(block, cik, start, end):
    """Use shared metadata validation without its financial-input form filter."""
    forms = block.get('form', [])
    expanded = dict(block, form=['8-K' if form in FORMS else '__out_of_scope__' for form in forms])
    original = dict(zip(block.get('accessionNumber', []), forms))
    rows = outcome_submission_rows(expanded, cik, 'CIK'+cik, start, end)
    for row in rows:
        row['filing_type'] = original[row['accession_number']]
    return rows


def inventory(task):
    checkpoint = Path(task['source_checkpoint']['path'])
    if digest(checkpoint) != task['source_checkpoint']['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    record = json.loads(checkpoint.read_text())
    rows, loaded, declared = {}, set(), set()
    for source in record['sources']:
        if '/submissions/' not in source['path']:
            continue
        path = Path(source['path'])
        if digest(path) != source['sha256']:
            raise ValueError('Submission hash mismatch')
        payload = json.loads(path.read_text())
        loaded.add(path.name)
        if 'cik' in payload and str(payload['cik']).zfill(10) != task['cik']:
            raise ValueError('Submission issuer mismatch')
        for history in payload.get('filings', {}).get('files', []):
            if history.get('filingFrom','0000') <= task['evidence_cutoff'] and history.get('filingTo','9999') > task['prediction_date']:
                declared.add(history['name'])
        block = payload.get('filings', {}).get('recent', payload)
        for row in scoped_submission_rows(block, task['cik'], task['prediction_date'], task['evidence_cutoff']):
            if row['filing_type'] in FORMS and task['prediction_date'] < row['filing_date'] <= task['evidence_cutoff']:
                accession = row['accession_number']
                if accession in rows and rows[accession] != row:
                    raise ValueError('Conflicting accession metadata')
                rows[accession] = row
    if not loaded:
        raise ValueError('No submission evidence')
    return sorted(rows.values(),key=lambda r:(r['filing_date'],r['accession_number'])), sorted(declared-loaded), record.get('reporting_followup_complete') is True


def scan(payload):
    text = visible_text(payload)
    if len(text) < 100 or re.search('request rate threshold exceeded|undeclared automated tool|access denied',text,re.I):
        raise ValueError('Empty, blocked or insufficient filing text')
    matches = list(TERMS.finditer(text))
    return {'text_characters': len(text), 'keyword_matches':len(matches),
            'excerpts':[text[max(0,m.start()-250):m.end()+450] for m in matches[:12]],
            'excerpt_limit':12, 'automated_negative_approval':False}


def run(download=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    queue = Path('data/processed/validation/rapid_review_v1/queue.jsonl')
    tasks = [r for r in map(json.loads,queue.read_text().splitlines()) if 'random negative audit' in r['review_reasons']]
    reports=[]
    for task in tasks:
        result={'unit_id':task['unit_id'],'company_name':task['company_name'],'documents':[], 'errors':[], 'approved_label':None}
        try:
            docs, missing, followup = inventory(task)
            result.update(expected_documents=len(docs),missing_history_files=missing,reporting_followup=followup)
            for doc in docs:
                path=ROOT/'raw'/(task['cik']+'_'+doc['accession_number']+'.html')
                item={'accession':doc['accession_number'],'form':doc['filing_type'],'filed':doc['filing_date'],'url':doc['source_url']}
                try:
                    if download:
                        coordinated_fetch(doc['source_url'],path)
                    if not path.exists():
                        item['status']='not_downloaded'
                    else:
                        item.update(scan(path.read_bytes()),path=str(path),sha256=digest(path),status='screened')
                except Exception as exc:
                    item.update(status='error',error=type(exc).__name__)
                result['documents'].append(item)
        except Exception as exc:
            result['errors'].append(str(exc))
        result['screened_documents']=sum(d['status']=='screened' for d in result['documents'])
        result['flagged_documents']=sum(d.get('keyword_matches',0)>0 for d in result['documents'])
        result['automated_screen_complete']=bool(result.get('expected_documents')) and result['screened_documents']==result.get('expected_documents') and not result.get('missing_history_files') and not result['errors'] and result.get('reporting_followup') is True
        reports.append(result)
        (ROOT/'companies').mkdir(exist_ok=True)
        (ROOT/'companies'/(task['cik']+'.json')).write_text(json.dumps(result,indent=2))
        print(task['cik'],result['screened_documents'],'screened;',result['flagged_documents'],'flagged',flush=True)
    summary={'companies':len(reports),'documents':sum(len(r['documents']) for r in reports),'screened':sum(r['screened_documents'] for r in reports),'flagged':sum(r['flagged_documents'] for r in reports),'automated_screen_complete_companies':sum(r['automated_screen_complete'] for r in reports),'queue_sha256':digest(queue),'implementation_sha256':digest(Path(__file__)),'human_approvals':0,'production_approval':False,'limits':'Primary report bodies only, not all exhibits or court records. Keyword screening cannot establish absence. Human coverage assessment and interpretation remain required.'}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    body=['<html><meta charset="utf-8"><h1>60 negative cases: automated coverage screening</h1><p>Prediction year: 2025. Evidence cutoff: July 19, 2026. No human approvals. Primary report bodies only; exhibits are not screened. Keyword absence does not verify a negative.</p>']
    for r in reports:
        body.append('<details><summary>'+html.escape(r['company_name'])+f" — {r['screened_documents']} screened, {r['flagged_documents']} flagged</summary><pre>"+html.escape(json.dumps({k:v for k,v in r.items() if k!='documents'},indent=2))+'</pre>')
        for d in r['documents']:
            body.append('<p><a href="'+html.escape(d['url'],quote=True)+'">'+html.escape(d['form']+' '+d['filed'])+'</a> — '+d['status']+'</p>')
            for excerpt in d.get('excerpts',[]):
                body.append('<blockquote>'+html.escape(excerpt)+'</blockquote>')
        body.append('</details>')
    (ROOT/'review.html').write_text('\n'.join(body)+'</html>')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    print(json.dumps(run(parser.parse_args().download),indent=2))
