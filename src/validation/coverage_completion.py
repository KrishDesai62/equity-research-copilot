"""Reconcile the reviewed sample and screen complete submissions, preserving approvals."""
import argparse
import json
from pathlib import Path
import re
import binascii
import base64
from io import BytesIO
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.validation.cohort_v3_data import digest
from src.validation.public_source_fetch import coordinated_fetch
from src.validation.sampled_negative_coverage import FORMS, TERMS
from src.validation.filing_text_signals import visible_text

ROOT = Path('data/processed/validation/coverage_completion_20261023')
COVERAGE = Path('data/processed/validation/sampled_negative_coverage_v1')
INTAKE = Path('data/processed/validation/human_review_intake')


def decode_uu_pdf_line(line):
    """Honor uuencoded byte count even when unused padding bits are nonzero."""
    try:
        return binascii.a2b_uu(line)
    except binascii.Error:
        count=(line[0]-32)&63
        encoded=line[1:1+4*((count+2)//3)]
        alphabet=b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        if len(encoded)!=4*((count+2)//3):raise ValueError('Truncated PDF encoding')
        return base64.b64decode(bytes(alphabet[(c-32)&63] for c in encoded))[:count]


def index_rows(text):
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) == 5 and fields[0].strip().isdigit():
            cik, name, form, date, filename = fields
            if form in FORMS and '2024-12-31' < date <= '2026-07-19':
                yield dict(cik=cik.strip().zfill(10), company=name, form=form,
                           filed=date, filename=filename,
                           accession=Path(filename).stem)


def screen_submission(payload):
    text = payload.decode('utf-8', errors='replace')
    documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', text, re.S | re.I)
    if not documents:
        raise ValueError('No SGML documents in complete submission')
    results = []
    for block in documents:
        def field(name):
            match = re.search(r'<' + name + r'>([^\r\n<]+)', block, re.I)
            return match.group(1).strip() if match else ''
        form, filename = field('TYPE'), field('FILENAME')
        if not form.upper().startswith('EX-') and form not in FORMS:
            continue
        body = re.search(r'<TEXT>(.*?)</TEXT>', block, re.S | re.I)
        item = dict(type=form, filename=filename, content_sha256=sha256(block.encode()).hexdigest(),
                    scope='exhibit' if form.upper().startswith('EX-') else 'primary_report')
        if not body or not filename.lower().endswith(('.htm', '.html', '.txt', '.xml', '.xsd', '.pdf')):
            item['status'] = 'unsupported_format_requires_review'
        else:
            try:
                if filename.lower().endswith('.pdf'):
                    from pypdf import PdfReader
                    encoded=re.search(r'begin [0-7]{3} [^\r\n]+\r?\n(.*?)\r?\nend',body.group(1),re.S)
                    if not encoded:raise ValueError('Missing PDF uuencoding')
                    data=b''.join(decode_uu_pdf_line(line.encode()) for line in encoded.group(1).splitlines() if line)
                    reader=PdfReader(BytesIO(data))
                    visible=' '.join((page.extract_text() or '') for page in reader.pages)
                    if not visible.strip():raise ValueError('PDF requires OCR')
                    item['pdf_pages']=len(reader.pages)
                else:
                    visible = visible_text(body.group(1).encode())
                matches = list(TERMS.finditer(visible))
                item.update(status='screened', keyword_matches=len(matches),
                            excerpts=[visible[max(0,m.start()-250):m.end()+450] for m in matches])
            except ValueError:
                item['status'] = 'unreadable_text_requires_review'
        results.append(item)
    return results


def run(download=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    paired = json.loads((INTAKE/'sampled_negative_60_paired_status_20261023.json').read_text())
    first = json.loads((INTAKE/'sampled_negative_60_assessment_20261023.json').read_text())
    companies = {}
    for name, expected in first['company_evidence_sha256'].items():
        path = COVERAGE/'companies'/name
        if digest(path) != expected:
            raise ValueError('Reviewed evidence changed: '+name)
        companies[path.stem] = json.loads(path.read_text())
    indexed, errors, proofs = {}, [], []
    for year, quarter in [(2025,q) for q in range(1,5)]+[(2026,q) for q in range(1,4)]:
        url=f'https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx'
        path=ROOT/'indexes'/f'{year}_QTR{quarter}.idx'
        try:
            if download: coordinated_fetch(url,path)
            text=path.read_text()
            if 'CIK|Company Name|Form Type|Date Filed|Filename' not in text:
                raise ValueError('Invalid SEC master index')
            proofs.append(dict(url=url,path=str(path),sha256=digest(path)))
            for row in index_rows(text):
                if row['cik'] in companies: indexed[(row['cik'],row['accession'])]=row
        except Exception as exc:
            errors.append(dict(url=url,error=type(exc).__name__))
    statuses=[]
    work=[]
    for cik, company in sorted(companies.items()):
        saved={d['accession'] for d in company['documents']}
        official={a for c,a in indexed if c==cik}
        result=dict(cik=cik,unit_id=company['unit_id'],
                    missing_from_reviewed_inventory=sorted(official-saved),
                    absent_from_master_index=sorted(saved-official),documents=[])
        statuses.append(result)
        work.extend((accession not in saved,cik,accession,result) for accession in sorted(official | saved))
    def acquire(entry):
        _,cik,accession,_=entry
        url=f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace("-", "")}/{accession}.txt'
        path=ROOT/'submissions'/f'{cik}_{accession}.txt'
        try:
            if download: coordinated_fetch(url,path)
            exhibits=screen_submission(path.read_bytes())
            return dict(accession=accession,url=url,path=str(path),sha256=digest(path),exhibits=exhibits)
        except Exception as exc:
            return dict(accession=accession,url=url,error=type(exc).__name__)
    # Amendments missed by the previous inventory are acquired first.
    work.sort(key=lambda entry:(not entry[0],entry[1],entry[2]))
    (ROOT/'companies').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(acquire,entry):entry for entry in work}
        for number,future in enumerate(as_completed(futures),1):
            _,cik,accession,company=futures[future]
            company['documents'].append(future.result())
            company['documents'].sort(key=lambda d:d['accession'])
            destination=ROOT/'companies'/f'{cik}.json'
            temporary=destination.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(company,indent=2)+'\n')
            temporary.replace(destination)
            if number%10==0 or number==len(work):
                print(number,'/',len(work),'complete submissions processed',flush=True)
    exhibits=[e for c in statuses for d in c['documents'] for e in d.get('exhibits',[]) if e['scope']=='exhibit']
    result=dict(index_sources=proofs,index_errors=errors,companies=len(statuses),
                inventory_additions=sum(len(c['missing_from_reviewed_inventory']) for c in statuses),
                inventory_unmatched=sum(len(c['absent_from_master_index']) for c in statuses),
                submission_errors=sum('error' in d for c in statuses for d in c['documents']),
                exhibits=len(exhibits),flagged_exhibits=sum(e.get('keyword_matches',0)>0 for e in exhibits),
                unsupported_exhibits=sum(e['status']!='screened' for e in exhibits),
                paired_negative_assessments=paired['agreeing_negative_units'],
                human_exhibit_review_complete=False,final_evaluation_ready=False,production_approval=False)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    print(json.dumps(run(parser.parse_args().download),indent=2))
