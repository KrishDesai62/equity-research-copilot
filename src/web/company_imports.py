"""On-demand, isolated SEC research imports; never changes training or review data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from threading import Lock
from urllib.parse import quote
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.data_layer.research_dataset import submission_rows, write_json, write_jsonl
from src.rag.local_index import build_index, LocalIndex
from src.validation.public_source_fetch import coordinated_fetch

FIELDS = ['Revenues_ttm','Revenues_quarter','OperatingIncomeLoss_ttm','OperatingCashFlow_ttm',
          'NetIncomeLoss_ttm','Assets','Liabilities','Cash','current_ratio','liabilities_to_assets',
          'revenue_growth_yoy','operating_margin']
FORMS = {'10-K','10-K/A','10-Q','10-Q/A','8-K','8-K/A','20-F','20-F/A','40-F','40-F/A','6-K','6-K/A'}


def today():
    return datetime.now(ZoneInfo('America/Chicago')).date().isoformat()


def normalize_ticker(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9.-]{0,19}', value.strip()):
        raise ValueError('Enter a valid SEC ticker, such as TSLA or BRK-B')
    return value.strip().upper().replace('.', '-')


def normalize_filings(block, cik, ticker, cutoff, *, start='1900-01-01'):
    forms = block.get('form', [])
    expanded = dict(block, form=['8-K' if f in FORMS else '__skip__' for f in forms])
    originals = dict(zip(block.get('accessionNumber', []), forms))
    rows = submission_rows(expanded, cik, ticker, start, cutoff)
    for row in rows:
        row['filing_type'] = originals[row['accession_number']]
    return rows


def choose_reports(rows, *, history=False):
    if history:
        return sorted(rows,key=lambda r:(r['filing_date'],r['accession_number']))[-60:]
    groups = [{'10-K','10-K/A','20-F','20-F/A','40-F','40-F/A'},
              {'10-Q','10-Q/A'}, {'8-K','8-K/A','6-K','6-K/A'}]
    selected = []
    for group in groups:
        selected.extend(sorted((r for r in rows if r['filing_type'] in group),
                               key=lambda r:(r['filing_date'],r['accession_number']))[-2:])
    return sorted(selected, key=lambda r:(r['filing_date'],r['accession_number']))


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


class CompanyImports:
    def __init__(self, root='data/processed/company_imports', frontend='frontend', fetcher=coordinated_fetch):
        self.root, self.frontend, self.fetch = Path(root), Path(frontend), fetcher
        self.lock = Lock()
        self.index_lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sec-import')
        self.active = {}
        self.jobs = {}

    def payload(self, ticker):
        ticker = normalize_ticker(ticker)
        pointer = self.root/'companies'/f'{ticker}.json'
        if not pointer.exists():
            return None
        record = json.loads(pointer.read_text())
        return json.loads((self.root/'versions'/record['version']/'company.json').read_text())

    def catalog(self):
        base = json.loads((self.frontend/'data/catalog.json').read_text())
        entries = {c['ticker']:c for c in base['companies']}
        for pointer in sorted((self.root/'companies').glob('*.json')):
            payload = self.payload(pointer.stem)
            entries[payload['company']['ticker']] = payload['company']
        base['companies'] = sorted(entries.values(), key=lambda c:c['ticker'])
        base['totals'] = dict(companies=len(entries),filings=sum(c['filing_count'] for c in entries.values()),
                              passages=sum(c['passage_count'] for c in entries.values()))
        return base

    def submit(self, value, *, refresh=False, history=False):
        ticker = normalize_ticker(value)
        if type(refresh) is not bool or type(history) is not bool:
            raise ValueError('Refresh must be true or false')
        refresh=refresh or history
        with self.lock:
            if not refresh and any(c['ticker'].replace('.','-') == ticker for c in self.catalog()['companies']):
                return dict(status='ready',ticker=next(c['ticker'] for c in self.catalog()['companies'] if c['ticker'].replace('.','-')==ticker),message='Company is already available')
            if ticker in self.active:
                return dict(self.jobs[self.active[ticker]])
            if len(self.active) >= 8:
                raise RuntimeError('The import queue is full. Please try again after another import finishes.')
            job_id = uuid4().hex
            job = dict(id=job_id,ticker=ticker,status='queued',stage='Waiting to import',as_of=today(),warnings=[],refresh=refresh,history=history)
            self.jobs[job_id] = job
            self.active[ticker] = job_id
            write_json(self.root/'jobs'/f'{job_id}.json',job)
            self.executor.submit(self._run,job_id)
            return dict(job)

    def status(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}',job_id):
            return None
        with self.lock:
            if job_id in self.jobs:
                return dict(self.jobs[job_id])
        path = self.root/'jobs'/f'{job_id}.json'
        if not path.exists():
            return None
        job = json.loads(path.read_text())
        if job['status'] not in {'ready','failed'}:
            job.update(status='failed',stage='Import interrupted',message='The server restarted. Search the ticker and retry the import.')
        return job

    def _update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)
            write_json(self.root/'jobs'/f'{job_id}.json',self.jobs[job_id])

    def _download_json(self, url, path, sources):
        sources.append(self.fetch(url,path))
        return json.loads(path.read_text())

    def _run(self, job_id):
        ticker = self.jobs[job_id]['ticker']
        try:
            self._build(job_id)
        except Exception as exc:
            # Do not expose contact headers, filesystem paths or raw network exceptions.
            message = str(exc) if isinstance(exc, ValueError) else 'SEC import could not finish. Please retry; the downloaded files are cached.'
            self._update(job_id,status='failed',stage='Import could not finish',message=message)
        finally:
            with self.lock:
                self.active.pop(ticker,None)

    def _build(self, job_id):
        import shutil
        if shutil.disk_usage(self.root).free < 2 * 1024**3:
            raise ValueError('Import storage is nearly full. Please contact the site owner.')
        job = self.jobs[job_id]
        ticker, cutoff = job['ticker'], job['as_of']
        raw = self.root/'raw'/cutoff/ticker
        if job.get('refresh'):
            raw=raw/'refresh'/job_id
        version = self.root/'versions'/job_id
        version.mkdir(parents=True,exist_ok=True)
        sources, warnings = [], []
        self._update(job_id,status='running',stage='Finding the company in SEC records')
        mapping = self._download_json('https://www.sec.gov/files/company_tickers.json',
                                      self.root/'raw'/cutoff/'company_tickers.json',sources)
        matches = [r for r in mapping.values() if normalize_ticker(r['ticker'])==ticker]
        if not matches:
            raise ValueError('Ticker not found in the current SEC company directory. Check the symbol; historical or non-SEC issuers may not be listed.')
        if len({r['cik_str'] for r in matches}) != 1:
            raise ValueError('This ticker maps to multiple issuers and needs manual resolution.')
        company = matches[0]
        cik = str(company['cik_str']).zfill(10)
        self._update(job_id,stage='Reading the company filing history',company_name=company['title'],cik=cik)
        submissions = self._download_json(f'https://data.sec.gov/submissions/CIK{cik}.json',raw/'submissions.json',sources)
        if str(submissions.get('cik')).zfill(10) != cik:
            raise ValueError('SEC filing history returned a different issuer.')
        start = str(int(cutoff[:4])-5)+cutoff[4:]
        rows = normalize_filings(submissions.get('filings',{}).get('recent',{}),cik,ticker,cutoff,start=start)
        archives = sorted(submissions.get('filings',{}).get('files',[]),key=lambda f:f.get('filingTo',''),reverse=True)
        start = str(int(cutoff[:4])-5)+cutoff[4:]
        overlapping = [f for f in archives if f.get('filingTo','')>=start]
        for archive in overlapping[:5]:
            name = archive['name']
            if not re.fullmatch(r'CIK\d{10}-submissions-\d+\.json',name):
                raise ValueError('SEC archive filename could not be validated.')
            block = self._download_json(f'https://data.sec.gov/submissions/{name}',raw/name,sources)
            rows.extend(normalize_filings(block,cik,ticker,cutoff,start=start))
        if len(overlapping)>5:
            warnings.append('Import uses recent filings and five archived submission pages; older history is not exhaustive.')
        unique = {}
        for row in rows:
            key = row['accession_number']
            if key in unique and unique[key]!=row:
                raise ValueError('SEC returned conflicting filing metadata.')
            unique[key] = row
        rows = list(unique.values())
        selected = choose_reports(rows,history=job.get('history',False))
        if not selected:
            raise ValueError('No supported annual, quarterly, or current reports were found for this issuer.')
        self._update(job_id,stage='Downloading and indexing SEC reports',documents_total=len(selected),documents_done=0)
        from src.data_layer.filing_changes import extract_sections
        from src.validation.filing_text_signals import visible_text
        from src.rag.chunking import chunk_section
        chunks, filings = [], []
        for i,row in enumerate(selected):
            path = raw/(row['accession_number']+'.html')
            try:
                source = self.fetch(row['source_url'],path)
                payload = path.read_bytes()
                text = visible_text(payload)
                if len(text)<100 or re.search('request rate threshold exceeded|undeclared automated tool|access denied',text,re.I):
                    raise ValueError('SEC returned an unreadable report')
                form = row['filing_type']
                sections = extract_sections(payload.decode('utf-8-sig',errors='replace'),form) if form in {'10-K','10-Q'} else {}
                if not sections:
                    sections = {'full_report':text}
                for section,body in sections.items():
                    for index,chunk in enumerate(chunk_section(body)):
                        chunk_id = sha256(f"{cik}:{row['accession_number']}:{section}:{index}:{chunk}".encode()).hexdigest()
                        chunks.append({**row,'section':section,'chunk_id':chunk_id,'chunk_index':index,
                                       'text':chunk,'document_sha256':source['sha256']})
                sources.append(source)
                filings.append(row)
            except Exception:
                warnings.append(f"Could not index {row['filing_type']} filed {row['filing_date']}; coverage is partial.")
            self._update(job_id,documents_done=i+1,warnings=list(warnings))
        if not chunks:
            raise ValueError('The reports could not be indexed. No company data was published; retry the import.')
        self._update(job_id,stage='Extracting reported financials')
        history = []
        try:
            facts = self._download_json(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',raw/'companyfacts.json',sources)
            if str(facts.get('cik')).zfill(10)!=cik:
                raise ValueError('SEC financials returned a different issuer.')
            from src.ml.financial_history import extract_history
            frame,_,_ = extract_history(facts,ticker,public_filings=rows)
            by_period = {}
            metadata = {r['accession_number']:r for r in rows}
            for row in frame.sort_values('filing_date').to_dict('records'):
                if row['filing_date']>cutoff or finite(row.get('Revenues_ttm')) is None:
                    continue
                by_period[row['fiscal_period_end']] = dict(period=row['fiscal_period_end'],filed=row['filing_date'],
                    source_url=metadata[row['accession_number']]['source_url'],**{f:finite(row.get(f)) for f in FIELDS})
            history = sorted(by_period.values(),key=lambda r:r['period'])[-12:]
            if not history:
                warnings.append('Comparable USD financial history is unavailable. Filing research is available.')
        except ValueError as exc:
            if 'different issuer' in str(exc):
                raise
            warnings.append('Comparable USD financial history is unavailable. Filing research is available.')
        except Exception:
            warnings.append('Financial extraction is unavailable for this import. Filing research is available.')
        self._update(job_id,stage='Checking citations and publishing the company',warnings=list(warnings))
        write_jsonl(version/'chunks.jsonl',chunks)
        build_index(version/'chunks.jsonl',version/'filings.sqlite')
        index = LocalIndex(version/'filings.sqlite')
        if not index.retrieve(chunks[0]['text'][:200],ticker=ticker,k=1):
            raise ValueError('The imported retrieval index did not pass its citation check.')
        entry = dict(ticker=ticker,name=submissions.get('name') or company['title'],cik=cik,
                     passage_count=len(chunks),filing_count=len(filings),latest=history[-1] if history else None,
                     imported_on=cutoff,coverage='partial' if warnings else 'selected_reports')
        fields = ['chunk_id','text','section','filing_date','filing_type','source_url','accession_number']
        payload = dict(company=entry,history=history,filings=filings,
                       passages=[{key:c.get(key) for key in fields} for c in chunks],
                       import_info=dict(as_of=cutoff,warnings=warnings,selection=('Up to 60 supported reports from the last five years; primary bodies only, not exhaustive.' if job.get('history') else 'Up to two recent annual, two quarterly, and two current reports; primary bodies only.')))
        write_json(version/'company.json',payload)
        write_json(version/'manifest.json',dict(ticker=ticker,cik=cik,as_of=cutoff,sources=sources,
                   company_sha256=sha256((version/'company.json').read_bytes()).hexdigest(),
                   model_predictions_created=False,training_data_modified=False))
        write_json(self.root/'companies'/f'{ticker}.json',dict(version=job_id))
        self._update(job_id,status='ready',stage='Ready to research',message=f"Indexed {len(filings)} reports and {len(chunks)} passages.",warnings=warnings)

    def retriever(self, ticker):
        ticker = normalize_ticker(ticker)
        pointer = self.root/'companies'/f'{ticker}.json'
        if not pointer.exists():
            return None
        record = json.loads(pointer.read_text())
        from src.rag.hybrid import wrap
        return wrap(LocalIndex(self.root/'versions'/record['version']/'filings.sqlite').retrieve)

    def base_retriever(self):
        """Build a host-local index from the shipped public snapshot, if needed."""
        with self.index_lock:
            catalog=json.loads((self.frontend/'data/catalog.json').read_text())
            paths=[self.frontend/'data/companies'/f"{c['ticker']}.json" for c in catalog['companies']]
            corpus_hash=sha256(b''.join(sha256(p.read_bytes()).digest() for p in paths)).hexdigest()
            folder=self.root/'base_indexes'/corpus_hash
            destination=folder/'filings.sqlite'
            if not destination.exists():
                chunks=[]
                for path in paths:
                    payload=json.loads(path.read_text())
                    company=payload['company']
                    chunks.extend(dict(p,ticker=company['ticker'],cik=company['cik']) for p in payload['passages'])
                folder.mkdir(parents=True,exist_ok=True)
                write_jsonl(folder/'chunks.jsonl',chunks)
                build_index(folder/'chunks.jsonl',destination)
            from src.rag.hybrid import wrap
            return wrap(LocalIndex(destination).retrieve)
