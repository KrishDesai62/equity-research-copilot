from hashlib import sha256
import json
from pathlib import Path
from threading import Event

import pytest
from src.agents.qa_agent import answer_question
from src.web.company_imports import CompanyImports, normalize_ticker, normalize_filings, choose_reports


@pytest.mark.parametrize('ticker',['../AAPL','AAPL/../../x','https://sec.gov','AAPL?x','',None,'x'*21])
def test_import_rejects_paths_and_invalid_tickers(ticker):
    with pytest.raises(ValueError): normalize_ticker(ticker)


def test_ticker_alias_and_amended_forms_are_preserved():
    assert normalize_ticker(' brk.b ') == 'BRK-B'
    block=dict(form=['10-K/A','6-K','4'],accessionNumber=['0000000123-25-000001','0000000123-25-000002','0000000123-25-000003'],filingDate=['2025-01-01']*3,primaryDocument=['report.htm']*3)
    rows=normalize_filings(block,'0000000123','TEST','2025-12-31')
    assert [r['filing_type'] for r in rows]==['10-K/A','6-K']
    assert len(choose_reports(rows))==2
    assert normalize_filings(block,'0000000123','TEST','2024-12-31')==[]


def manager(tmp_path, *, fail=False, gate=None):
    frontend=tmp_path/'frontend';(frontend/'data').mkdir(parents=True)
    (frontend/'data/catalog.json').write_text(json.dumps(dict(companies=[],totals={},snapshot_date='2026-10-16')))
    def fetch(url,path):
        if gate is not None and url.endswith('company_tickers.json'): gate.wait(timeout=10)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if url.endswith('company_tickers.json'):
            payload=json.dumps({'0':dict(cik_str=123,ticker='TEST',title='Test Company')})
        elif url.endswith('submissions/CIK0000000123.json'):
            payload=json.dumps(dict(cik='123',name='Test Company',filings=dict(files=[],recent=dict(form=['10-K'],accessionNumber=['0000000123-25-000001'],filingDate=['2025-01-01'],primaryDocument=['report.htm']))))
        elif 'companyfacts' in url:
            payload=json.dumps(dict(cik=123,facts={}))
        else:
            if fail: raise ConnectionError('Network failed with a private internal detail')
            payload='<html><body><p>'+'The company maintains liquidity through cash reserves and operating cash flows. '*20+'</p></body></html>'
        path.write_text(payload)
        return dict(url=url,path=str(path),sha256=sha256(path.read_bytes()).hexdigest())
    return CompanyImports(root=tmp_path/'imports',frontend=frontend,fetcher=fetch)


def test_import_publishes_cited_searchable_company_without_modifying_base(tmp_path):
    service=manager(tmp_path);base=(service.frontend/'data/catalog.json').read_bytes()
    job=service.submit('test');service.executor.shutdown(wait=True)
    assert service.status(job['id'])['status']=='ready'
    assert (service.frontend/'data/catalog.json').read_bytes()==base
    payload=service.payload('TEST');assert payload['company']['filing_count']==1
    assert payload['company']['latest'] is None
    assert payload['import_info']['warnings']
    result=answer_question('liquidity','TEST',as_of='2025-12-31',retriever=service.retriever('TEST'))
    assert result['status']=='evidence_found'
    assert result['citations'][0]['source_url'].startswith('https://www.sec.gov/Archives/edgar/data/123/')
    assert not answer_question('liquidity','TEST',as_of='2024-12-31',retriever=service.retriever('TEST'))['citations']
    assert service.catalog()['totals']==dict(companies=1,filings=1,passages=1)
    assert service.submit('TEST')['status']=='ready'
    assert service.status('../escape') is None


def test_failed_import_never_publishes_a_partial_company(tmp_path):
    service=manager(tmp_path,fail=True);job=service.submit('TEST');service.executor.shutdown(wait=True)
    status=service.status(job['id']);assert status['status']=='failed'
    assert service.payload('TEST') is None
    assert not service.catalog()['companies']
    assert 'private internal detail' not in status['message']


def test_unknown_sec_symbol_does_not_create_a_company(tmp_path):
    service=manager(tmp_path);job=service.submit('UNKNOWN');service.executor.shutdown(wait=True)
    assert service.status(job['id'])['status']=='failed'
    assert 'not found' in service.status(job['id'])['message']
    assert service.payload('UNKNOWN') is None


def test_concurrent_duplicate_import_reuses_the_same_job(tmp_path):
    gate=Event();service=manager(tmp_path,gate=gate)
    try:
        first=service.submit('TEST');second=service.submit('test')
        assert first['id']==second['id']
    finally:
        gate.set();service.executor.shutdown(wait=True)


def test_failed_refresh_preserves_previously_published_company(tmp_path):
    service=manager(tmp_path)
    first=service.submit('TEST')
    # Wait for the initial import without closing the reusable worker.
    service.executor.submit(lambda: None).result(timeout=10)
    original=(service.root/'companies/TEST.json').read_bytes()
    def broken_fetch(*args): raise ConnectionError('failed refresh')
    service.fetch=broken_fetch
    refresh=service.submit('TEST',refresh=True)
    service.executor.shutdown(wait=True)
    assert service.status(refresh['id'])['status']=='failed'
    assert (service.root/'companies/TEST.json').read_bytes()==original
    assert service.payload('TEST')['company']['filing_count']==1


def test_refresh_creates_new_version_without_rewriting_old_sources(tmp_path):
    service=manager(tmp_path);first=service.submit('TEST')
    service.executor.submit(lambda: None).result(timeout=10)
    pointer=json.loads((service.root/'companies/TEST.json').read_text())
    old=json.loads((service.root/'versions'/pointer['version']/'manifest.json').read_text())
    refresh=service.submit('TEST',refresh=True);service.executor.shutdown(wait=True)
    assert service.status(refresh['id'])['status']=='ready'
    assert json.loads((service.root/'companies/TEST.json').read_text())['version']!=pointer['version']
    assert all(sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256'] for s in old['sources'])


def test_base_retrieval_bootstraps_from_public_frontend_export(tmp_path):
    service=manager(tmp_path)
    company=dict(ticker='BASE',cik='456',name='Base Company',filing_count=1,passage_count=1)
    (service.frontend/'data/companies').mkdir()
    (service.frontend/'data/companies/BASE.json').write_text(json.dumps(dict(company=company,passages=[
        dict(chunk_id='base-1',text='Revenue growth supported cash liquidity.',section='risk_factors',
             filing_date='2025-01-01',filing_type='10-K',accession_number='0000000456-25-000001',
             source_url='https://www.sec.gov/source')])))
    (service.frontend/'data/catalog.json').write_text(json.dumps(dict(companies=[company],totals={})))
    retriever=service.base_retriever()
    assert retriever('liquidity',ticker='BASE',k=1)[0]['metadata']['cik']=='456'
    assert not retriever('liquidity',ticker='UNKNOWN',k=1)
    service.executor.shutdown()
