import json
import time
from pathlib import Path
import pytest
from cryptography.fernet import Fernet
from src.web.accounts import Accounts,Denied
from src.web.providers import synthesize,market
from src.web.backups import backup
from src.web.filing_monitor import run_once
from src.rag.local_index import build_index,LocalIndex
from src.rag.hybrid import wrap

@pytest.fixture
def accounts(tmp_path,monkeypatch):
    monkeypatch.setenv('LEDGER_AUTH_REQUIRED','false')
    a=Accounts();a.dsn=str(tmp_path/'accounts.sqlite');a.migrate()
    with a.db() as c:
        for name in ['alice','bob']:a.execute(c,'INSERT INTO users VALUES(?,?,?)',(name,name+'@example.com',0))
    return a


def test_account_preferences_usage_delete_and_export_data(accounts):
    accounts.data('alice',{'watchlist':['AAPL'],'notes':{}})
    assert accounts.preferences('alice',{'filing_alerts':True})['filing_alerts']
    assert not accounts.preferences('bob')['filing_alerts']
    accounts.limit('daily:alice',200,86400)
    assert next(r for r in accounts.usage('alice') if r['name']=='daily')['used']==1
    accounts.erase('alice')
    with accounts.db() as c:
        assert accounts.execute(c,'SELECT 1 FROM users WHERE id=?',('alice',)).fetchone() is None
        assert accounts.execute(c,'SELECT 1 FROM users WHERE id=?',('bob',)).fetchone()
        assert accounts.execute(c,'SELECT 1 FROM deleted_accounts WHERE digest=?',(accounts.digest('alice'),)).fetchone()


def test_backup_encrypted_and_excludes_sessions(accounts,tmp_path,monkeypatch):
    key=Fernet.generate_key();monkeypatch.setenv('LEDGER_BACKUP_KEY',key.decode())
    path=backup(accounts,tmp_path/'backups')
    raw=path.read_bytes();assert b'alice@example.com' not in raw
    data=json.loads(Fernet(key).decrypt(raw))
    assert 'sessions' not in data['tables']
    assert len(data['tables']['users']['rows'])==2
    assert path.stat().st_mode&0o077==0


def test_optional_providers_honestly_report_missing_keys(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY',raising=False);monkeypatch.delenv('FINNHUB_API_KEY',raising=False)
    assert synthesize('question',[])['status']=='not_configured'
    assert market('AAPL')['status']=='not_configured'


def test_ai_rejects_invented_citation(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY','test');monkeypatch.setenv('ANTHROPIC_MODEL','test')
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'content':[{'type':'text','text':json.dumps({'statements':[{'text':'Invented','source_ids':[99],'quote':'Not in any source text'}]})}]}
    monkeypatch.setattr('src.web.providers.requests.post',lambda *a,**k:Response())
    assert synthesize('question',[{'id':1,'excerpt':'Cash increased over the reported period.'}])['status']=='unavailable'


def test_hybrid_respects_dates_and_companies(tmp_path):
    rows=[]
    for i in range(5):
        rows.append(dict(chunk_id=str(i),ticker='TEST',filing_type='10-K',filing_date='2025-01-01',section='risk_factors',text=['Liquidity cash debt financing obligations','Revenue sales margins operations','Cash flow capital funding liquidity','Competition risk competitors business','Debt credit financing maturity'][i],source_url='https://www.sec.gov/test',accession_number=str(i)))
    rows.append(dict(rows[0],chunk_id='future',filing_date='2027-01-01'))
    source=tmp_path/'chunks.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows));dest=tmp_path/'index.sqlite';build_index(source,dest)
    retrieve=wrap(LocalIndex(dest).retrieve)
    result=retrieve('funding cash',ticker='TEST',date_range=('2020-01-01','2026-01-01'),k=3)
    assert result and all(r['filing_date']<='2026-01-01' for r in result)
    assert not retrieve('cash',ticker='UNKNOWN',date_range=('2020-01-01','2026-01-01'),k=3)


def test_monitor_baseline_and_deduplicated_private_alerts(accounts,tmp_path):
    accounts.data('alice',{'watchlist':['AAPL'],'notes':{}});accounts.preferences('alice',{'filing_alerts':True})
    class Imports:
        root=tmp_path/'imports'
        def catalog(self):return {'companies':[{'ticker':'AAPL','cik':'0000320193'}]}
    imports=Imports();count=[1]
    def fetcher(url,path):
        n=count[0]
        path.write_text(json.dumps({'filings':{'recent':{'form':['10-K']*n,'accessionNumber':['0000320193-25-'+str(i).zfill(6) for i in range(n)],'filingDate':[__import__('datetime').date.today().isoformat()]*n,'reportDate':['2024-12-31']*n,'primaryDocument':['aapl.htm']*n}}}))
    assert run_once(accounts,imports,fetcher=fetcher)['new_reports']==0
    count[0]=2
    assert run_once(accounts,imports,fetcher=fetcher)['new_reports']==1
    assert len(accounts.alerts('alice'))==1
    assert not accounts.alerts('bob')
    assert run_once(accounts,imports,fetcher=fetcher)['new_reports']==0
    assert accounts.alerts('alice',read=True)[0]['read']


def test_market_quote_can_survive_unavailable_news_or_history(monkeypatch):
    monkeypatch.setenv('FINNHUB_API_KEY','test')
    from src.web.providers import _market
    _market.cache_clear()
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'c':100.0,'t':1700000000}
    def get(url,**kwargs):
        if url.endswith('/quote'):return Response()
        raise RuntimeError('Provider plan does not support endpoint')
    monkeypatch.setattr('src.web.providers.requests.get',get)
    result=market('TEST')
    assert result['status']=='available' and result['quote']['c']==100
    assert not result['history'] and not result['metrics'] and len(result['warnings'])==3
    _market.cache_clear()


def test_ai_valid_draft_keeps_source_ids_and_gated_context(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY','test');monkeypatch.setenv('ANTHROPIC_MODEL','test-model')
    captured={}
    quote='Cash increased over the reported period.'
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'content':[{'type':'text','text':json.dumps({'statements':[{'text':'The filing reports increased cash.','source_ids':[1],'quote':quote}]})}]}
    def post(*args,**kwargs):captured.update(kwargs['json']);return Response()
    monkeypatch.setattr('src.web.providers.requests.post',post)
    result=synthesize('Explain cash',[{'id':1,'excerpt':quote}],model_context={'status':'withheld','score':None})
    assert result['status']=='draft' and result['statements'][0]['source_ids']==[1]
    assert json.loads(captured['messages'][0]['content'])['model_context']['score'] is None
