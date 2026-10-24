import pytest
from src.web.accounts import Accounts, Denied

@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.setenv('LEDGER_AUTH_REQUIRED','false')
    a=Accounts();a.dsn=str(tmp_path/'accounts.db');a.migrate()
    with a.db() as c:
        a.execute(c,'INSERT INTO users VALUES(?,?,?)',('alice','a@example.com',0))
        a.execute(c,'INSERT INTO users VALUES(?,?,?)',('bob','b@example.com',0))
    return a

def test_isolation_and_import_ownership(accounts):
    accounts.data('alice',{'watchlist':['AAPL'],'notes':{'AAPL':{'text':'Private','date':'2026-10-23'}}})
    assert accounts.data('bob')=={'watchlist':[],'notes':{}}
    assert accounts.own('alice','job',True)
    assert not accounts.own('bob','job')

def test_quota_does_not_reset_on_rejection(accounts):
    accounts.limit('alice',2,86400);accounts.limit('alice',2,86400)
    for _ in range(2):
        with pytest.raises(Denied) as e: accounts.limit('alice',2,86400)
        assert e.value.status==429

def test_session_expiration_logout_hashing(accounts):
    import time
    with accounts.db() as c:
        accounts.execute(c,'INSERT INTO sessions VALUES(?,?,?)',(accounts.digest('secret'),'alice',int(time.time())+100))
    assert accounts.user('secret')['id']=='alice'
    accounts.logout('secret')
    with pytest.raises(Denied):accounts.user('secret')

def test_saved_data_validation(accounts):
    with pytest.raises(Denied):accounts.data('alice',{'notes':{'AAPL':{'text':'x'*4001,'date':'2026-10-23'}}})
    with pytest.raises(Denied):accounts.data('alice',{'user_id':'bob'})

def test_public_mode_fails_closed(monkeypatch):
    monkeypatch.setenv('LEDGER_AUTH_REQUIRED','true')
    monkeypatch.delenv('DATABASE_URL',raising=False)
    with pytest.raises(RuntimeError):Accounts()

def test_oauth_state_rejects_mismatch(accounts):
    with pytest.raises(Denied):accounts.callback('code','state','other')

def test_http_enforces_login_origin_and_private_imports(accounts):
    import json
    import threading
    import urllib.request
    import urllib.error
    import time
    from functools import partial
    from http.server import ThreadingHTTPServer
    from src.web.server import Handler
    accounts.enabled=True;accounts.origin='https://ledger.example.com'
    with accounts.db() as c:
        accounts.execute(c,'INSERT INTO sessions VALUES(?,?,?)',(accounts.digest('session'),'alice',int(time.time())+100))
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory='frontend'))
    server.accounts=accounts;server.work_slots=threading.BoundedSemaphore(2)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    root='http://127.0.0.1:'+str(server.server_port)
    def request(path,headers=None,body=None):
        req=urllib.request.Request(root+path,headers=headers or {},data=body)
        try:
            with urllib.request.urlopen(req) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    try:
        assert request('/api/me')[0]==401
        cookie={'Cookie':'ledger_session=session'}
        assert request('/api/me',cookie)[0]==200
        assert request('/api/imports/another-users-job',cookie)[0]==404
        assert request('/api/me',cookie,b'{}')[0]==403
        headers=dict(cookie,Origin=accounts.origin)
        assert request('/api/me',headers,b'{"watchlist":["MSFT"],"notes":{}}')[0]==200
        assert accounts.data('alice')['watchlist']==['MSFT']
        assert accounts.data('bob')['watchlist']==[]
    finally:
        server.shutdown();server.server_close();thread.join()
