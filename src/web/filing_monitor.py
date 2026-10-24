"""Poll followed companies for new SEC reports; store deduplicated in-app notices."""
import json
from datetime import date,timedelta
import logging
import threading
import time
from hashlib import sha256
from src.web.company_imports import normalize_filings, today
from src.validation.public_source_fetch import coordinated_fetch


def run_once(accounts, imports, *, fetcher=coordinated_fetch):
    with accounts.db() as conn:
        rows=accounts.execute(conn,'SELECT d.user_id,d.payload FROM user_data d JOIN preferences p ON p.user_id=d.user_id WHERE p.filing_alerts=1').fetchall()
    followers={}
    for user,raw in rows:
        for ticker in json.loads(raw).get('watchlist',[]):followers.setdefault(ticker,[]).append(user)
    companies={c['ticker']:c for c in imports.catalog()['companies']}
    state_path=imports.root/'monitor/state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    checked=events=0;errors=[]
    for ticker,users in sorted(followers.items()):
        if ticker not in companies:continue
        try:
            cik=str(companies[ticker]['cik']).zfill(10)
            path=imports.root/'monitor/cache'/str(int(time.time())//3600)/f'{cik}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            fetcher(f'https://data.sec.gov/submissions/CIK{cik}.json',path)
            data=json.loads(path.read_text())
            reports=normalize_filings(data['filings']['recent'],cik,ticker,today(),start=(date.fromisoformat(today())-timedelta(days=366)).isoformat())
            known=set(state.get(ticker,[]))
            # Initial baseline prevents a flood of historical notifications.
            if ticker in state:
                for report in reports:
                    accession=report['accession_number']
                    if accession in known:continue
                    payload={k:report[k] for k in ['ticker','filing_type','filing_date','source_url','accession_number']}
                    event=sha256((ticker+accession).encode()).hexdigest()
                    with accounts.db() as conn:
                        for user in users:
                            accounts.execute(conn,'INSERT INTO notifications VALUES(?,?,?,?,0) ON CONFLICT DO NOTHING',(event,user,json.dumps(payload),int(time.time())))
                    events+=1
            state[ticker]=[r['accession_number'] for r in reports]
            checked+=1
        except Exception:
            errors.append(ticker)
    state_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(state_path)
    status=dict(checked=checked,new_reports=events,failed_tickers=errors,checked_at=int(time.time()))
    (imports.root/'monitor/status.json').write_text(json.dumps(status))
    return status


def start(accounts, imports, stop):
    def worker():
        while not stop.is_set():
            try:run_once(accounts,imports)
            except Exception:logging.error('SEC scheduled check failed')
            stop.wait(3600)
    thread=threading.Thread(target=worker,name='filing-monitor',daemon=True);thread.start();return thread
