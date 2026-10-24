"""Bounded loopback-only retrieval load check with temporary synthetic accounts.

Never imports filings, executes model inference, or calls AI/market providers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import secrets
import statistics
import time
from urllib.parse import urlparse
from uuid import uuid4
import requests
from src.web.accounts import Accounts


def run(base='http://127.0.0.1:8000',count=24,concurrency=2):
    if urlparse(base).hostname not in {'127.0.0.1','localhost'}:raise ValueError('Load check is restricted to loopback')
    if not 1<=count<=48 or not 1<=concurrency<=2:raise ValueError('Bounded check allows at most 48 requests and two workers')
    from src.web.providers import capabilities
    if capabilities()['ai_synthesis']:raise ValueError('Disable AI synthesis for this no-provider load check')
    accounts=Accounts()
    if not accounts.enabled:raise ValueError('Load check requires the configured account-backed server')
    users=['loadcheck-'+uuid4().hex for _ in range(4)];tokens=[secrets.token_urlsafe(32) for _ in users]
    records=[]
    try:
        with accounts.db() as conn:
            for user,token in zip(users,tokens):
                accounts.execute(conn,'INSERT INTO users VALUES(?,?,?)',(user,'synthetic-load@example.invalid',int(time.time())))
                accounts.execute(conn,'INSERT INTO sessions VALUES(?,?,?)',(accounts.digest(token),user,int(time.time())+600))
        def request(i):
            start=time.perf_counter()
            try:
                ticker=['AAPL','MSFT','AMZN'][i%3]
                response=requests.post(base+'/api/answer',headers={'Cookie':'ledger_session='+tokens[i%4],'Origin':accounts.origin},json={'ticker':ticker,'question':'What competition risks are disclosed?','as_of':'2026-10-24'},timeout=45)
                payload=response.json()
                valid=response.status_code==200 and all(c.get('cik') and c['filing_date']<='2026-10-24' and c.get('source_url','').startswith('https://www.sec.gov/') for c in payload.get('citations',[]))
                return {'status':response.status_code,'seconds':round(time.perf_counter()-start,4),'citations':len(payload.get('citations',[])),'response_valid':valid}
            except requests.RequestException:
                return {'status':'network_error','seconds':round(time.perf_counter()-start,4),'response_valid':False}
        warmup=[request(i) for i in range(3)]
        started=time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:records=list(pool.map(request,range(count)))
        elapsed=time.perf_counter()-started
        latency=sorted(r['seconds'] for r in records)
        return dict(scope='single_ec2_loopback_warm_retrieval',requests=count,concurrency=concurrency,warmup_requests=len(warmup),warmup_successes=sum(r['status']==200 for r in warmup),status_counts=dict(Counter(str(r['status']) for r in records)),valid_responses=sum(r['response_valid'] for r in records),p50_seconds=round(statistics.median(latency),4),p95_seconds=latency[min(len(latency)-1,int(.95*len(latency)))],elapsed_seconds=round(elapsed,4),requests_per_second=round(count/elapsed,3),limitation='Small warm-cache check on one server; not a public internet capacity or SLA benchmark. No SEC downloads, inference or generation performed.',records=records)
    finally:
        for user in users:
            # Never invoke Cognito identity deletion for synthetic DB-only users.
            with accounts.db() as conn:
                for table in ['notifications','preferences','import_owners','user_data','sessions']:
                    accounts.execute(conn,f'DELETE FROM {table} WHERE user_id=?',(user,))
                accounts.execute(conn,'DELETE FROM quotas WHERE bucket LIKE ?',('%:'+user+':%',))
                accounts.execute(conn,'DELETE FROM users WHERE id=?',(user,))

if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env')
    p=argparse.ArgumentParser();p.add_argument('--requests',type=int,default=24);p.add_argument('--concurrency',type=int,default=2);p.add_argument('--output',default='data/processed/evaluation/bounded_load_20261024.json');args=p.parse_args()
    result=run(count=args.requests,concurrency=args.concurrency);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
