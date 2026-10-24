"""Serve Ledger and its existing extractive retrieval API on the same origin."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs
from http.cookies import SimpleCookie
import logging
import threading
from src.web.accounts import Accounts, Denied

FRONTEND = Path(__file__).resolve().parents[2] / 'frontend'


def validate_question(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a JSON object')
    question = payload.get('question', '')
    ticker = payload.get('ticker', '')
    as_of = payload.get('as_of', '')
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
        raise ValueError('Enter a question of 1–500 characters')
    if not isinstance(ticker, str) or not 1 <= len(ticker) <= 20:
        raise ValueError('A company ticker is required')
    from src.web.company_imports import normalize_ticker
    ticker = normalize_ticker(ticker)
    from datetime import date
    if not isinstance(as_of, str):
        raise ValueError('A cutoff date is required')
    as_of=date.fromisoformat(as_of).isoformat()
    from src.web.company_imports import today
    if not '1900-01-01' <= as_of <= today():
        raise ValueError('Cutoff date must be a historical or current date')
    return dict(question=question.strip(), ticker=ticker.strip().upper(), as_of=as_of, k=3)


class Handler(SimpleHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def end_headers(self):
        path=urlparse(self.path).path
        if path=='/' or path.endswith(('.html','.js','.css')):
            self.send_header('Cache-Control','no-cache, max-age=0, must-revalidate')
        for key, value in {
            'X-Content-Type-Options':'nosniff',
            'Referrer-Policy':'same-origin',
            'X-Frame-Options':'DENY',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        }.items(): self.send_header(key,value)
        super().end_headers()

    def log_message(self, fmt, *args):
        # Never log OAuth codes, cookies, request bodies or database credentials.
        logging.info('%s %s', self.command, urlparse(self.path).path)

    def cookies(self):
        try:
            c=SimpleCookie(); c.load(self.headers.get('Cookie',''))
            return {k:v.value for k,v in c.items()}
        except Exception: return {}

    def redirect(self, url, cookie=None):
        self.send_response(302)
        self.send_header('Location',url)
        self.send_header('Cache-Control','no-store')
        if cookie: self.send_header('Set-Cookie',cookie)
        self.end_headers()

    def do_GET(self):
        return self.dispatch(False)

    def do_POST(self):
        return self.dispatch(True)

    def dispatch(self, post):
        accounts=getattr(self.server,'accounts',None)
        path=urlparse(self.path).path
        self.account_user=None
        try:
            if accounts and accounts.enabled:
                # The application trusts the socket peer only; nginx applies real-IP limits.
                accounts.limit('peer:'+self.client_address[0],1200,60)
                cookies=self.cookies()
                if path=='/api/session' and not post:
                    try: user=accounts.user(cookies.get('ledger_session',''))
                    except Denied: user=None
                    return self.json_response(200,{'required':True,'user':user,'logout_url':accounts.domain+'/logout?'+__import__('urllib.parse',fromlist=['urlencode']).urlencode({'client_id':accounts.client,'logout_uri':accounts.origin+'/'})})
                if path=='/auth/login' and not post:
                    state,url=accounts.login()
                    return self.redirect(url,'ledger_oauth='+state+'; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=600')
                if path=='/auth/callback' and not post:
                    q=parse_qs(urlparse(self.path).query)
                    token=accounts.callback(q.get('code',[''])[0],q.get('state',[''])[0],cookies.get('ledger_oauth',''))
                    return self.redirect('/','ledger_session='+token+'; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=28800')
                if path.startswith('/api/') and path!='/api/health':
                    self.account_user=accounts.user(cookies.get('ledger_session',''))
                    if post and self.headers.get('Origin')!=accounts.origin: raise Denied(403,'Request origin rejected')
                    if path=='/api/logout' and post:
                        accounts.logout(cookies.get('ledger_session',''))
                        return self.redirect(accounts.domain+'/logout?'+__import__('urllib.parse',fromlist=['urlencode']).urlencode({'client_id':accounts.client,'logout_uri':accounts.origin+'/'}),'ledger_session=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0')
                    user_id=self.account_user['id']
                    if path=='/api/account/export' and not post:
                        return self.json_response(200,dict(profile=self.account_user,research=accounts.data(user_id),preferences=accounts.preferences(user_id),notifications=accounts.alerts(user_id)))
                    if path=='/api/account/usage' and not post:
                        return self.json_response(200,{'usage':accounts.usage(user_id)})
                    if path in {'/api/account/preferences','/api/account/delete','/api/notifications'}:
                        body={}
                        if post:
                            accounts.limit('account:'+user_id,30,60)
                        if post:
                            length=int(self.headers.get('Content-Length','0'))
                            if not 0<length<=1024:raise Denied(400,'Invalid account request')
                            body=json.loads(self.rfile.read(length))
                        if path=='/api/account/delete':
                            if not post or body!={'confirmation':'DELETE'}:raise Denied(400,'Confirm account deletion')
                            identity_deleted=accounts.erase(user_id)
                            return self.json_response(200,{'status':'account_deleted' if identity_deleted else 'application_account_deleted','identity_status':'deleted' if identity_deleted else 'Cognito login identity remains; contact site owner for identity deletion. Re-entry is blocked.'})
                        if path=='/api/account/preferences':
                            return self.json_response(200,accounts.preferences(user_id,body if post else None))
                        return self.json_response(200,{'notifications':accounts.alerts(user_id,read=post)})
                    if path=='/api/me':
                        payload=None
                        if post:
                            length=int(self.headers.get('Content-Length','0'))
                            if not 0<length<=500000: raise Denied(413,'Saved data is too large')
                            payload=json.loads(self.rfile.read(length))
                        return self.json_response(200,accounts.data(self.account_user['id'],payload))
                    if post:
                        user=self.account_user['id']
                        accounts.limit('requests:'+user,20,60)
                        accounts.limit('daily:'+user,200,86400)
                        if path=='/api/imports': accounts.limit('imports:'+user,5,86400)
                    if path.startswith('/api/imports/') and not accounts.own(self.account_user['id'],path.split('/')[-1]): raise Denied(404,'Import not found')
            elif path=='/api/session':
                return self.json_response(200,{'required':False,'user':None})
            if post:
                if not self.server.work_slots.acquire(blocking=False): raise Denied(429,'Server is busy. Try again shortly.')
                try: return self.handle_post()
                finally: self.server.work_slots.release()
            return self.handle_get()
        except Denied as exc:
            return self.json_response(exc.status,{'error':exc.message})
        except (ValueError,TypeError):
            return self.json_response(400,{'error':'Invalid request'})
        except Exception:
            logging.error('Request failed; inspect service configuration and database availability')
            return self.json_response(503,{'error':'Service temporarily unavailable'})

    def json_response(self, status, value):
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(encoded)

    def handle_get(self):
        path = urlparse(self.path).path
        imports = self.server.imports
        if path == '/config.json':
            return self.json_response(200, {'apiBase': '', 'imports': True, 'research': True, 'capabilities': __import__('src.web.providers',fromlist=['capabilities']).capabilities()})
        if path == '/api/capabilities':
            from src.web.providers import capabilities
            return self.json_response(200,capabilities())
        if path == '/api/model':
            return self.json_response(200, self.server.research.model_status())
        if path == '/data/catalog.json':
            return self.json_response(200, imports.catalog())
        if path.startswith('/data/companies/') and path.endswith('.json'):
            try:
                payload = imports.payload(unquote(path.removeprefix('/data/companies/').removesuffix('.json')))
            except ValueError:
                return self.json_response(400, {'error': 'Invalid ticker'})
            if payload is not None:
                return self.json_response(200, payload)
        if path.startswith('/api/imports/'):
            job = imports.status(path.removeprefix('/api/imports/'))
            return self.json_response(200, job) if job else self.json_response(404, {'error': 'Import not found'})
        if path == '/api/health':
            accounts=getattr(self.server,'accounts',None)
            if accounts and accounts.enabled:
                try:
                    with accounts.db() as conn: accounts.execute(conn,'SELECT 1').fetchone()
                except Exception:return self.json_response(503,{'status':'database_unavailable'})
            return self.json_response(200, {'status': 'ok', 'mode': 'hybrid_retrieval'})
        return super().do_GET()

    def handle_post(self):
        path = urlparse(self.path).path
        if path not in {'/api/answer', '/api/imports', '/api/risk', '/api/research', '/api/changes', '/api/market'}:
            return self.json_response(404, {'error': 'Unknown endpoint'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                return self.json_response(413, {'error': 'Request body is too large or empty'})
            payload = json.loads(self.rfile.read(length))
            if path == '/api/imports':
                if not isinstance(payload,dict) or not isinstance(payload.get('refresh',False),bool) or not isinstance(payload.get('history',False),bool):raise ValueError('Invalid import options')
                job = self.server.imports.submit(payload.get('ticker') if isinstance(payload, dict) else None,
                    refresh=payload.get('refresh',False) if isinstance(payload,dict) else False,history=payload.get('history',False))
                if self.account_user and job.get('id'): self.server.accounts.own(self.account_user['id'],job['id'],create=True)
                return self.json_response(200 if job['status']=='ready' else 202, job)
            if path in {'/api/risk','/api/research','/api/changes','/api/market'}:
                if not isinstance(payload,dict):raise ValueError('Expected JSON object')
                arguments=validate_question(dict(payload,question='research'))
            else:
                arguments = validate_question(payload)
        except (ValueError, TypeError):
            return self.json_response(400, {'error': 'Enter a valid ticker' if path == '/api/imports' else 'Provide a ticker, a valid cutoff date, and a question of 1–500 characters'})
        except RuntimeError as exc:
            return self.json_response(429, {'error': str(exc)})
        try:
            if path in {'/api/risk','/api/research','/api/changes','/api/market'}:
                if path=='/api/market':
                    from src.web.providers import market
                    return self.json_response(200,market(arguments['ticker']))
                method={'/api/risk':'risk','/api/research':'memo','/api/changes':'changes'}[path]
                result=getattr(self.server.research,method)(arguments['ticker'],arguments['as_of'])
                return self.json_response(200,result)
            from src.agents.qa_agent import answer_question
            retriever = self.server.imports.retriever(arguments['ticker']) or self.server.imports.base_retriever()
            result = answer_question(**arguments, **({'retriever': retriever} if retriever else {}))
            from src.web.providers import synthesize
            result['synthesis']=synthesize(arguments['question'],result['citations'])
        except ValueError:
            return self.json_response(422, {'error':'The company sources or requested date could not be verified.'})
        except Exception:
            return self.json_response(503, {'error': 'Retrieval is unavailable. Check that the local filing index is installed.'})
        return self.json_response(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--import-root', default='data/processed/company_imports',
                        help='Persistent directory for imported company versions, SEC caches and job status')
    parser.add_argument('--research-scores',action='store_true',
                        help='Enable explicitly unvalidated research scores; no alert categories or production approval')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    accounts=Accounts()
    if accounts.enabled: accounts.migrate()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(FRONTEND)))
    server.accounts=accounts
    server.work_slots=threading.BoundedSemaphore(2)
    from src.web.company_imports import CompanyImports
    server.imports = CompanyImports(root=args.import_root, frontend=FRONTEND)
    from src.web.research_service import ResearchService
    server.research = ResearchService(server.imports,allow_research=args.research_scores)
    stop=threading.Event()
    if accounts.enabled:
        from src.web.filing_monitor import start
        start(accounts,server.imports,stop)
    print(f'Ledger: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
