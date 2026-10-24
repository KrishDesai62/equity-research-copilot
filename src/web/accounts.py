"""Managed Cognito sign-in, opaque sessions and transactional user quotas."""
import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from urllib.parse import urlencode


class Denied(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Accounts:
    def __init__(self):
        self.enabled = os.getenv('LEDGER_AUTH_REQUIRED', 'false').lower() == 'true'
        self.origin = os.getenv('LEDGER_PUBLIC_ORIGIN', '').rstrip('/')
        self.domain = os.getenv('COGNITO_DOMAIN', '').rstrip('/')
        self.client = os.getenv('COGNITO_CLIENT_ID', '')
        self.issuer = os.getenv('COGNITO_ISSUER', '').rstrip('/')
        self.dsn = os.getenv('DATABASE_URL', '')
        if self.enabled and (not self.origin.startswith('https://') or not self.domain.startswith('https://') or not self.issuer.startswith('https://') or not self.client or not self.dsn.startswith('postgresql://')):
            raise RuntimeError('Public accounts require HTTPS, Cognito and PostgreSQL configuration')
        self.keys = None
        if self.enabled:
            import jwt
            self.keys = jwt.PyJWKClient(self.issuer + '/.well-known/jwks.json', timeout=5)

    @contextmanager
    def db(self):
        if self.dsn.startswith('postgresql://'):
            import psycopg
            conn = psycopg.connect(self.dsn, connect_timeout=5)
        else:
            conn = sqlite3.connect(self.dsn, timeout=10)
        try:
            if isinstance(conn, sqlite3.Connection): conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, conn, sql, args=()):
        return conn.execute(sql.replace('?', '%s') if not isinstance(conn, sqlite3.Connection) else sql, args)

    def migrate(self):
        with self.db() as conn:
            for sql in [
                'CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT NOT NULL, created BIGINT NOT NULL)',
                'CREATE TABLE IF NOT EXISTS sessions (digest TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), expires BIGINT NOT NULL)',
                'CREATE TABLE IF NOT EXISTS oauth_states (digest TEXT PRIMARY KEY, verifier TEXT NOT NULL, expires BIGINT NOT NULL)',
                'CREATE TABLE IF NOT EXISTS user_data (user_id TEXT PRIMARY KEY REFERENCES users(id), payload TEXT NOT NULL)',
                'CREATE TABLE IF NOT EXISTS quotas (bucket TEXT PRIMARY KEY, used INTEGER NOT NULL, expires BIGINT NOT NULL)',
                'CREATE TABLE IF NOT EXISTS import_owners (job TEXT NOT NULL, user_id TEXT NOT NULL REFERENCES users(id), PRIMARY KEY(job,user_id))',
                'CREATE INDEX IF NOT EXISTS session_expiry ON sessions(expires)',
                'CREATE TABLE IF NOT EXISTS preferences (user_id TEXT PRIMARY KEY REFERENCES users(id), filing_alerts INTEGER NOT NULL)',
                'CREATE TABLE IF NOT EXISTS notifications (event_id TEXT NOT NULL, user_id TEXT NOT NULL REFERENCES users(id), payload TEXT NOT NULL, created BIGINT NOT NULL, is_read INTEGER NOT NULL, PRIMARY KEY(event_id,user_id))',
                'CREATE TABLE IF NOT EXISTS deleted_accounts (digest TEXT PRIMARY KEY, created BIGINT NOT NULL)',

            ]: self.execute(conn, sql)

    @staticmethod
    def digest(value): return hashlib.sha256(value.encode()).hexdigest()

    def limit(self, key, count, seconds):
        now = int(time.time()); bucket = key + ':' + str(now // seconds)
        with self.db() as conn:
            self.execute(conn, 'DELETE FROM quotas WHERE expires < ?', (now,))
            row = self.execute(conn, 'INSERT INTO quotas(bucket,used,expires) VALUES(?,1,?) ON CONFLICT(bucket) DO UPDATE SET used=quotas.used+1 RETURNING used', (bucket, now + seconds * 2)).fetchone()
            if row[0] > count: raise Denied(429, 'Usage limit reached. Please try again later.')

    def login(self):
        import base64
        state = secrets.token_urlsafe(32); verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
        with self.db() as conn:
            self.execute(conn, 'DELETE FROM oauth_states WHERE expires < ?', (int(time.time()),))
            self.execute(conn, 'INSERT INTO oauth_states VALUES(?,?,?)', (self.digest(state), verifier, int(time.time()) + 600))
        return state, self.domain + '/oauth2/authorize?' + urlencode(dict(client_id=self.client, response_type='code', scope='openid email', redirect_uri=self.origin+'/auth/callback', state=state, code_challenge=challenge, code_challenge_method='S256'))

    def callback(self, code, state, cookie_state):
        if not state or not cookie_state or not secrets.compare_digest(state, cookie_state): raise Denied(400, 'Sign-in expired. Please try again.')
        with self.db() as conn:
            row = self.execute(conn, 'DELETE FROM oauth_states WHERE digest=? AND expires>? RETURNING verifier', (self.digest(state), int(time.time()))).fetchone()
        if not row: raise Denied(400, 'Sign-in expired. Please try again.')
        import requests
        import jwt
        response = requests.post(self.domain+'/oauth2/token', data=dict(grant_type='authorization_code', client_id=self.client, code=code, redirect_uri=self.origin+'/auth/callback', code_verifier=row[0]), timeout=10)
        response.raise_for_status()
        token = response.json()['id_token']
        claims = jwt.decode(token, self.keys.get_signing_key_from_jwt(token).key, algorithms=['RS256'], audience=self.client, issuer=self.issuer, options={'require':['exp','iat','sub']})
        if claims.get('token_use') != 'id' or claims.get('email_verified') is not True: raise Denied(403, 'Verify your email before signing in.')
        with self.db() as conn:
            deleted=self.execute(conn,'SELECT 1 FROM deleted_accounts WHERE digest=?',(self.digest(claims['sub']),)).fetchone()
        if deleted: raise Denied(403,'This application account was deleted. Contact the site owner to remove its login identity.')
        session = secrets.token_urlsafe(32)
        with self.db() as conn:
            self.execute(conn, 'INSERT INTO users VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET email=excluded.email', (claims['sub'], claims['email'], int(time.time())))
            self.execute(conn, 'DELETE FROM sessions WHERE expires < ?', (int(time.time()),))
            self.execute(conn, 'INSERT INTO sessions VALUES(?,?,?)', (self.digest(session), claims['sub'], int(time.time())+28800))
        return session

    def user(self, session):
        with self.db() as conn:
            row = self.execute(conn, 'SELECT users.id,users.email FROM sessions JOIN users ON users.id=sessions.user_id WHERE digest=? AND expires>?', (self.digest(session), int(time.time()))).fetchone()
        if not row: raise Denied(401, 'Sign in to continue.')
        return {'id':row[0], 'email':row[1]}

    def logout(self, session):
        with self.db() as conn: self.execute(conn, 'DELETE FROM sessions WHERE digest=?', (self.digest(session),))

    def data(self, user, payload=None):
        with self.db() as conn:
            if payload is not None:
                if not isinstance(payload,dict) or set(payload)-{'watchlist','notes'}: raise Denied(400,'Invalid saved data')
                stars, notes = payload.get('watchlist', []), payload.get('notes', {})
                if not isinstance(stars,list) or len(stars)>100 or any(not isinstance(x,str) or len(x)>20 for x in stars) or not isinstance(notes,dict) or len(notes)>100 or any(not isinstance(k,str) or len(k)>20 or not isinstance(v,dict) or set(v)-{'text','date'} or not isinstance(v.get('text'),str) or len(v['text'])>4000 or not isinstance(v.get('date'),str) or len(v['date'])>10 for k,v in notes.items()): raise Denied(400,'Saved data is too large')
                encoded = json.dumps(payload)
                self.execute(conn,'INSERT INTO user_data VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload',(user,encoded))
            row=self.execute(conn,'SELECT payload FROM user_data WHERE user_id=?',(user,)).fetchone()
            return json.loads(row[0]) if row else {'watchlist':[], 'notes':{}}

    def own(self, user, job, create=False):
        with self.db() as conn:
            if create: self.execute(conn,'INSERT INTO import_owners VALUES(?,?) ON CONFLICT DO NOTHING',(job,user))
            return bool(self.execute(conn,'SELECT 1 FROM import_owners WHERE job=? AND user_id=?',(job,user)).fetchone())

    def usage(self, user):
        now=int(time.time())
        limits=[('requests',20,60),('daily',200,86400),('imports',5,86400)]
        with self.db() as conn:
            result=[]
            for name,limit,seconds in limits:
                row=self.execute(conn,'SELECT used FROM quotas WHERE bucket=?',(f'{name}:{user}:{now//seconds}',)).fetchone()
                result.append(dict(name=name,used=row[0] if row else 0,limit=limit,resets_at=(now//seconds+1)*seconds))
            return result

    def preferences(self, user, payload=None):
        with self.db() as conn:
            if payload is not None:
                if not isinstance(payload,dict) or set(payload)!={'filing_alerts'} or not isinstance(payload['filing_alerts'],bool): raise Denied(400,'Invalid preferences')
                self.execute(conn,'INSERT INTO preferences VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET filing_alerts=excluded.filing_alerts',(user,int(payload['filing_alerts'])))
            row=self.execute(conn,'SELECT filing_alerts FROM preferences WHERE user_id=?',(user,)).fetchone()
            return {'filing_alerts':bool(row[0]) if row else False}

    def alerts(self, user, read=False):
        with self.db() as conn:
            if read:self.execute(conn,'UPDATE notifications SET is_read=1 WHERE user_id=?',(user,))
            rows=self.execute(conn,'SELECT event_id,payload,is_read FROM notifications WHERE user_id=? ORDER BY created DESC LIMIT 100',(user,)).fetchall()
            return [dict(json.loads(r[1]),id=r[0],read=bool(r[2])) for r in rows]

    def erase(self, user):
        """Erase application data, revoke all sessions and prevent Cognito re-entry.

        Cognito identity deletion is a separate privileged operation, reported in UI.
        """
        identity_deleted=False
        if os.getenv('LEDGER_COGNITO_DELETE_ENABLED','false').lower()=='true':
            import boto3
            pool=self.issuer.rsplit('/',1)[-1]
            region=pool.split('_',1)[0]
            boto3.client('cognito-idp',region_name=region).admin_delete_user(UserPoolId=pool,Username=user)
            identity_deleted=True
        with self.db() as conn:
            self.execute(conn,'INSERT INTO deleted_accounts VALUES(?,?) ON CONFLICT DO NOTHING',(self.digest(user),int(time.time())))
            for table in ['notifications','preferences','import_owners','user_data','sessions']:
                self.execute(conn,f'DELETE FROM {table} WHERE user_id=?',(user,))
            self.execute(conn,'DELETE FROM quotas WHERE bucket LIKE ?',('%:'+user+':%',))
            self.execute(conn,'DELETE FROM users WHERE id=?',(user,))
        return identity_deleted
