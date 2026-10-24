"""Encrypted application database backup and isolated PostgreSQL restore drill.

Sessions and transient OAuth state are excluded so restoration requires login.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import secrets
from cryptography.fernet import Fernet
from src.web.accounts import Accounts

TABLES=['users','user_data','preferences','notifications','deleted_accounts']


def backup(accounts, folder):
    key=os.environ.get('LEDGER_BACKUP_KEY')
    if not key:raise RuntimeError('LEDGER_BACKUP_KEY is required')
    tables={}
    with accounts.db() as conn:
        # PostgreSQL repeatable read provides a single snapshot of every table.
        import sqlite3
        if not isinstance(conn,sqlite3.Connection):conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for table in TABLES:
            cursor=accounts.execute(conn,f'SELECT * FROM {table}')
            tables[table]={'columns':[d[0] for d in cursor.description],'rows':cursor.fetchall()}
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);folder.chmod(0o700)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target=folder/f'accounts-{stamp}.fernet'
    target.write_bytes(Fernet(key.encode()).encrypt(json.dumps({'version':1,'tables':tables}).encode()));target.chmod(0o600)
    for old in sorted(folder.glob('accounts-*.fernet'))[:-7]:old.unlink()
    return target


def restore_drill(accounts, path):
    """Restore into a fresh isolated schema, compare every row, then drop it."""
    import psycopg
    from psycopg import sql
    payload=json.loads(Fernet(os.environ['LEDGER_BACKUP_KEY'].encode()).decrypt(Path(path).read_bytes()))
    if payload.get('version')!=1 or set(payload['tables'])!=set(TABLES):raise ValueError('Unsupported backup')
    schema='ledger_restore_'+secrets.token_hex(8)
    count=0
    with psycopg.connect(accounts.dsn) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        for table,record in payload['tables'].items():
            target=sql.Identifier(schema,table)
            conn.execute(sql.SQL('CREATE TABLE {} (LIKE public.{} INCLUDING ALL)').format(target,sql.Identifier(table)))
            cols=sql.SQL(',').join(map(sql.Identifier,record['columns']))
            for row in record['rows']:
                conn.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(target,cols,sql.SQL(',').join(sql.Placeholder() for _ in row)),row)
            restored=conn.execute(sql.SQL('SELECT {} FROM {}').format(cols,target)).fetchall()
            canonical=lambda rows:sorted(json.dumps(list(row),sort_keys=True) for row in rows)
            if canonical(restored)!=canonical(record['rows']):raise ValueError('Restore comparison failed')
            count+=len(restored)
        conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
    return {'status':'verified','tables':len(TABLES),'rows_compared':count,'sessions_restored':False}


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default='/home/ubuntu/ledger-backups');parser.add_argument('--drill',action='store_true')
    args=parser.parse_args();accounts=Accounts();path=backup(accounts,args.folder)
    status={'backup_created':True,'completed_at':datetime.now(timezone.utc).isoformat()}
    if args.drill:status['restore']=restore_drill(accounts,path)
    (Path(args.folder)/'status.json').write_text(json.dumps(status));print(json.dumps(status))

if __name__=='__main__':main()
