"""Optional failure alerts to a configured SNS destination; no secrets in messages."""
import argparse
import json
import os
from pathlib import Path
import re
import time


def notify(unit, root='/home/ubuntu/ledger-operation-alerts'):
    if unit not in {'ledger-health','ledger-backup'}:raise ValueError('Unknown operation')
    topic=os.getenv('LEDGER_ALERT_TOPIC_ARN','')
    if not topic:return {'status':'not_configured'}
    if not re.fullmatch(r'arn:aws[a-z-]*:sns:[a-z0-9-]+:\d{12}:[A-Za-z0-9_-]+',topic):raise ValueError('Invalid SNS topic')
    folder=Path(root);folder.mkdir(parents=True,exist_ok=True);folder.chmod(0o700)
    marker=folder/(unit+'.json');now=int(time.time())
    if marker.exists() and now-json.loads(marker.read_text())['sent_at']<3600:return {'status':'cooldown'}
    import boto3
    region=topic.split(':')[3]
    boto3.client('sns',region_name=region).publish(TopicArn=topic,Subject='Ledger operation needs attention',Message=unit+' failed. Inspect the systemd journal on the Ledger server. This alert contains no user data or credentials.')
    marker.write_text(json.dumps({'sent_at':now}));marker.chmod(0o600)
    return {'status':'sent'}

if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env')
    p=argparse.ArgumentParser();p.add_argument('unit');args=p.parse_args();print(json.dumps(notify(args.unit)))
