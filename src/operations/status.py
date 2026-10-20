"""Read-only operator endpoint; invoke through an AWS_IAM Lambda Function URL."""
from datetime import datetime, timezone
import json

from src.aws.filing_monitor import MonitorStore
from src.aws.s3_cache import S3Cache


def assess_health(status, *, now=None, max_age_hours=30):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Health assessment needs an aware UTC clock')
    if not status:
        return {'healthy': False, 'reason': 'No successful monitor checkpoint', 'production_model_approved': False}
    try:
        timestamp = datetime.fromisoformat(status['completed_at'])
        if timestamp.tzinfo is None:
            raise ValueError('Missing timezone')
        age = (now - timestamp).total_seconds() / 3600
        healthy = status.get('state') == 'success' and 0 <= age <= max_age_hours
    except (KeyError, TypeError, ValueError):
        healthy, age = False, None
    return {'healthy': healthy, 'checkpoint_age_hours': age, 'monitor_state': status.get('state'),
            'production_model_approved': False,
            'note': 'Monitor health is separate from model approval; no predictions are served by this endpoint.'}


def handler(event, context):
    principal = event.get('requestContext', {}).get('authorizer', {}).get('iam', {}).get('userArn')
    if not principal:
        return {'statusCode': 401, 'body': json.dumps({'error': 'IAM authentication required'})}
    if event.get('rawPath') != '/status' or event.get('requestContext', {}).get('http', {}).get('method') != 'GET':
        return {'statusCode': 404, 'body': json.dumps({'error': 'Not found'})}
    cache = S3Cache.from_env()
    if cache is None:
        raise ValueError('Configure the existing cache bucket before serving status')
    result = assess_health(MonitorStore(cache).read_json('monitor/health.json'))
    return {'statusCode': 200 if result['healthy'] else 503,
            'headers': {'Content-Type': 'application/json', 'Cache-Control': 'no-store'}, 'body': json.dumps(result)}
