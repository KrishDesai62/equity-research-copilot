# Deploy the scheduled filing monitor

This deploys a daily EventBridge Scheduler invocation of a small Python Lambda.
The Lambda checks a CIK watchlist, saves new original 10-K/10-Q filings through
the existing S3 cache, writes filing-change reports, then checkpoints successes.
Training, embeddings, and the other research agents are not part of this function.

## Prepare the bucket and watchlist

Use AWS CLI v2 with an authenticated profile or role. Choose one AWS region for
the cache bucket, code bucket, and Lambda. Deploy `cache.yaml` if you do not
already have the cache bucket:

```bash
aws cloudformation deploy --template-file infra/aws/cache.yaml --stack-name equity-research-cache
aws cloudformation describe-stacks --stack-name equity-research-cache --query 'Stacks[0].Outputs'
```

Use the returned bucket name in place of `YOUR_BUCKET` below. You may use that
same bucket for the Lambda artifact. Copy `config/monitor_watchlist.example.json`
to a local file and replace the companies with your own watchlist (1-30 companies).
The example's Apple/Microsoft entries are a monitoring demo, not a historical
training universe. Each company must include a CIK; ticker is a display label.
CIKs stay stable across ticker changes and avoid ambiguous ticker lookups.

```bash
aws s3 cp config/monitor_watchlist.example.json s3://YOUR_BUCKET/equity-research/monitor/watchlist.json
```

Edit the file before uploading to monitor your intended companies. Changing this
S3 object changes the next run's watchlist without redeploying. Removing a company
stops monitoring it; its existing reports and checkpoints remain intact.

## Build and deploy

From the repository root:

```bash
.venv/bin/python scripts/build_monitor.py
aws s3 cp build/filing-monitor.zip s3://YOUR_BUCKET/deploy/filing-monitor-v1.zip
aws cloudformation deploy \
  --template-file infra/aws/monitor.yaml \
  --stack-name equity-research-monitor \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    CacheBucketName=YOUR_BUCKET \
    CodeBucketName=YOUR_BUCKET \
    CodeObjectKey=deploy/filing-monitor-v1.zip \
    EdgarUserAgent='YourApp your-real-email@your-domain.com' \
    ScheduleState=DISABLED
aws cloudformation describe-stacks --stack-name equity-research-monitor --query 'Stacks[0].Outputs'
```

The build downloads only Lambda runtime dependencies, targeting Linux x86_64
Python 3.12. It excludes ML libraries, local `.env`, credentials, and raw data.
It does not deploy anything. Every code update should use a **new code object
key** (for example `filing-monitor-v2.zip`) and pass that key on redeployment,
so CloudFormation detects the code change.

For a first run, use the `FunctionName` output in this command:

```bash
aws lambda invoke --function-name YOUR_FUNCTION_NAME \
  --invocation-type Event --payload '{}' build/monitor-invocation.json
aws logs tail /aws/lambda/YOUR_FUNCTION_NAME --since 30m
aws s3 ls s3://YOUR_BUCKET/equity-research/monitor/reports/ --recursive
```

An asynchronous invocation acknowledges acceptance, not completion: inspect logs
and S3 for results. First observation processes only the latest original filing
of **each** supported form and records older listed filings as a baseline. If no
comparable prior filing is listed, the raw document is still stored and a status
report explains the missing comparison. Later runs process every unseen accession
in the recent submissions list, not merely the newest one. Amendments and future
filing dates are excluded.

After checking the results, rerun the **same deployment command** with
`ScheduleState=ENABLED`. The default schedule runs daily at **6 PM America/Chicago**,
including weekends, and follows daylight saving time. Customize with
`ScheduleExpression='cron(0 18 * * ? *)'` and `ScheduleTimezone=America/Chicago`.
To pause, redeploy with `ScheduleState=DISABLED`.

## Where results go

All paths below are inside your S3 bucket under `equity-research/` (or the
configured `CachePrefix`):

| Path | Contents |
| --- | --- |
| `filings/<CIK>/<accession>/<document>` | Cached raw filings |
| `monitor/watchlist.json` | Your company configuration |
| `monitor/state/<CIK>.json` | Known accession numbers and last successful checkpoint |
| `monitor/reports/<CIK>/<accession>.json` | Structured change report with citations |
| `monitor/reports/<CIK>/<accession>.md` | Readable report |

Report uploads precede checkpoint writes. A failed upload leaves the filing
pending for retry. Output keys are deterministic, so retrying replaces the same
logical report instead of creating another report with a timestamped name. S3
versioning can retain multiple physical versions after a retried write. This is
at-least-once processing, not an exactly-once transaction.

## Failures and practical limits

- The function has one reserved concurrent execution. Do not deploy another
  writer or run `run_monitor` separately against the same state prefix. This also
  keeps the monitor's SEC requests sequential and rate limited.
- Each company's successful filings are checkpointed separately. One company's
  failure does not prevent checking the remaining watchlist; the run then raises
  so Lambda can retry. Time-budget exhaustion also preserves earlier progress.
- Lambda execution failures get two asynchronous retries. Scheduler delivery
  failures have their own retries. Exhausted failures go to the stack's encrypted
  SQS failure queue, retained for 14 days. Inspect the queue and logs, fix the cause,
  and invoke the function again; successful filings are skipped.
- CloudWatch logs are retained for 14 days. The stack creates an error alarm in
  CloudWatch; it does **not** configure email/SMS notifications.
- The monitor uses SEC **recent submissions**, not the archived submissions files.
  It is intended for daily monitoring, not historical backfills. A very long outage
  can miss filings that roll out of that list. Removed companies are not polled.
- The initial pass or a large backlog can exceed Lambda's 15-minute timeout. Use
  small watchlists first. If a sustained workload cannot fit, move work to a queue
  rather than raising the watchlist limit. Reports with unavailable sections are
  saved as such; they do not imply that disclosures were unchanged.
- The role supports the AES256 bucket from `cache.yaml`. A bucket requiring a
  customer KMS key needs matching encryption settings and KMS permissions.
- S3, Lambda, Scheduler, logs, and SQS can incur AWS charges. No SageMaker, GPU,
  persistent server, or managed vector database is provisioned.

No live infrastructure is created by editing the repository or running tests.
The deployment commands above create resources in your configured account.

AWS references: [Scheduler Lambda targets](https://docs.aws.amazon.com/lambda/latest/dg/with-eventbridge-scheduler.html),
[asynchronous Lambda error handling](https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-error-handling.html).
