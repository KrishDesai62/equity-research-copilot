# Public accounts deployment

Implemented locally; these resources are not provisioned by this repository.
Keep the current server loopback-only until configuration is complete.

1. Create a Cognito user pool with email sign-in, automatic email verification,
   self-service sign-up and password recovery. Require verified email. Enable
   managed login and a public application client (no client secret), authorization
   code grant only, scopes openid and email. Set the callback to
   https://YOUR-HOST/auth/callback and logout to https://YOUR-HOST/.
   Use short token lifetimes; Ledger sessions last eight hours and logout deletes
   the local session. Cognito password reset/global logout does not revoke an
   already established Ledger session; account administrators can delete that
   user's database sessions when disabling access.
2. Provision PostgreSQL (RDS or equivalent). Do not expose port 5432 publicly.
   Allow the EC2 security group only. Use a dedicated database and non-superuser
   app login with table privileges. The app creates its tables on startup.
   Set DATABASE_URL using TLS certificate verification and the AWS RDS CA bundle.
   Configure managed encrypted daily backups and retention (e.g. seven days).
3. Install requirements-web.txt into the server virtual environment. Append the
   settings in accounts.env.example to the private .env (chmod 600). Configure a
   real HTTPS hostname and certificate. The public mode refuses missing Cognito,
   PostgreSQL or HTTPS settings. Existing private mode remains account-free.
4. Adapt nginx.conf for that hostname and certificate; nginx -t before reloading.
   Keep Python bound to 127.0.0.1 and ports 8000/5432 closed in EC2. Open only
   80/443 publicly and SSH to your IP. Restart equity-research.service.
5. Test two separate accounts: verify email, sign in, save distinct notes,
   reload, sign out, and check one user's import status is invisible to the other.
   API calls require login; profile writes also require the exact configured
   Origin. Test expired sessions and quota exhaustion. No tokens live in browser
   localStorage. Existing local notes are not automatically copied into accounts.

## Usage and operations

Per user: 20 work requests/minute, 200/day, five imports/day. All attempts consume
quota, including malformed or failed work. UTC epoch fixed windows; limits are
atomic in PostgreSQL and shared across processes. Two synchronous work slots per
process; the existing shared import queue has one worker and an eight-job bound.
Nginx independently limits by connection IP (no trust in client forwarded headers).
The application also caps proxy-peer traffic at 1200/minute. Do not run multiple
import-worker processes sharing the import directory. Public SEC imports remain
shared across accounts; only personal notes and job access are private.

Monitor /api/health using an external HTTPS uptime check, plus systemd failures,
EC2 CPU, disk, memory (CloudWatch agent), RDS storage/connections and backup
failures. Create CloudWatch alarms with a notification destination you control.
Request logs omit query strings/bodies/cookies. Retain logs with a bounded policy.
The health endpoint is a process liveness check, not a database readiness check.

backup.sh supplies optional encrypted S3 copies of a PostgreSQL custom dump and
import files. Requires pg_dump, aws CLI and a least-privilege instance role for a
private backup bucket. Enable bucket versioning and lifecycle retention. Schedule
when no imports are active for a consistent import-directory copy. Do not place
credentials in crontab; load the protected environment in a service. RDS automated
backups are the primary database recovery mechanism.

Restore drill: stop Ledger; restore a dump with pg_restore into a fresh private
PostgreSQL database; restore imports to a fresh directory; point a private test
instance at both and confirm two-account isolation and cited retrieval before
cutover. Delete restored sessions to require new logins. Backups contain private
notes and email addresses: restrict access and deletion permissions.

Provisioning Cognito, a database, certificates, alarms and backup schedules is
still required. No claim of completed live security audit or recovery test.

Local health timer: copy ledger-health.service and ledger-health.timer to
/etc/systemd/system/, run systemctl daemon-reload and enable --now
ledger-health.timer. A failed probe leaves a failed service; connect this to a
CloudWatch/log notification destination before relying on alerts. This does not
silently claim email alerts are configured.

Database quotas intentionally use an application-wide table; user data queries
always bind the authenticated session user and never accept a user ID from the
browser. Database credentials have access to all application rows: this is
application authorization, not a claim of PostgreSQL row-level security.

## Integrated features (October 24)

Account and Filing alerts navigation now provide saved-data export, usage counters,
alert preferences, a private filing inbox and application-account deletion.
Deletion erases app data and sessions and blocks re-entry by the Cognito subject.
By default the Cognito identity itself remains; the interface explicitly says so.
Full identity deletion can be enabled with LEDGER_COGNITO_DELETE_ENABLED=true only
once the instance role allows cognito-idp:AdminDeleteUser on this exact user pool.
Do not grant broad administrator permissions. Existing backups expire by retention
rather than being rewritten when a user deletes their data.

A single server-owned thread checks alert-enabled watchlists hourly using the
shared SEC request limiter. First observation establishes a baseline; later new
supported report accessions generate deduplicated in-app notices. It does not send
email and does not automatically consume user import quotas or re-import reports.
Monitor details live in company_imports/monitor/status.json. Followed companies
must be present in the catalog. Imports now support history=true (up to 60
supported reports from five years, subject to archive-page limits), in addition to
the default six-report selection. Neither mode is exhaustive court or SEC coverage.

Retrieval combines SQLite BM25 and bounded per-company TF-IDF/TruncatedSVD latent
semantic vectors with reciprocal-rank fusion. Embeddings are fit only to passages
eligible at the requested date and cached in memory (8 slices, max 2,000 passages
per slice). This is local statistical semantic retrieval, not a neural embedding
service or a new Chroma index. It needs no provider key or model download.

AI drafts: set ANTHROPIC_API_KEY and ANTHROPIC_MODEL (an available model ID from your
provider account). Statements must cite supplied source IDs and include a matching
exact source quote; invalid output falls back to excerpts. Citation checks do not
prove the generated interpretation is correct. Public SEC excerpts/questions go
to the configured provider; account notes and credentials are never sent.
Provider protocol: https://platform.claude.com/docs/en/api/overview

Market & news: set FINNHUB_API_KEY with suitable subscription and redistribution
rights. Server-side quote/news requests are cached five minutes. Quote timestamps
are shown; the app does not promise real-time exchange data. These are current
market data, independent of the historical SEC evidence cutoff and never inputs
to the frozen ML model. Provider: https://finnhub.io/docs/api

Backups: python -m src.web.backups --drill encrypts the database snapshot using
LEDGER_BACKUP_KEY and restores it into an isolated temporary PostgreSQL schema,
compares every row, then drops that schema. Sessions/OAuth state are excluded.
The daily ledger-backup.timer retains seven encrypted copies on EC2. An off-server
copy and a separate protected recovery key were saved locally during deployment;
automatic off-server replication still needs a configured destination/instance
role. Losing both the key and server prevents decryption. Keep the recovery key in
your password manager. The minute health check now also checks database liveness.
These systemd checks record failures locally; external email/SNS alarm delivery
still needs an authorized notification destination and AWS permissions.

## Optional operations stack (prepared, not provisioned)

`operations.yaml` creates a private encrypted/versioned backup bucket, an EC2
instance profile restricted to that bucket and the specified Cognito user pool,
and an SNS topic with EC2-status/RDS-storage alarms. Email subscription is optional
and requires recipient confirmation. CloudFormation needs your account permissions
and CAPABILITY_IAM acknowledgement; resources can incur S3/CloudWatch/SNS charges.
Validate with `cfn-lint infra/web/operations.yaml` before creation.

After you create the stack in us-east-2, attach its output InstanceProfile to the
existing EC2 server in Actions > Security > Modify IAM role. Do not replace a role
used by another application without merging its required permissions. Set these
private server environment values from stack outputs:

```
LEDGER_BACKUP_BUCKET=<BackupBucketName>
LEDGER_ALERT_TOPIC_ARN=<AlertTopicArn>
LEDGER_COGNITO_DELETE_ENABLED=true
```

Copy ledger-alert@.service and the updated health/backup services into
/etc/systemd/system, run systemctl daemon-reload, restart equity-research and run
ledger-backup.service. The backup command uploads only the encrypted dump and
verifies the downloaded bytes. Failure notifications contain the operation name
and an instruction to inspect logs; each operation has a one-hour cooldown.
Nothing is sent when the SNS destination is unconfigured. Confirm subscription
and exercise a controlled alert before claiming external alarm delivery works.

This template does not attach itself to an existing unmanaged EC2 instance or
create paid AI/market-provider subscriptions. It was statically validated only.
