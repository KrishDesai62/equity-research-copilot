# Additional readiness work — October 24, 2026

## Completed and deployed

A reproducible retrieval development fixture/evaluator found a long-passage bug:
the relevance filter was looking only at the first 64 unique passage words. Query
length remains bounded, but passage relevance now examines the full passage. A
regression test covers relevant content after a long preamble. The deployed
Microsoft cybersecurity query now finds the annotated risk passage.

The fixture has seven positive source examples and three abstention cases across
three issuers. Labels are assistant-authored and source-checked, not independently
human reviewed or held out. They are incomplete examples of relevance. Do not
present these development results as a validated human-labeled retrieval score.

Before the bug fix: hybrid Hit@5 4/7; lexical baseline 7/7.
After: hybrid Hit@5 7/7, MRR@5 1.000; lexical MRR@5 0.905.
All three abstention checks and all ten citation/date checks passed.
These same cases helped identify the defect, so post-fix results are regressions,
not evidence of generalization. A source review packet is generated for later
human adjudication; its visible rankings make it unsuitable for blind labeling.

A bounded loopback check on the live EC2 host used temporary synthetic users:
- 24 measured requests, two workers, three separate warmup requests.
- 24 HTTP 200 responses with source/date schema checks passing.
- Median 0.2936 s; p95 0.3117 s; 3.5657 s measured window.
- Warm-cache, single-server retrieval only. No SEC downloads, AI/market calls,
  bankruptcy-model inference or final evaluation. This is not an SLA or public
  internet capacity test. Temporary users, sessions and quotas were cleaned up.

Encrypted account backup/recovery checks reran successfully. Fresh encrypted copies
were copied off-server to ignored local operations storage. Scheduled automatic
S3 replication is now implemented, including read-back byte verification, but
remains inactive without a bucket/role. Local backup recovery key remains private.

## Prepared for user activation

infra/web/operations.yaml passed cfn-lint. It provisions a private encrypted S3
bucket, a narrowly scoped EC2 role (exact Cognito pool, bucket prefix and SNS topic),
EC2/RDS alarms, and optional confirmed email subscription. It does not attach the
role to an unmanaged existing instance. It has NOT been provisioned; resources
can incur charges. Account permissions and a confirmed recipient remain needed.

Optional OnFailure services and SNS notifications are deployed but inactive without
LEDGER_ALERT_TOPIC_ARN. Messages contain no user data and are limited to one per
operation per hour. No external alert delivery was claimed or exercised.

Full Cognito identity deletion already has a permission-gated implementation; the
account page now distinguishes enabled full deletion from application-only deletion.
It is still not enabled without the instance role.

## Validation and unchanged gates

351 tests passed, with one existing toy neural-network convergence warning.
AWS template validation passed. Public model scores remain disabled. Frozen model,
final test, review approvals and thresholds were not changed. Provider keys are
still missing; live AI generation and market-data validation cannot be completed.

Reproduce:
- python -m scripts.evaluate_retrieval
- python -m scripts.load_check --requests 24 --concurrency 2 (EC2 only)
- cfn-lint infra/web/operations.yaml

Local results/review packet: data/processed/evaluation/
Operational setup: infra/web/README.md
Rollback archive: /home/ubuntu/ledger-code-backups/pre-readiness-fixes-20261024.tar.gz
