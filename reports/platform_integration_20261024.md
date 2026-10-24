# Ledger integration deployment — October 24, 2026

Live site: https://d3lj7roa7yk0i3.cloudfront.net

## Deployed and verified

- Account settings, per-user usage, research export, alert preferences and private
  filing inbox. Application-account deletion erases personal app records, revokes
  sessions and blocks the deleted Cognito subject from re-entry. Full Cognito
  identity deletion is implemented behind an explicit instance-permission flag,
  not enabled on the current server. The UI discloses this distinction.
- Two temporary synthetic users exercised the actual deployed HTTP authorization
  layer and PostgreSQL database. Verified private note isolation, export, foreign
  user-ID rejection, import ownership and deletion affecting only its own sessions.
  Temporary users were removed. This verifies backend isolation; it is not a claim
  that two real Cognito signup flows were run unattended.
- Hybrid retrieval combines FTS5 and per-company, date-eligible TF-IDF/SVD latent
  vectors. No external embedding service, model download or new Chroma index.
  Live Apple broad-risk query returns three risk-factor citations.
- Extra history import: up to 60 supported report bodies from five years,
  subject to archive-page limits. New low-disk guard. This is not exhaustive
  filing or bankruptcy-event coverage.
- One server-owned worker checks alert-enabled watchlists hourly. First check is
  a baseline; subsequent new supported report accessions generate deduplicated
  in-app alerts. Email and automatic full report re-import are not enabled.
- Frozen research artifact deployed with checksum verification. Model is verified
  but unapproved; public scores remain gated. No final evaluation or retraining.
- Encrypted daily account backups retain seven copies; a PostgreSQL restore drill
  populated a separate temporary schema and compared all five tables/rows, then
  dropped the schema. Session and OAuth tokens are not backed up. First backup and
  separate private recovery key copied off EC2 into git-ignored local operations
  storage. No secret values are included in this report or tool output.
- Minute systemd health checks include database readiness and tolerate startup.
  Health and backup timers are active. Failures are recorded locally.
- Versioned assets/revalidation headers fixed browser-stale UI. Public browser
  verified new navigation, current evidence cutoff, account sign-in gate, desktop
  and mobile layout without overflow, and no runtime exceptions.

## Integrated but unavailable until configured

- AI research summaries: ANTHROPIC_API_KEY plus ANTHROPIC_MODEL. Output cites
  supplied sources and exact quotes; invalid output falls back to excerpts.
  Source checks cannot prove the generated interpretation is correct.
- Quote, price history, provider valuation ratios and news: FINNHUB_API_KEY with
  appropriate plan/redistribution rights. Current provider data is separate from
  dated SEC evidence and frozen ML inputs. Unsupported endpoints degrade
  independently, without inventing missing prices/history.
- Full identity deletion: limited instance role for AdminDeleteUser on the exact
  Cognito pool plus LEDGER_COGNITO_DELETE_ENABLED=true. Not enabled without role.
- External failure notification and automatic off-server backup replication need
  an authorized destination/instance role. No email or SNS delivery is claimed.

## ML release remains blocked

The saved evaluation gate reports 467 exhibits / 2,434 excerpts awaiting human
interpretation, 19 positives lacking second-review records, 255 negatives outside
reviewed sample, and no released operating threshold. The final test remains
unconsumed. Enabling a score now would misrepresent validation.

## Validation and rollback

344 automated tests passed in the final regression run, with one existing toy
neural-network convergence warning. Local UI tests exercised settings, usage,
preferences, alerts, market-unavailable state and mobile layout. Live service health
and provider-state responses are verified separately from mocked-provider tests.

Code rollback archive on EC2:
/home/ubuntu/ledger-code-backups/pre-completion-20261024.tar.gz

Encrypted backup copies: data/processed/operations/20261024/backups/
Separate private recovery key: data/processed/operations/20261024/backup-recovery.key
Keep the recovery key private and copy it to a password manager. Backups expire by
retention; account deletion does not rewrite prior encrypted backup files.

Setup and provider activation details: infra/web/README.md
