# Implementation status for the six production requirements

**Superseding October 18 expansion:** the [new study](sec_disclosure_expansion_20261018.md)
now has 20 final positives and 315 disclosure negatives, 1,282 usable historical
training snapshots, seven added filing features and a completed nine-model
comparison. Precision/recall and independent review remain incomplete; the final
is unscored. Earlier counts below describe the archived stage.


**October 18 update:** [the separate SEC-disclosure cohort](sec_disclosure_v1.md)
now has 29 positive and 230 completed-follow-up disclosure-negative snapshots.
Five models were fitted, but no development operating threshold met both floors;
the reserved final has nine positives and was not scored. This supersedes older
"zero recent negatives" statements only for the new disclosure target, not
exhaustive court-bankruptcy coverage. Statistical approval, independent label
review, trading validation and live AWS tests remain incomplete.

The supporting code below is implemented and tested locally. The statistical
requirements are **not all complete**, and the model is not approved for production.
No new recent negative labels, new blind test results, agent profits or live AWS
success are claimed. Existing V3 data, candidate and final results are preserved.
Local verification: **189 tests passed**, compilation and all three CloudFormation
templates passed lint. The existing small neural-network test fixture emits a
convergence warning. The updated Lambda package was built and its packaged source
bytes checked against the workspace. These checks do not establish model accuracy
or live AWS correctness.

| Requirement | Implemented now / already present | Evidence still required |
|---|---|---|
| 1. Expand reviewed outcomes | Primary SEC evidence acquired for three additional 2024–2025 Chapter 11 events. Added outcome-review admission with exact CIK, two named reviewers, hashed source evidence, explicit prior-case checks and full-year negative coverage. | First-case/group review and prediction-unit pairing for the new events; complete recent negative coverage. Zero new negative labels were manufactured. |
| 2. Improve development detection | Seven additional features extracted into a separate 2,439-row development table; disclosure inputs available on 137 rows. Added fixed five-model development tuning, company-fold calibration, threshold selection and candidate freezing. Non-converged candidates are rejected. | Reviewed labels and original filing pairs for the new reserved development population. No new candidate has been fitted on real reviewed recent data; improved accuracy is unproven. |
| 3. Recent probability validation | Added frozen-artifact evaluation with source hashes, completed horizons, feature freshness, missingness checks, weighted losses and calibration bins. | A complete reviewed recent cohort, a documented probability sampling design and sufficient events. Weighted/descriptive metrics do not auto-certify calibration. |
| 4. New untouched final test | Added company reservation excluding prior experiments and newly known-positive evidence; later final anchors, immutable plans, disjoint roles, frozen model hashes and reservation-wide one-use scoring. A different output directory cannot reset the test. | An actual new outcome-blind sampled company list and qualified reviewed outcomes. No new real final cohort was reserved or scored in this update. |
| 5. Buy/sell evaluation | Replaced the backtest stubs with a dated decision-panel evaluator, fixed risk/news comparator, buy-and-hold, turnover costs, drift-aware holdings and loss/delisting support. Rejects future inputs, overlapping periods and missing firms. | Actual recorded agent decisions and complete point-in-time total returns. Bull/bear excerpts are not validated directional decisions; synthesis/portfolio agents remain unfinished. No actual profitability backtest has been run. |
| 6. Operations | Added read-only IAM status endpoint, durable monitor success/failure health, SNS alarm routing, missing-run/queue alarms, approved-model version registration and atomic activation/rollback. Local data snapshot versioning/rollback already existed. Updated CI and built the Lambda package. | AWS account/profile, buckets, an operator role and an alert topic, followed by live schedule/auth/alert/recovery tests. AWS remains undeployed as previously requested. This status endpoint is operator access, not a complete authenticated trading application. |

## Newly acquired primary event evidence

- **WW International, 2025-05-06:** [SEC 8-K](https://www.sec.gov/Archives/edgar/data/105319/000119312525114307/d934787d8k.htm).
- **23andMe, 2025-03-23:** [SEC 8-K](https://www.sec.gov/Archives/edgar/data/1804591/000119312525067076/d928462d8k.htm).
- **Tupperware, 2024-09-17:** [SEC filing exhibit](https://www.sec.gov/Archives/edgar/data/1008654/000100865424000068/exhibit991tupperwarepres.htm).

Downloaded source URLs, hashes and event-content checks are recorded at
`data/processed/validation/recent_event_evidence.json`; raw documents are under
`data/raw/recent_event_evidence/`. These are known-positive development evidence,
not a random population sample, first-recorded-case labels or eligible final-test
companies. None was previously assigned in the saved experiments when checked.
Their addition does not change the 46-company research snapshot or the sealed V3
model/dataset. Evidence configuration is `config/evidence/recent_events.json`.
Once their first-case and prediction-unit reviews are complete, known-case IDs
can be added with reservation option `--training-cases PATH_TO_KNOWN_CASE_CIKS.json`.
These are forced to training and cannot enter calibration or final evaluation.
They do not count as outcome-blind sampled companies.

## Development features

`src/ml/disclosure_features.py` adds:

- Accruals / assets: trailing-year net income minus operating cash flow, divided by assets.
- Net-debt **proxy** / assets: total liabilities minus cash, divided by assets. This is not reconstructed interest-bearing debt.
- Operating cash burn / cash, with missing/nonpositive denominators left missing.
- Risk-factor and MD&A textual change ratios and added distress-term counts.

Change reports must match issuer, original form and filing order. Only reports
strictly before the prediction date, at most 550 days old, may join a row. Missing
sections remain missing, not zero. A textual change or phrase count is a feature,
not a reviewed materiality or bankruptcy label.

```bash
.venv/bin/python -m src.ml.disclosure_features
.venv/bin/python -m src.validation.recent_event_evidence
```

The active derived table is
`data/processed/validation/disclosure_development_v2/features.csv`, with source
hashes in its manifest. An earlier extraction directory is an intermediate risk-only
artifact and is not used for model fitting. The current command verifies and
reuses its existing output; changed source files require a new `--output` directory.

## Reviewed outcomes and new experiment

`config/templates/outcome_review.example.json` is intentionally **not a valid
review**. Fill records only from real evidence and actual reviews; never invent
reviewer identities or mark incomplete searches as complete. For positives use
`event_date`, `chapter` 7/11 and evidence kind `bankruptcy_event`. For negatives
supply `coverage_through`, `complete_event_coverage`, a coverage method and evidence
kind `complete_outcome_coverage`. Dates must be completed by the assessment/review
cutoff and file hashes must match. The validator verifies the review contract and
source integrity; it cannot prove a reviewer's legal conclusions or independence.

Save genuine review records as JSONL. Select the new sampling frame without
consulting outcomes; reserve all its company IDs before feature/label acquisition.
Do not seed a final cohort with the three known-positive evidence companies above.
Document inclusion probabilities and nonresponse; a hash partition alone is not
proof of population representativeness. Keep reservations in `config/reservations/`.

```bash
.venv/bin/python -m src.validation.future_cohort reserve \
  --companies PATH_TO_OUTCOME_BLIND_CIK_LIST.json \
  --output config/reservations/next.json \
  --seed PRECOMMITTED_SEED --final-start 2024-12-31

.venv/bin/python -m src.validation.tune_reviewed \
  --reservation config/reservations/next.json \
  --features PATH_TO_DEVELOPMENT_FEATURES.csv \
  --reviews PATH_TO_DEVELOPMENT_REVIEWS.jsonl \
  --output data/processed/validation/next_candidate --as-of YYYY-MM-DD

.venv/bin/python -m src.validation.prospective_evaluation evaluate-once \
  --frozen data/processed/validation/next_candidate/frozen_candidate.json \
  --features PATH_TO_FINAL_FEATURES.csv \
  --reviews PATH_TO_FINAL_REVIEWS.jsonl \
  --output data/processed/validation/next_final --as-of YYYY-MM-DD
```

These commands require real input files; placeholders are not supplied data.
Development CSVs must contain only reserved training/calibration firms and no target
column; targets come from separately verified reviews. Feature rows need CIK,
prediction date, feature filing date, latest input date and financial values.
Optional design weights must reflect a reviewed sampling design. Training outcomes
must finish before calibration predictions, and calibration outcomes must finish
before the reserved final period. New features absent throughout training are
explicitly omitted, not silently imputed into a fictitious feature.

Final inputs must include every reserved final company at the reserved anchor,
without replacements or silent coverage-based dropping. Missing coverage blocks
scoring. The reservation-wide marker is consumed before prediction, including on
subsequent failures; changing output folders or freezing another candidate cannot
reuse it. Do not delete markers to create apparent independent evidence.

## Agent backtesting

```bash
.venv/bin/python -m src.backtest.backtest_engine \
  --panel PATH_TO_RECORDED_PANEL.csv --cost-bps 10 \
  --output data/processed/validation/strategy_run.json
```

Required panel columns: `ticker`, `decision_at`, `available_at`,
`model_trained_through`, `entry_at`, `exit_at`, `agent_action` (buy/hold/sell),
`bankruptcy_probability`, `news_sentiment` (-1 to 1), `evidence_url` and
`total_return` (decimal). Use UTC timestamps, actual model cutoffs and the latest
availability of all evidence used. Returns must include dividends and delisting
losses. Do not use a present-day survivor list as historical ground truth.

The current engine accepts a complete fixed-universe panel with non-overlapping
execution intervals. Buy/sell targets invest 1/N per selected stock, hold preserves
the stock's invested/cash decision, and uninvested capital earns zero. The mechanical
comparator buys when risk <10% and news sentiment >0. Buy-and-hold buys equal
initial weights and lets them drift. Costs apply per traded dollar, including
initial entries; final liquidation, tax, market impact and borrowing are excluded.
No profits or statistical significance are inferred from synthetic unit tests.

## AWS and rollback

- `infra/aws/monitor.yaml`: schedule/retries/DLQ plus alarms and optional existing
  `AlarmTopicArn`. Blank means notifications are not delivered; confirmed SNS
  subscriptions/topic permissions must be tested live.
- `infra/aws/status.yaml`: read-only `GET /status` with `AWS_IAM` authorization.
  It attaches URL and URL-only invocation permissions to an existing operator role.
  Authentication follows [AWS's Function URL access-control documentation](https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html).
- `src/operations/status.py`: returns monitor health/freshness, never a model score
  or approval. Missing/stale/future/failure checkpoints are unhealthy. Run the
  monitor successfully before testing the S3-backed endpoint.
- `src/operations/model_registry.py`: registers only policy-approved binary
  bankruptcy artifacts, preserves immutable versions and atomically changes the
  `current` symlink. Rollback rechecks the current release policy. Configure
  `RiskAgentTool(model_registry_path=REGISTRY_ROOT/current)` to consume it. No
  current research candidate is eligible for registration.

```bash
.venv/bin/python -m src.operations.model_registry --root PATH_TO_REGISTRY register \
  --artifact PATH_TO_APPROVED_MODEL --release PATH_TO_APPROVED_RELEASE.json
.venv/bin/python -m src.operations.model_registry --root PATH_TO_REGISTRY activate --version REGISTERED_ID
.venv/bin/python -m src.operations.model_registry --root PATH_TO_REGISTRY rollback --version PREVIOUS_APPROVED_ID
```

Local package: `build/filing-monitor-ops-20261017-v2.zip`, built for Linux Python
3.12 with the monitor and status modules. It contains no `.env`. Use an immutable,
versioned S3 object key when eventually deploying, preserve its SHA-256 and retain
an earlier package for code rollback. Do not overwrite an existing deployed ZIP
key. The schedule remains disabled until a successful manual run, denied/allowed
IAM checks, alarm delivery and injected-failure/recovery tests have passed.

No deployment or message delivery was performed. Live AWS validation requires
actual account configuration and remains deferred until model validation, following
the user's earlier instruction.
