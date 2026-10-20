# Production readiness — October 17, 2026

The local research pipeline has stronger operating controls. The product and
bankruptcy model are not approved for production. No AWS resources were deployed.

Latest V3 experiment: a fixed 915-company acquisition produced 542 eligible
snapshots. Five models were compared with design-weighted company-fold calibration.
The selected logistic model's one-use test covers 75 companies and five bankruptcies,
with weighted ROC-AUC 0.794. It has no qualifying precision/recall threshold and
insufficient final positive companies; probability calibration remains unvalidated.
The Brier improvement over the constant baseline has an interval spanning zero.
No model was promoted. See [the V3 results](cohort_v3_results.md) and
[the data card](cohort_v3_data_card.md). All 171 tests pass. The historical
experiments below remain preserved as earlier evidence.

Earlier development experiment: a 435-issuer panel and ten model/calibration
combinations were completed. The selected V2 research candidate reduced loss
relative to the previous candidate on 39 new negative observations, but did not
beat the constant baseline. Recall remains unmeasurable on that sample. No
candidate was promoted. See [the full improvement report](model_improvement_v2.md).

Latest calibration audit: 51 eligible, previously unexposed historical companies
had zero reviewed one-year events while mean scores were 26.2%. The sample has
substantial coverage/selection limits, but it raises a serious calibration concern.
See [population calibration](population_calibration.md). AWS is not needed for
this work. Local test count is now 143.

Update: the new frozen candidate has now passed through a separate independent
historical benchmark: precision 68.8%, recall 76.8%, accuracy 80.6% on 47 companies.
See [the independent evaluation](independent_candidate_evaluation.md).
Production remains blocked by population/calibration evidence and the 50-company
floor. The result below belongs to the **earlier candidate**, preserved for audit.
The new candidate's evidence is reported separately; it is not approved for serving.

## Confirmed product scope

The user confirmed the intended release: predict bankruptcy risk and have agents
produce buy/hold/sell recommendations using real news and bankruptcy evidence.
Recommendations are research outputs; order execution is outside this request.

The bankruptcy output should specify its prediction horizon (initially one year),
observation date, calibrated probability, model version and supporting filings.
Unavailable or unvalidated risk must be explicit, rather than treated as low risk.

News evidence needs source URLs, publication timestamps and retrieval timestamps.
An agent must distinguish reported facts from its inference, account for stale or
conflicting evidence, and abstain when the evidence is insufficient. External
article text is untrusted input, never authority to change tool instructions.

Recommendation quality needs a separate evaluation: low bankruptcy risk alone
does not imply positive investment returns. Define a recommendation horizon and
test the combined policy with historical news available at each decision time,
delisted firms, benchmark returns and trading costs. Select rules on development
data and freeze them before independent testing. Bankruptcy classification
metrics do not validate buy/hold/sell performance.

## Earlier candidate's final result

The frozen experiment selected XGBoost from development results before scoring
the separate final cohort once. Its evidence is stored in
`data/processed/validation/reserved_20261017/final_result.json`.

| Measure | Result |
|---|---:|
| Training rows | 1,290 |
| Final test rows / companies | 84 / 13 |
| Positive rows / independent positive companies | 15 / 5 |
| Decision threshold | 0.5 |
| True negatives / false positives / false negatives / true positives | 69 / 0 / 15 / 0 |
| Accuracy | 82.1% |
| Precision / recall | 0 / 0 |
| Average precision / ROC AUC | 0.769 / 0.895 |

The accuracy equals predicting no bankruptcy for every row. Ranking metrics do
not establish useful classifications or calibrated probabilities. This selected
case-control cohort is small and unrepresentative; repeated financial rows are
not independent bankruptcy events. Do not tune thresholds using this final test
or rerun it to select a better model. Its exclusive evaluation marker is retained.

## Implemented safeguards

- Production risk inference requires a locally trusted approved release record,
  artifact checksum, release-policy assessment and calibrated-model declaration.
  Current evidence fails these gates, so no production model was promoted.
- Inference checks source provenance, feature freshness, missingness, duplicate
  latest observations, future inputs and future training/release dates. Research
  mode is explicit and reports unvalidated status.
- Model writes use atomic replacement; integrity checks happen before trusted
  joblib deserialization. Checksums do not authenticate an untrusted publisher.
- Refreshes build separate version directories, validate source/QA checks and
  replace the active symlink under an exclusive publication lock. Previous
  versions survive. The initial directory migration preserves a legacy copy.
- A refresh does not retrain, rescore the final experiment or promote a model.
- Unsupported sentiment scores are unavailable rather than fabricated.
- The tested local environment has pinned versions in `requirements-local.lock`.
  CI runs unit tests, compilation, template lint and whitespace checks with
  pinned action revisions and read-only repository permission. The workflow has
  been written; it has not been executed by GitHub during this session.

`config/release_policy.json` sets initial engineering floors of 0.5 precision,
0.5 recall, 50 test companies and 20 independent positive events, plus independent
testing, population representativeness and calibration evidence. These are
conservative development gates, not a validated business-loss policy or permission
to make autonomous investment decisions. Release records remain trusted local
operator inputs; deployment identity and artifact signing remain future work.

## Local operation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.validation.update_recent
.venv/bin/python -m src.validation.production_readiness --as-of 2026-10-17
.venv/bin/python -m src.ml.release
```

Use the actual operator date for readiness checks. The last two commands exit
with status 2 while release is blocked; this is expected. `update_recent` uses
the dated configuration in `config/research_recent.json`; advance its observation
date deliberately when requesting newer data. Current refresh: 46 companies,
2,439 financial rows, 138 documents and 5,132 chunks. All 46 source fidelity
checks passed, including early-date and unknown-company abstention. These checks
do not measure answer relevance or numerical reasoning accuracy.

The active dataset is `data/processed/research`, pointing to a directory under
`data/processed/research_versions`. Preserve raw source caches and versions used
by experiments. To roll back, validate a previous complete version and call
`src.data_layer.snapshots.publish_snapshot(previous_version, current_path)`; the
same publication checks and lock apply. Investigate a leftover
`data/processed/.research_publish.lock` before removing it: a process may still
own it. Failed builds leave the active pointer intact. Symlink publication is a
local filesystem mechanism; an S3 deployment needs its own version-pointer design.

## Remaining release work

1. Collect representative, independently reviewed outcomes for the intended
   population, including recent negatives with completed horizons. The free BRD
   coverage supports qualifying large public-company negatives only in 2010–2022.
   Recent filing data alone does not create recent outcome labels.
2. Fit calibration and select a decision threshold on development/calibration
   data using explicit false-positive/false-negative costs. Freeze a new model
   and acquire a fresh independent final cohort before another release decision.
   The exposed final cohort cannot become a new untouched test.
3. Establish a human-reviewed citation/relevance benchmark and numerical Q&A
   benchmark for the research product. Accounting-risk tracks still lack complete
   reviewed negative coverage; synthesis, portfolio and backtest stubs are not
   validated product features.
4. Deliver the confirmed bankruptcy-risk and cited recommendation surface with authenticated
   access, request limits, secrets handling, audit retention, deployment/rollback,
   health checks, backups and tested incident procedures for that surface.
5. After model validation, configure AWS and test scheduled SEC polling end to
   end: IAM, encrypted storage, scheduler, checkpoint recovery, duplicate handling,
   failure alarms and actual alert delivery. Existing templates and local monitor
   tests are preparation, not evidence of a working live deployment.

Local verification: **118 tests passed** after adding release/inference and
snapshot-publication regression checks. Passing software checks does not improve
or certify the model's measured predictive performance.
