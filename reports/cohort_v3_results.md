# V3 historical bankruptcy experiment

**Production status: blocked; research only.**

The fixed acquisition, source audit, weighted model comparison and one-use final
evaluation have completed locally. No serving model was replaced and AWS was not used.

## Dataset

The 48 quarterly SEC archives define 7,929 historical companies
after exclusions. The fixed plan selected 915 companies.
Acquisition produced 1,334 calendar snapshots: 542
source-eligible rows and 792 unknown/outside-scope rows, which are not training labels.
Checkpoints: {'complete': 913, 'unavailable': 2}. Raw source files verified: 1,985.

This includes all registry case companies in the defined frame plus a 10% random
background sample. Inverse company inclusion weights apply to every snapshot.
Coverage remains restricted by public-filer history, financial-source availability,
the BRD interval through 2022 and large-public-company eligibility. It is not a
complete current-market dataset. See the [data card](cohort_v3_data_card.md).

| Stage | Rows | Companies | Positive companies |
|---|---:|---:|---:|
| Training | 180 | 81 | 12 |
| Calibration / selection | 124 | 75 | 15 |
| Final test | 238 | 75 | 5 |

Companies and time periods are disjoint across stages. Calibration uses
three company-disjoint folds. Preprocessing is fitted only on training rows.

## Fixed candidate comparison

| Candidate | Weighted out-of-company-fold log loss | Weighted Brier loss | Warnings |
|---|---:|---:|---|
| logistic_regularized | 0.067435 | 0.015204 | none |
| logistic_balanced | 0.058683 | 0.014498 | none |
| xgb_regularized | 0.084098 | 0.015857 | none |
| xgb_sqrt_balance | 0.080761 | 0.015742 | none |
| neural_network | 0.069300 | 0.015229 | Stochastic Optimizer: Maximum iterations (500) reached and the optimization hasn't converged yet. |

Selected before final predictions: **logistic_balanced**, using weighted calibration log loss
with Brier loss as the tie-breaker. All other final candidate scores remain uncomputed.
Threshold: `None`. An absent threshold means no calibration operating point
met both fixed 50% precision and 50% recall floors; it was not lowered afterward.

## One-use final evaluation

| Measure | Selected candidate | Constant calibration-rate baseline |
|---|---:|---:|
| Brier loss (lower is better) | 0.002318 | 0.002330 |
| Log loss (lower is better) | 0.018764 | 0.025002 |
| Mean probability | 1.06% | 1.61% |

Weighted observed event frequency: 0.21%.
PR-AUC: **20.75%**.
ROC-AUC: **79.40%**.
Classification precision, recall and accuracy are not reported because there is
no qualifying frozen operating threshold. Ranking scores do not replace this requirement.

These are design-weighted historical estimates, not verified 2026 market probabilities.
Unweighted audit metrics, fixed-bin calibration diagnostics and company-cluster
bootstrap intervals are retained in `final_evaluation.json`.

| Measure | Exploratory company-cluster 95% interval |
|---|---|
| brier | 0.000723 to 0.004413 |
| log_loss | 0.011091 to 0.028414 |
| average_precision | 0.002338 to 0.672593 |
| roc_auc | 0.562432 to 0.976880 |
| brier_improvement_over_constant | -0.000387 to 0.000465 |

Intervals are conditional on the observed companies and design weights. They do not
correct source nonresponse, censoring/eligibility selection or imperfect registry identities.

## Gates and remaining work

- PASS: minimum final companies.
- BLOCKED: minimum final positive companies.
- PASS: adequate calibration positive companies.
- BLOCKED: threshold exists.
- BLOCKED: precision floor.
- BLOCKED: recall floor.
- BLOCKED: precision interval lower floor.
- BLOCKED: recall interval lower floor.
- PASS: brier beats constant.
- PASS: log loss beats constant.
- PASS: independent company and time test.
- BLOCKED: current population source completeness reviewed.
- BLOCKED: external probability calibration validated.

The final test is consumed. Its results cannot be used to retune this candidate
and then reused as independent validation. A future revision needs newly reserved
companies/outcomes. Current-population outcome coverage and external calibration
review remain necessary before a production risk probability can be approved.
News-driven buy/sell recommendations also require their own dated evidence and
out-of-time strategy evaluation; this experiment does not validate trading returns.

## Evidence and commands

- Dataset SHA-256: `a4279302ac0d6bfa5380b32ecced394ebc1229954281b774feae1c436d51d658`.
- Candidate SHA-256: `4a4f14ef2575e2857e9ebe28243c2b0acefe22091254f85d02c57cdb628e3244`.
- Dataset, source seal, raw hashes, checkpoints, attrition report and frozen model:
  `data/processed/validation/cohort_v3/`.
- Frozen sampling/evaluation specification: `config/cohort_v3.json` and
  `config/cohort_v3_acquisition.json`.

Read verified saved evidence without downloading, retraining or rescoring:

```bash
.venv/bin/python -m src.validation.cohort_v3 summary
.venv/bin/python -m src.validation.local_workflow
```
