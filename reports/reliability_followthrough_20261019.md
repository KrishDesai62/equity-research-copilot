# Reliability follow-through: audited false alerts and tested two interventions

Completed locally using cached public SEC sources, without AWS, downloads or additional accounts. The model remains research only. No final test was scored and no serving model or release threshold changed.

1. Audited the five false-alert companies in the unstable partition. All have negative operating cash flow, completed reporting follow-up and no unresolved bankruptcy-disclosure accessions in the frozen checkpoint. No contrary issuer petition evidence was established; labels were retained. This is source-linked cache verification, not independent human review or an exhaustive bankruptcy-court search. See [the quarterly-trend source audit](quarter_trends_20261019.md).
2. Added nine point-in-time quarterly-trend/age features and compared two fixed refits with the unchanged classifier. The balanced refit reduced median weighted probability loss 2.2% but degraded alert precision. It remains separate probability research and was rejected as an alert replacement.
3. Mined hard negatives using company-held-out predictions within training only. Identified **176 negative training snapshots from 115 companies** with raw mining scores ≥0.5. Applied a fixed 3× loss weight to these negatives, rebalanced training class loss and fitted one refit. No development company, including the five audited false positives, entered mining or training. These are reweighted existing examples, not newly verified companies.

The hard-negative experiment uses the same original 63 features and the same five company partitions, nested probability calibration and held-company alert-threshold procedure. Configuration and implementation hashes were sealed before fitting. All comparisons below use the same 167 resolved development companies and unchanged design weights.

| Candidate | Median weighted log loss | Median PR-AUC | Held-company alert precision range | Recall among covered companies |
| --- | ---: | ---: | ---: | ---: |
| frozen_reference | 0.07258 | 0.507 | 26.5%–65.5% | 52.9%–55.9% |
| hard_negative | 0.07051 | 0.523 | 23.1%–28.6% | 38.9%–50.0% |

The hard-negative refit reduced median probability loss **2.8%**, but alert precision fell to 23.1–28.6%. Some fitting folds could not supply a threshold meeting the requirements and abstained; covered-company recall excludes those abstentions, so it must not be represented as overall recall. It was therefore also rejected as an alert replacement. Lower probability loss does not imply a better bankruptcy alert policy.

Both refits remain saved as research experiments. The existing alert classifier remains the operational reference. No target, label, prediction horizon, sampling weight, precision/recall floor or company reservation was relaxed to obtain a pass.

Reproduce the saved research experiments:

```bash
.venv/bin/python -m src.validation.quarter_trend_data
.venv/bin/python -m src.validation.quarter_trend_models
.venv/bin/python -m src.validation.hard_negative_development
.venv/bin/python -m src.validation.local_reliability_status
```

Evidence is insufficient for production reliability. These company partitions overlap and have already been used to select models; another favorable result on them would not establish independent generalization. Next evidence needs remain independently reviewed outcomes, additional informative company histories and an untouched evaluation after development choices are frozen. AWS does not supply those missing outcomes. We cannot guarantee a particular accuracy by repeating local fits until a diagnostic passes.

Verification: both experiments reproduced the original reference predictions across all five partitions, saved hashes match, and the final-evaluation marker remains absent. The full suite passed **278 tests** with the existing neural-network toy convergence warning. New tests cover future-data exclusion, reconciled original filings, missing/conflicting balance history and proof that mining never scores a company seen by its mining classifier.
