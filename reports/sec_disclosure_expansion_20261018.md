# Expanded public outcomes and development results — October 18, 2026

The recent final cohort now meets the **20-positive-company count minimum**:
**20 source-corroborated issuer petitions and 315 completed-follow-up disclosure
negatives**. It remains unscored because no development model meets both fixed
50% precision and 50% recall requirements. Neither an AWS deployment nor a
production model promotion occurred.

## Public data actually acquired and corrected

- Queried the [SEC's own full-text search](https://www.sec.gov/edgar/search/) for
  Item 1.03 disclosures filed January 1, 2025–July 19, 2026. Verified the returned
  root-form/item/date filters, exact result count, pagination and unique IDs:
  **126 primary filing hits, 83 candidate companies**. Root-form 8-K search includes
  amendments. Search hits are candidates, not automatically bankruptcy labels.
- Found 30 inventory candidates in the previously unsampled 2024 annual-filer
  population. Before acquiring their financial inputs, froze a **362-company**
  complementary case/background plan: all 30 candidates and 332 independently
  hash-sampled background firms. Original calibration companies were excluded.
- Acquired **267 new financial snapshots**: initially six positives, 206
  disclosure negatives and 55 unknowns. Another 95 selected companies were
  excluded for stale/missing financial features or known prior bankruptcy.
- Fixed one acquisition failure caused by an unrelated Form 4 with an empty
  document path. Its original source is preserved; the outcome-specific recovery
  skips Form 4 while retaining normal validation of 10-K, 10-Q and 8-K records.
  All acquisition errors are resolved. No company was substituted.
- Applied versioned primary-evidence corrections uniformly to historical training,
  recent calibration and final data: singular case/proceeding language; explicit
  later confirmations within 200 days; Chapter 7 liquidation wording; and trailing
  SEC state annotations such as `/DE` in names. Date/action binding, explicit
  issuer participation, chapter scope and public-date constraints remain required.
  Agreement dates, subsidiary-only statements, prospective actions and year/date
  disagreements were not guessed into positives.

| Recent cohort after source corrections | Rows | Positive | Negative | Unknown |
|---|---:|---:|---:|---:|
| Calibration, predicting 2023 | 176 | 23 | 121 | 32 |
| Combined reserved final, predicting 2025 | 415 | 20 | 315 | 80 |

These are **automated source-corroborated SEC-disclosure outcomes**, not exhaustive
court-bankruptcy outcomes or independent human approvals. Negative admission
still requires continued original financial reporting covering the complete
calendar-year horizon, sufficient distinct reporting periods, bounded reporting
gaps and no unresolved potential qualifying disclosure. Unknowns stay missing.
The 200-day reporting grace is complete for the 2025 horizon; completed full-year
outcomes for predictions made during 2026 are not claimed.

Original datasets and reports remain intact. Raw sources, per-company checkpoints,
publication dates, hashes, original labels and every admission change are retained.
The separately versioned source policies and acquisition plans are under
`config/sec_disclosure_expansion/`.

## Correct weighting of the combined final pool

Appending only known cases to the old final sample would bias evaluation. The
new table applies **marginal union inclusion probabilities to both phases**:

- Original non-proxy companies: `0.1 × 0.5 + 0.9 × extension_probability`.
  This is **0.95** for a SEC-inventory candidate and **0.14** for background firms.
- Original certainty-proxy companies retain their original final-role inclusion
  probability; they cannot be re-enrolled after being assigned to calibration.
- Every row receives the inverse union probability regardless of acquisition
  phase, actual label or whether it later becomes unknown.

The original V1 table/weights remain unchanged; the combined study is explicitly
new. Known design weights do not correct nonresponse, follow-up censoring,
legal-identity errors or source-index omissions. This remains a proxy-assisted
sample, not certified current-market population coverage.

## Larger historical training build

Reused sealed raw public financials and submissions for the 915 historical study
companies, constructing new year-end snapshots for **2014–2020**. The old historical
final companies were explicitly retired into development for this separate study;
none of their old model predictions was read or rescored. All recent calibration
and final reservations are disjoint from these historical companies.

The label was rebuilt from the **same SEC-disclosure definition**, rather than
assigning negatives from absence in BRD. Historical BRD information is used only
for the original sampling stratum and known-prior-event exclusions. This removes
the broader historical negative-target assumption from the expanded training set.

The build produced **1,595 snapshots**: **1,282 usable training rows from 390
companies**, including **44 positive companies**, and **313 unresolved rows**.
The earlier matched-target training set had 299 rows and 22 positive companies.
All expanded outcomes are observable before the December 2022 calibration
prediction date. Financial inputs remain original, reconciled public filings
strictly before prediction, no more than 180 days old, with the fixed missingness
limit. No calibration/final preprocessing was fitted.

## Seven actual filing features added

Reconstructed from the cached SEC submission arrays for training and calibration:

1. Late 10-K/10-Q notices in the prior 365 days.
2. Material-default Item 2.04 disclosures in the prior 365 days.
3. Material-default Item 2.04 disclosures in the prior 90 days.
4. Non-reliance Item 4.02 disclosures in the prior 365 days.
5. Executive-change Item 5.02 disclosures in the prior 90 days.
6. Current-report volume in the prior 90 days.
7. Days since the latest original annual report.

Only filings strictly before the prediction date enter these features. Missing
item indexing makes item counts missing rather than zero. These are disclosure
signals, not proof of misconduct or an investment recommendation. Final feature
rows were **not enriched or scored**.

The model comparison uses **28 input features**, including the original 21
financial features. A new preprocessing regression prevents the 99th-percentile
clipper from turning a rare event flag into zero when it occurs in less than 1%
of training rows. Financial outliers are still clipped using training-only
bounds; sparse event features and missingness indicators are preserved.

## Actual model results; no improved-accuracy claim

Ran expanded financial-only comparisons, then the fixed filing-feature comparison.
Logistic iteration budgets were raised to 10,000 after observed nonconvergence;
neural candidates retain 2,000 iterations. All nine candidates in the latest
comparison converged. Every model receives the same company-fold, weighted
calibration on **144 resolved calibration companies, including 23 positives**.
Thresholds and model choice use development data only.

| Latest comparison | Development ROC-AUC | Weighted PR-AUC | Weighted log loss |
|---|---:|---:|---:|
| logistic_regularized | 0.4397 | 0.0245 | 0.09735 |
| logistic_balanced | 0.5155 | 0.0260 | 0.09821 |
| xgb_regularized | 0.3559 | 0.0162 | 0.10009 |
| xgb_sqrt_balance | 0.5003 | 0.0300 | 0.09685 |
| neural_network | 0.3510 | 0.0197 | 0.10053 |
| logistic_c1_balanced | 0.5333 | 0.0282 | 0.09911 |
| hist_gradient_balanced | 0.6478 | 0.0387 | 0.09295 |
| random_forest_balanced | 0.5109 | 0.0593 | 0.09604 |
| previous_frozen_reference | 0.7678 | 0.0454 | 0.09096 |

Exact values are stored in the selection JSON; this rounded table is descriptive.
The selected reference retains its **original 299-row training base** and was
recalibrated in the same new folds; it ignores the seven added inputs through an
explicit feature-subset adapter. The new 1,282-row set was used to fit the expanded
candidates, not to retrain this reference. More data/features did not establish
better detection, so none of those candidates replaced it in production.

The earlier 0.836 development ROC-AUC used 20 calibration positives. The corrected
cohort has 23; 0.7678 is the current reference result on that expanded cohort.
These are different development evaluations, not independent final accuracy rates
or evidence of a production regression/improvement. Repeated development selection
is optimistic. **No candidate meets both fixed precision and recall floors.**
The final remains unconsumed; no final probabilities or new final accuracy rate
were generated. No release approval is inferred from reaching 20 positives.

## Independent review packet and remaining gates

Generated a deterministic evidence packet for **335 final review units** (20
positives, 315 negatives), excluding the 80 unknowns. It contains exact CIKs,
prediction/horizon/cutoff dates, source checkpoints, hashes, positive-statement
proofs and source links. It includes no model predictions. Reviewer identities,
approved labels and review statuses remain empty/pending—no independent reviewers
were invented. A compact positive-source CSV accompanies the JSONL packet.

Remaining substantive gates: independent label review, demonstrably stronger
precision/recall and probability validation, one frozen final evaluation, separate
dated trading-agent tests including costs/delistings, and live AWS scheduling,
authentication, alerts and recovery. AWS remains deferred as requested. The next
model-development hypothesis should address the remaining limited bankruptcy
training coverage and domain shift; repeating a larger hyperparameter search on
these 23 development positives is not a substitute for new evidence.

## Reproduce and inspect

```bash
.venv/bin/python -m src.validation.sec_event_inventory
.venv/bin/python -m src.validation.sec_disclosure_extension
.venv/bin/python -m src.validation.sec_disclosure_training
.venv/bin/python -m src.validation.sec_disclosure_admission_v3
.venv/bin/python -m src.validation.sec_disclosure_models_v4
.venv/bin/python -m src.validation.sec_disclosure_review_packet
```

Uncached acquisition requires SEC network access and the configured contact header.
Model and feature commands verify/reuse frozen outputs. The one-time, documented
Form 4 recovery is `src.validation.sec_disclosure_repair`; it is not a general
validator bypass. Source evidence is under the corresponding
`data/processed/validation/sec_*` directories. Old V1 and historical V3 outputs
are preserved and remain independently inspectable.

Verification completed: **228 tests passed**, compilation and whitespace checks
passed. One existing 500-iteration neural toy fixture emits a convergence warning;
all nine actual latest comparison fits converged. A second run verified/reused the
frozen selection and review packet without fitting or scoring the final cohort.
Latest selection SHA-256:
`0ee310a9531f979565a28972e274d4526b3d4bb4152f175a164708ae54f4385a`.
Latest research artifact SHA-256:
`b9a57342e10373b4d0a7a81d76bbb6fe35ec9355fbca206b9c4ac7ef39953921`.
The recent cohort consumption markers and final prediction files do not exist.
All archived historical V3 implementation dependency hashes remain unchanged.
These checks establish reproducibility/software behavior, not production accuracy.
