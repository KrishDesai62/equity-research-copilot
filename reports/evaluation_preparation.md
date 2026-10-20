# Bankruptcy evaluation preparation — October 17, 2026

**Subsequent evaluation:** this cohort was scored once on October 17, 2026 and
must no longer be described as untouched. Precision/recall were 0 at the frozen
threshold; 0 of 15 positive rows were detected. See
[production readiness](production_readiness.md) for the result and release blockers.

The three requested prerequisites are implemented. This is a historically scoped
research evaluation, not proof of production accuracy for every US company.

## 1. Source-reviewed negative examples

The actual full BRD snapshot was checked: **every case in 2010–2022 has a usable
CIK**, unlike the older portion of the database. Its protocols describe complete
coverage of qualifying large, public-company debtor groups through 2022-12-31.

The reviewer now checks each prospective negative row individually:

- Its entire one-calendar-year horizon is within 2010–2022.
- An original SEC annual filing establishes public-company status through the
  horizon under the protocol's three-year rule.
- The last pre-anchor annual and every intervening original annual have at least
  $400 million in consolidated USD assets. This conservatively exceeds every
  inflation-adjusted threshold in the protocol's 2010–2022 table (maximum
  $355,163,835). The rule is not extrapolated beyond 2022.
- The submissions index contains no intervening annual missing from the
  financial extraction. An unknown annual value cannot be skipped.
- No registry bankruptcy case occurs within the horizon, no prior covered
  bankruptcy is present, and no unresolved event-date conflict exists.

This produced **1,630 negative quarterly/annual rows**. They are repeated
observations of companies, not 1,630 independent healthy firms. Evidence lists
the supporting annual accessions, minimum assets, horizon, source and review
method in `data/processed/validation/model_readiness/historical_coverage_evidence.csv`.

These labels rely on the BRD's published inclusion and group-identity rules.
They are not a nationwide court-record certification of the absence of every
kind of bankruptcy, nor do they authorize labels for small firms or 2023–2026.
An arbitrary CIK missing from the database is still insufficient by itself.

## 2. Toys R Us date adjudication

For this target, the event date means the **court docket filing date**.
The official case 17-34665-KLP docket states **09/19/2017**, agreeing with BRD.
The issuer-reported **09/18/2017** date is preserved separately rather than
discarded. This resolves the modeling convention without claiming that the
issuer and court used identical dates.

The decision, source URL, local evidence checksum, original date and selected
date are recorded in `data/curated/bankruptcy_event_adjudications.csv`. The
canonical event reconciliation applies it before label construction; checksum
or date disagreements fail validation. Older raw snapshots retain their original
values; the corrected modeling table is `model_readiness/reviewed_features.csv`.

Source: [official court docket](https://www-a.vaeb.uscourts.gov/servecal/files/megadkt/677269.html).
Coverage source: [BRD protocols](https://lopucki.law.ufl.edu/documentation/BRD_protocols.pdf).

## 3. Reserved final cohort and preprocessing

`config/reserved_cohort_20261017.json` was committed **before** fetching any of
its financial feature values. Its 16 CIKs are absent from the 46-company research
sample and the earlier modeling inputs. Eight cases were selected by a fixed CIK
hash from first BRD Chapter 11 filings in 2020–2022; eight named public companies
were prespecified as controls. This is an explicitly enriched case-control
benchmark, not a sample of natural market bankruptcy prevalence.

The historical eligibility rules yield **84 test rows from 13 companies**. Three
companies have no eligible rows in the test interval and were **not replaced**.
The manifest records the full requested cohort and eligibility losses.

- Training filing dates **and label-availability dates** must precede 2020-01-01.
- Final test filing dates are 2020-01-01 through 2021-12-31; outcome review ends
  2022-12-31.
- All history of each final-test company is excluded from development/training.
- Imputation and scaling live inside each model's training pipeline. A regression
  test verifies that an extreme test value cannot alter fitted preprocessing.
- Immutable development copies and source/data hashes freeze model selection.
  XGBoost was selected by development mean fold average precision; the threshold
  is fixed at 0.5. No final-test predictions were used in either choice.

The reserved final data are prepared and **unscored**. Development comparisons
ran for logistic regression, XGBoost and the NN with company-disjoint, purged
chronological folds. Those comparisons are not independent final accuracy.

## Commands

```bash
# Reproduce source/coverage review and development comparisons:
.venv/bin/python -m src.validation.model_readiness --evaluate

# Verify the immutable reserved cohort and preparation:
.venv/bin/python -m src.validation.reserved_cohort --prepare

# Freeze/verify choices without exposing final metrics:
.venv/bin/python -m src.validation.final_evaluation

# When conducting the final experiment, score the selected model once:
.venv/bin/python -m src.validation.final_evaluation --evaluate-once
```

The one-time runner checks hashes, company separation, time separation and both
classes, then creates an exclusive evaluation marker before fitting/scoring.
It writes results under `data/processed/validation/reserved_20261017/`, never the
production model registry. Once evaluated, the test is exposed; it must not be
reused to tune features, models or thresholds. The small case-control cohort
also cannot establish market calibration or production readiness by itself.
AWS deployment remains deferred.
