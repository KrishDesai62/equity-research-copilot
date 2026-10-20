# V3 historical bankruptcy dataset and evaluation protocol

This dataset supports a **one-calendar-year first-recorded bankruptcy prediction**
task for a restricted historical population of large nonfinancial public issuers.
It does not establish production readiness for all US companies or for 2026.
Actual results and release blockers are recorded in `cohort_v3_results.md` after
the committed acquisition and one-use evaluation finish.

## Sources and population

- Universe: original 10-K submission metadata from all 48 SEC quarterly financial
  statement archives, 2010–2021. The frame has 7,929 companies after the committed
  exclusions. It is a historical filer frame, not a current exchange membership list.
- Financial features: original SEC Companyfacts observations reconciled to public
  submission metadata. Later amendments, future statements and unmatched
  accessions cannot provide earlier inputs. Source URLs and SHA-256 hashes are
  retained. Sources are downloaded with the configured SEC contact and rate limit.
- Outcomes: the imported, checksum-reviewed LoPucki Bankruptcy Research Database
  release, with reviewed complete identities for the 2010–2022 interval. The
  target includes first-recorded **Chapter 7 and Chapter 11** cases. It is not a
  broader insolvency, delisting, default or trading-return target.
- Negative evidence: full calendar-year follow-up within the registry interval,
  qualifying public annual filings, complete reconciliation of expected annual
  accessions, and at least $400 million in relevant annual assets. Unknowns are
  quarantined, never silently labelled healthy. These are automated source and
  scope checks, not individual court searches for every company.
- Exclusions: known earlier cases, all companies assigned to previous experiments,
  financial issuers at the prediction date, absent/stale financial inputs and
  insufficient source/outcome evidence. Prior experiments are preserved.

Sources: [SEC financial statement datasets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets),
[SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[BRD cases](https://lopucki.law.ufl.edu/download_cases_table.php),
[BRD protocols](https://lopucki.law.ufl.edu/documentation/BRD_protocols.pdf).
The actual downloaded versions and their hashes, rather than whatever the URLs
serve later, define this experiment. Check the dataset owners' terms before
redistributing source material; no blanket redistribution license is asserted.

## Sampling and weights

The fixed plan selects 915 companies. It includes every in-frame company whose
first registry case occurs in 2012–2022, plus a deterministic 10% Bernoulli sample
of the other companies. Separate identifier hashes assign companies to training,
calibration and final evaluation. The plan is saved before financial acquisition.
No failed company is replaced, and sample seeds are not tried until metrics improve.

Each company carries its original inclusion probability: 1 for registry case
companies, 0.1 for background companies. Every snapshot receives its company's
inverse probability, including **negative snapshots of companies that fail in a
later year**. Assigning weights from a snapshot's current label would be wrong.
Training, calibration, threshold selection and primary evaluation use these
weights. Unweighted metrics remain available as an audit, not a population estimate.

This is a company-sampled historical design with repeated fixed-date snapshots.
Weighting addresses the known oversampling only. It cannot repair source
nonresponse, imperfect legal-entity matches, registry omissions, conditioning on
subsequent size/filing coverage, or the exclusion of previously exposed companies.
Attrition by role and sampling stratum is reported, not hidden. A weighted score
is not automatically a calibrated present-day market probability. The general
need for design weighting in sampled risk-model assessment is discussed in
[Derivation and assessment of risk prediction models using case-cohort data](https://doi.org/10.1186/1471-2288-13-113).
The implementation here is fixed-horizon binary prediction, not a survival model.

## Prediction units and leakage controls

The unit is `CIK@prediction_date`. Dates are December 31 anchors:

| Stage | Prediction years | Outcome horizons finish by |
|---|---|---|
| Training | 2011, 2012, 2013 | 2014-12-31 |
| Calibration and model selection | 2015, 2016 | 2017-12-31 |
| Final evaluation | 2018, 2019, 2020, 2021 | 2022-12-31 |

Companies never cross stages. Features come from the most recent original
10-K/10-Q strictly before the anchor, at most 180 days old. Same-day statements
are excluded because day-level timestamps do not establish ordering. Prediction
and feature filing dates are distinct. Labels refer to the prediction date,
not the older filing date. Negatives require the entire subsequent calendar year;
confirmed positives are observed at their event date and do not need a fictitious
post-bankruptcy annual filing. Prior recorded cases are excluded.

The target and follow-up fields, sampling stratum, identifiers, source URLs,
coverage evidence and stage assignments are never model features. Source review
may inspect later records to verify labels; later records cannot enter features.

## Features and fitting

Inputs include four Altman-style accounting components, liquidity, leverage,
cash/operating cash flow, margins, earnings proxy, interest coverage, year-over-year
changes, log assets and feature age. Quarter flow values are reconciled into
trailing-year quantities; unavailable values remain missing.

Median imputation, missingness indicators, outlier clipping and robust scaling are
fitted on training data only. A fixed five-candidate comparison tests regularized
logistic regression, balanced logistic regression, two shallow boosted-tree
variants and a regularized neural network. Design weights are normalized to mean
one during fitting, preserving relative inclusion probabilities. Optional class
balancing is a separate training choice, never an evaluation weight.

Calibration uses three company-disjoint folds within the separate calibration
cohort. For each fixed base model, sigmoid calibration is fitted on the other
companies and scored on held-out companies. Model selection uses weighted pooled
out-of-fold log loss, then Brier loss. The final selected calibrator is fitted on
all calibration companies. Saved out-of-fold predictions make this auditable.
Model selection can still make development estimates optimistic; only the later
final test supplies new evaluation evidence.

The classification threshold maximizes calibration recall subject to the fixed
50% precision and 50% recall floors. If no threshold qualifies, it stays absent;
it is not lowered after seeing final results. No synthetic outcomes are created
when a fold lacks a class. Insufficient events block validation.

## Final evaluation and release

The candidate, threshold, source evidence, dataset, environment and configuration
are frozen before final predictions. An exclusive evaluation marker is consumed
before prediction; a failed or completed evaluation does not reset it. Saved
results and predictions have a checksum receipt. Summary commands verify saved
results and never score the final set again.

Primary metrics are design-weighted precision/recall where a threshold exists,
PR-AUC, ROC-AUC, Brier loss and log loss. A constant probability estimated from
calibration data is a fixed comparator. Company-cluster bootstrap intervals retain
all snapshots and weights of a resampled company. These intervals describe
uncertainty conditional on the observed sample and do not certify source coverage.

Release checks retain at least 50 final companies, 20 independent positive final
companies, 10 calibration positive companies, precision/recall floors and beating
the constant baseline. Precision and recall interval lower bounds must also pass
the fixed floors. Current-population source review and independent external
probability calibration remain separate gates. This historical experiment cannot
approve itself for production simply by obtaining favorable metrics. It also
cannot validate buy/sell recommendations, news reasoning or trading returns.

## Reproduce locally

From the repository root, using the configured `.venv` and SEC User-Agent:

```bash
.venv/bin/python -m src.validation.cohort_v3 prepare
.venv/bin/python -m src.validation.cohort_v3_audit
.venv/bin/python -m src.validation.cohort_v3 fit
.venv/bin/python -m src.validation.cohort_v3 evaluate-once
.venv/bin/python -m src.validation.cohort_v3 summary
```

Acquisition resumes the same per-company checkpoints. `fit` verifies an existing
frozen candidate instead of retuning it. `evaluate-once` intentionally refuses a
second run. Use `summary` afterward. An interrupted evaluation is consumed and
must be investigated; deleting its marker would invalidate the protocol.

Data and artifacts live in `data/processed/validation/cohort_v3/`; public raw
sources live in `data/raw/cohort_v3/`. Large sources and trained artifacts are
ignored by Git. Retain this directory and its hashes with the experiment, use
trusted local artifacts only, and do not download arbitrary pickle/joblib models.
All commands force local cache mode. AWS configuration remains deferred.
