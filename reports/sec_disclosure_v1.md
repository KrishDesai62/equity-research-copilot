# Public SEC-disclosure outcome cohort — October 18, 2026

**Superseding October 18 expansion:** the [new study](sec_disclosure_expansion_20261018.md)
now has 20 final positives and 315 disclosure negatives, 1,282 usable historical
training snapshots, seven added filing features and a completed nine-model
comparison. Precision/recall and independent review remain incomplete; the final
is unscored. Earlier counts below describe the archived stage.


This is a **new, explicitly narrower prediction target**, not a relabeling of the
actual-bankruptcy V3 dataset: an issuer Chapter 7/11 petition explicitly described
in an SEC Item 1.03 filing, with the petition in the next calendar year and the
disclosure observed by horizon plus 200 days. It does not establish exhaustive
court-bankruptcy coverage, first-ever bankruptcy, healthy-company status, trading
profits or production readiness.

## Data actually acquired

Eight original [SEC financial-statement quarterly archives](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets)
for 2022 and 2024 define annual-filer membership. Exact-CIK SEC submissions,
submission archives and Companyfacts provide publication metadata and as-filed
financials. Primary 8-K and 8-K/A bodies provide event evidence. Source URLs,
SHA-256 hashes and per-company checkpoints are retained locally.

[OpenFundex](https://huggingface.co/datasets/ttchopper/openfundex), pinned revision
`a12f20990b4540ecc3db0e0294e985b7c81f58e9`, supplies candidate-company strata,
**not accepted outcome labels**. Only its training file was opened; its validation,
test and live files remain unopened. Its original 151 alias rows represent 119
unique company/date/chapter events. The stronger primary-source parser corroborates
83 issuer events across that candidate queue; 36 remain unresolved. These queue
counts differ from admitted prediction snapshots below.

The acquisition plan was saved before feature acquisition and model fitting:
`config/sec_disclosure/acquisition_v1.json`. It selected **498 companies**, excluding
prior model-study companies. Calibration snapshots predict 2023 from December 31,
2022; final snapshots predict 2025 from December 31, 2024. Reporting grace periods
are complete as of this run. This provides outcome coverage through **2025**, not
completed one-year outcomes for predictions made during 2026.

| Cohort | Financial snapshots | Positive | Negative | Unknown |
|---|---:|---:|---:|---:|
| Calibration | 176 | 20 | 121 | 35 |
| Reserved final | 148 | 9 | 109 | 30 |
| Total | 324 | 29 | 230 | 65 |

The remaining **174 selected companies** are exclusions: 113 stale/missing feature
sets, 55 without annual membership in their assigned year's frame, four known
prior bankruptcies and two without original USD Assets facts. Those last two
initially appeared as acquisition errors; an independently checked availability
resolution records the exact missing source facts. The original acquisition report
is preserved. No outcome labels were assigned or changed by that resolution.
There are no unresolved acquisition errors after this availability audit.

## Admission and weighting

A positive requires a dated petition statement, Chapter 7/11 and explicit issuer
participation. Chapter 15, involuntary and prospective statements are not admitted.
The parser binds dates to actions, preventing an earlier agreement date from being
assigned to a later petition. Subsidiary-only or ambiguous statements remain
unknown. This is conservative automated corroboration, **not two independent
human reviews**.

A negative requires complete metadata acquisition, resolved potential Item 1.03
disclosures, no qualifying issuer event, at least four distinct post-anchor fiscal
periods, a report covering the full outcome horizon and no reporting gaps over
180 days. Original 8-Ks with empty item metadata and 8-K/A bodies in the observation
window are checked rather than silently skipped. Unindexed bankruptcy language
is unresolved, not automatically negative. These are **SEC-disclosure negatives**;
absence of a court petition has not been proved.

Inputs are original 10-K/10-Q financials published strictly before prediction,
no more than 180 days old, with at most 25% missing among the 21 features. The
features cover profitability, liquidity, leverage, cash flow, growth, accruals,
a net-liability proxy, cash burn, size and input age. Imputation, clipping and
scaling are fitted on training data only. Future reporting is used for outcome
coverage, never for the prediction features.

This is a **proxy-assisted case/background design**, not an outcome-blind random
population sample. Candidate companies are sampled with certainty; background
companies with probability 0.1. Role probabilities depend on candidate year but
are strictly between zero and one. Every snapshot, including a negative snapshot
of a future candidate, receives inverse **sampling × role** probability weight.
Source-review exposure is documented separately from model-training exposure.
The original stricter outcome-blind reservation workflow is unchanged. Weights
address known design sampling, not follow-up censoring, legal-entity mistakes,
source nonresponse or population drift. Current-market calibration is not certified.

## Models actually fitted

Historical V3 development rows supply training, excluding its consumed final
cohort. Historical positives require primary issuer-petition corroboration before
reuse for the disclosure target. Five unmatched positive rows were excluded;
**299 training rows with 22 positive companies** remain. Historical negatives
retain their restricted BRD source-coverage assumptions; they are not newly
court-reviewed negatives. The [LoPucki BRD protocols](https://lopucki.law.ufl.edu/documentation/BRD_protocols.pdf)
limit that historical coverage through 2022. The old V3 dataset, artifact, final
predictions and frozen implementation dependencies were preserved.

Five predefined models were compared with company-fold, design-weighted
calibration on **141 resolved recent calibration companies, including 20 positives**.
The neural model uses the predefined 2,000-iteration budget. All five converged.

| Model | Development ROC-AUC | Weighted PR-AUC | Weighted log loss |
|---|---:|---:|---:|
| Regularized logistic | 0.8380 | 0.0356 | 0.04901 |
| Balanced logistic, selected | 0.8362 | 0.0332 | 0.04814 |
| Regularized boosted trees | 0.4487 | 0.0086 | 0.05436 |
| Balanced boosted trees | 0.7947 | 0.0809 | 0.05188 |
| Neural network | 0.8537 | 0.0372 | 0.04860 |

Selection followed the precommitted lowest weighted cross-fitted log loss, then
Brier rule; final labels/scores did not select the model or threshold. Weighted
calibration prevalence is 0.9594%. Model selection still makes development
estimates optimistic. Results are not directly comparable with the old V3 target
and cohort, and are not a new final accuracy rate. **No candidate threshold met
both 50% precision and 50% recall.**

The final cohort has **118 resolved companies but only nine positive companies**.
The evaluator correctly blocked before predicting, preserving the test without a
consumption marker. The required minimum is 20 independent positives. Do not
retune or change a sampling seed to force a favorable score. Broader precommitted
public-source coverage, independent label review, stronger development detection,
separate dated trading validation and live operational tests still remain.
The new artifact is research-only and was not registered for serving.

## Run and inspect locally

```bash
.venv/bin/python -m src.validation.sec_disclosure_data
.venv/bin/python -m src.validation.sec_disclosure_experiment
```

SEC acquisition requires the configured contact header and network access for
uncached sources. No AWS account is required. Repeated runs validate/reuse source
checkpoints and frozen selection instead of refitting a completed selection.
The experiment preserves incomplete fitting for a resumable retry, verifies source
and dependency hashes, and uses a cohort-wide exclusive marker before any eligible
final prediction. Missing-data exclusions cannot bypass a real network/parser error.

Outputs are in `data/processed/validation/sec_disclosure_v1/`: `dataset.csv`,
`report.json`, `availability_resolution.json`, company checkpoints and
`experiment/selection.json`, `training.csv`, `training_sources.json`,
`calibration_predictions.csv`, `candidate.joblib`, and `evaluation_blocked.json`.
No final predictions were generated. AWS deployment remains deferred.

Verification completed: **204 tests passed**, source compilation and whitespace
checks passed. One existing 500-iteration neural toy-test fixture warns about
convergence; all five actual new experiment fits converged. A second experiment
run verified/reused selection SHA-256
`9df0e295182338a2d5b20047dc7aa6365ba3d0bd17a5b67bcb8e92615179b6bf`
and artifact SHA-256
`1efeff15d727a2e2d58d425484233ce73a406a8436924f0590a2a818b594280c`.
No final prediction file or consumption marker exists. All sealed V3 dependency
hashes still match. Passing software checks does not establish predictive accuracy.
