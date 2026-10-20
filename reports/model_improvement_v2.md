# Local model improvement experiment — October 17, 2026

Completed: additional historical issuer acquisition, five model variants, two
calibration choices, a frozen candidate and a one-time comparison on new firms.
No AWS resources were used and no serving model was replaced. The result remains
**research only**; this experiment does not establish better bankruptcy detection.

## Added data

A fixed plan sampled 150 nonfinancial annual issuers from each of SEC 2012,
2014 and 2016 Q1 archives. After deduplication, this produced **435 issuer IDs**.
Selection used identifier hashes, excluded prior experiments and did not consult
bankruptcy outcomes or model scores. Each issuer enters only from its actual
historical sampling filing date, preventing later inclusion from authorizing
earlier training observations.

All source acquisitions completed without terminal errors. There are **2,667
post-entry annual rows**, of which **1,356** pass the frozen historical eligibility
rules. Sources, hashes, extraction issues and resumable per-company checkpoints
are preserved. History after entry was available for 432 of the 435 companies.
The corpus remains conditional on annual-filer sampling, source availability,
large-public-company rules and prior-exposure exclusions; it is not all-market
or recent-population ground truth.

## Training and tuning

The bounded search compared:

- Regularized logistic regression, with and without balanced class weights.
- Regularized shallow XGBoost, unweighted and with square-root class weighting.
- A smaller neural network with stronger regularization.

Preprocessing uses training-only median imputation, missing-value indicators,
quantile clipping and robust scaling. Repeated observations receive weights so
each training company contributes equal total weight. XGBoost uses two CPU
threads. The neural network avoids random early-stopping validation splits.

Each model was compared with raw scores and with a separate intercept-only
log-odds correction fitted to natural calibration observations. A fixed
pseudocount prevents zero observed rare events from asserting zero risk. This is
an uncertain research estimate, not validated calibration. The implementation
follows the separation principle in the
[scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html);
tree regularization controls are described in the
[XGBoost parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html).

| Stage | Companies | Rows | Positive companies |
|---|---:|---:|---:|
| Training | 81 | 720 | 4 |
| Natural calibration, 2016 | 36 | 36 | 0 |
| Natural model selection, 2018 | 41 | 41 | 0 |
| New final comparison, 2020 | 39 | 39 | 0 |

Training also uses the existing positive-enriched development data, restricted
to filings before 2015 and outcomes available before 2016. Calibration, selection
and final companies are separate from training and from each other. Those stages
use one annual anchor per issuer and fully completed one-year horizons.

The selection objective, fixed beforehand, was natural-sample log loss with
Brier loss as a tie-breaker. It chose **balanced regularized logistic regression
with intercept correction**. Because the selection sample has no positive
events, no classification threshold was selected. No model can be certified as
a useful detector from this selection sample.

## Frozen comparison on the same 39 new companies

The selected artifact and comparison protocol were frozen before final scores.
The earlier candidate and a constant rate estimated from calibration data were
fixed comparators. All final observations were negative under the reviewed rules.

| Measure | New research candidate | Previous candidate | Constant calibration-rate baseline |
|---|---:|---:|---:|
| Mean score | 2.43% | 19.83% | 2.63% |
| Brier loss, lower is better | 0.02201 | 0.06344 | **0.00069** |
| Log loss, lower is better | 0.06743 | 0.74365 | **0.02667** |

The new candidate improved probability losses relative to the previous candidate,
but **the constant baseline was better**. The final sample contains no events,
so it cannot establish recall, discrimination or positive-risk calibration. No
100%-accuracy claim is made. The machine-readable assessment explicitly rejects
production readiness and does not promote a model based on these results.

The new final comparison has been consumed. It must not be rerun or used to
choose another model. Earlier final cohorts were not rescored.

## Additional coverage corrections

Source review found an eligibility bug: a confirmed registry bankruptcy could be
excluded because an annual filing after the bankruptcy was missing. The BRD
protocol defines qualifying assets/public status before the case. Future coverage
review therefore stops a verified positive at its registry event date, while a
negative still requires its entire one-year horizon. Same-day filing/event
ordering remains unreviewed when only calendar dates are available.

Coverage is also capped by the source observation endpoint, so a later registry
snapshot cannot authorize negative outcomes beyond the supplied filing history.
A date-column conversion fixes reviewed-label assignment on saved pandas CSVs.

A separate audit changed eligibility for four development rows and exported
**1,638 reviewed development rows across 198 companies**, with **19 positive rows
across six companies**. It excludes prior held-out firms and all V2 final-role
companies. Outputs are in `data/processed/validation/coverage_revision_v2/`.
This corrected export is for future development, not a silently replaced input
to the measured V2 experiment. Archived feature files and evaluations remain
unchanged. V2 retains its original stricter coverage rule for reproducibility.

## Using the completed work

```bash
.venv/bin/python -m src.validation.local_workflow --train-v2
.venv/bin/python -m src.validation.improve_v2 --summary
```

These commands verify the frozen candidate/read saved results without predicting
on the exposed final cohorts. Data preparation can be resumed with
`src.validation.improve_v2 --prepare`; completed checkpoints and sealed data are
verified and reused. `--evaluate-once` refuses after its evaluation marker exists.

Evidence lives in `data/processed/validation/improvement_v2/`: acquisition and
development reports, candidate/checksum, final predictions, final result and
research assessment. No approved release manifest was created.

## Remaining constraint

The bottleneck is independently reviewed bankruptcy events in population-based
calibration/validation data, not the number of optimization runs. More negative
rows alone cannot establish detection quality. The reviewed registry still ends
in 2022; recent filings do not establish completed recent outcomes. A future
candidate needs enough independent events, a comparison against simple baselines
and a fresh final cohort before any production or investment-recommendation claim.

Verification: **162 tests passed**, along with compilation, CloudFormation lint
and whitespace checks. The local `--train-v2` workflow completed without AWS or
final-test rescoring. Earlier sealed feature checksums remain unchanged. The tiny
neural-network unit fixture emits an iteration-limit warning; the full experiment's
neural-network fit recorded no warnings. No claimed performance depends on that
synthetic fixture.
