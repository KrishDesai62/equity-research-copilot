# Independent research-candidate evaluation — October 17, 2026

The fitted research candidate has now been evaluated once on new companies at
its frozen threshold. It performs usefully on this historical benchmark, but it
is not approved for production or autonomous buy/hold/sell decisions.

## Design and isolation

- The artifact checksum and threshold **0.15601644123520086** were frozen before
  the new cohort was selected or downloaded. There was no refitting or tuning.
- The plan in `config/independent_candidate_20261017.json` was committed before
  downloading company features: 40 first BRD Chapter 11 cases in 2020–2021 and
  40 controls from original annual filings in the SEC 2019 Q4 archive. Selection
  used a deterministic identifier hash. Controls were not selected from today's
  surviving ticker list. Controls exclude financial SICs; case/control sector
  comparability is not established.
- All original development, calibration, tuning and previously held-out company
  identifiers were excluded. Test filing dates are 2020–2021, with outcome
  observation through 2022-12-31 and no future feature inputs.
- Sources downloaded for all 80 planned companies. **47 had eligible rows** after
  the existing BRD size, identity, public-status and outcome-coverage checks.
  No replacement companies were added. The 33 losses are listed in the sealed
  manifest. The remaining sample includes 31 case-role and 16 control-role firms;
  30 firms have positive rows in the fixed prediction window.
- An exclusive evaluation marker was created before prediction. The evaluated
  cohort is now exposed and must not be reused to select models or thresholds.
  Both the earlier experiment and this candidate remain preserved separately.

## Results at the frozen threshold

| Measure | Result |
|---|---:|
| Test rows / companies | 206 / 47 |
| Positive rows / independent bankruptcy companies | 69 / 30 |
| Precision | 68.8% |
| Recall | 76.8% |
| F1 | 72.6% |
| Accuracy | 80.6% |
| Always-no-bankruptcy accuracy | 66.5% |
| Average precision / constant-score baseline | 0.782 / 0.335 |
| ROC AUC | 0.864 |
| True negatives / false positives / false negatives / true positives | 113 / 24 / 16 / 53 |
| Bankruptcy companies with at least one positive-window alert | 25 / 30 |
| Brier loss | 0.149 |

The event-detection fraction (25/30) asks whether any positive-window row alerted;
it differs from row-level recall (53/69). Repeated rows are not independent events.

Probability bins show remaining mismatches: for example, the 21 rows scored
between 0.2 and 0.4 have mean score 0.235 and observed event fraction 0.857. Even
in this selected sample, the scores should not be presented as validated event
probabilities. Brier loss mixes discrimination and calibration and cannot certify
calibration by itself.

## Limits and release decision

This is an intentionally enriched case-control sample, with substantial
eligibility attrition and possible sector/selection effects. Its precision,
accuracy and outcome prevalence do not estimate those quantities for all listed
companies or for 2026. Scope remains the reviewed historical large-public-company
registry interval. No reviewed 2023–2026 negative coverage was created.

The release policy still blocks promotion because:

1. Only 47 companies survived eligibility checks, below the unchanged 50-company
   floor. Do not add three companies after viewing results just to pass a gate.
2. Representative-population evidence is absent.
3. Probability calibration for that intended population remains unvalidated.

Buy/hold/sell quality has not been evaluated. That requires a separate fixed
recommendation policy and timestamped news/price benchmark, including delistings,
comparison returns and trading costs. AWS is still undeployed as requested.

## Evidence and checks

Outputs are in `data/processed/validation/independent_candidate_20261017/`:
`final_result.json`, `predictions.csv`, `coverage_evidence.csv`,
`acquisition_report.json`, `sealed_manifest.json`, and `evaluation_started.json`.
The candidate artifact is still outside the serving registry.

The workflow is implemented in `src/validation/independent_candidate.py`.
`--prepare` reuses the sealed cohort; `--evaluate-once` refuses to predict again
after the marker exists. Never delete the marker to rerun an experiment.
`src.validation.production_readiness` now reports this independent evidence and
checks it against the saved candidate checksum. Exit status 2 remains expected
while the release gates fail.

Local verification: **135 tests passed**, including company/time leakage,
full-horizon negatives, duplicate periods and prevention of repeated scoring.
Compilation and whitespace checks passed.
