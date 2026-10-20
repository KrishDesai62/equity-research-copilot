# Local population calibration audit — October 17, 2026

No AWS connection is required. The audit ran locally using free public SEC and
the already-reviewed BRD sources. No model weights, calibration parameters or
decision thresholds were changed, and no AWS resources were deployed.

## Sampling and scope

The plan in `config/population_calibration_20261017.json` fixes 120 issuer IDs
sampled by a seeded identifier hash from SEC 2019 Q1 original nonfinancial annual
filers. It was committed before downloading company features and before looking
up sample outcomes. Selection did not balance bankrupt and healthy companies.

Two sampled companies were excluded as previously exposed. The remaining source
downloads completed. Each company contributes its first extracted original
financial filing during 2020, selected before label/coverage checks; the script
does not skip an ineligible anchor to search for a more convenient later row.

105 companies had a 2020 anchor before coverage checks. **51** passed the
historical large-public-company and full-horizon outcome rules. All other sampled
IDs remain recorded; no replacements were added. This attrition, the one-quarter
historical frame and prior-exposure exclusions limit population representativeness.
It is not a representative test of all current issuers.

## Frozen candidate's result

| Measure | Result |
|---|---:|
| Eligible companies, one observation each | 51 |
| Reviewed one-year bankruptcy outcomes | 0 |
| Mean model score | 26.2% |
| Sum of scores, interpreted as expected events | 13.36 |
| Brier loss | 0.134 |
| Log loss | 1.745 |

Five companies received scores above 0.8; none had a reviewed event in the fixed
one-year horizon. This exposes a serious calibration concern in the audited
sample. The previous case-control benchmark's 68.8% precision and 76.8% recall
remain valid for that benchmark, but they do not establish valid probabilities
in a naturally sampled population.

Zero observed events among 51 companies does not prove zero population risk.
There are no positive examples here to evaluate sensitivity or fit a useful
two-class recalibrator. Scores were not changed using this audit, and this
exposed sample must not be presented as an untouched test in another experiment.

## Data-quality correction

One source accession had a fiscal end after its filing date. The extractor now
quarantines that entire accession before historical feature calculation and
retains the reason. It does not select older values from the bad accession or
allow it to influence later trends. The sample was retried using cached sources;
the model artifact, frozen threshold and company list stayed unchanged.

## What is still needed

Acquire a substantially larger, population-based cohort with verified complete
outcomes and enough independent bankrupt companies, separating calibration from
an untouched final test. Match the population and sampling design to the product's
intended users. Fit recalibration only on its designated development/calibration
partition and assess reliability on the separate test. Do not fit on these 51
audit outcomes or reuse the previously exposed final cohorts for selection.

The current free registry establishes the reviewed historical scope only through
2022. Neither AWS nor newer SEC financial statements establish complete recent
bankruptcy follow-up. A production claim for 2023–2026 needs separate outcome
coverage. Buy/hold/sell validation also remains a distinct task.

Implementation: `src/validation/population_calibration.py`. Evidence is preserved
under `data/processed/validation/population_calibration_20261017/`, including the
sealed manifest, acquisition issues, anchors, scores and report. The exclusive
evaluation marker prevents repeated scoring. Production remains blocked.

Verification: **143 tests passed**, including outcome-independent sampling,
single-class calibration metrics and quarantine of impossible fiscal dates.
