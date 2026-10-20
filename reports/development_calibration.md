# Development model repair — October 17, 2026

**Subsequent update:** the candidate below has now been tested once on a separate
47-company historical cohort, achieving precision 68.8% and recall 76.8%. See
[the independent evaluation](independent_candidate_evaluation.md). Representative
population and probability-calibration evidence remain outstanding.

The dataset and model are not fully validated for production. This change fixes
additional software defects and supplies a fitted research candidate without
reusing the exposed final test.

## What was corrected

- Missing release/as-of dates, empty provenance, missing feature-input dates and
  nonbinary model outputs can no longer silently pass production inference.
- Joblib deserialization consumes the exact in-memory bytes verified by the
  checksum, closing a file-replacement race. Only trusted artifacts may be loaded.
- Bull/bear research agents accept unavailable scores, preserve citations and
  filter by company and observation date. They no longer manufacture favorable
  or severe-risk conclusions from generic retrieval or failures. They abstain
  from investment recommendations until a recommendation policy is validated.
- News search retains dated source URLs and retrieval times, rejects undated and
  future snippets, and never counts an error message as an article. Search is an
  optional dependency; live news service operation has not been validated here.
  Search snippets are not verified full articles or validated sentiment labels.
- The model-readiness report now records the already-scored final result rather
  than incorrectly describing that cohort as untouched. Future development
  comparisons explicitly exclude both previously held-out company groups.

## Additional development data

The original fixed calibration partition had zero bankrupt companies. A separate
acquisition plan was committed before downloads, using first Chapter 11 events
from the reviewed BRD registry and deterministic company assignment. It excludes
all original development and held-out firms. Six candidates per stage were
selected from prespecified event windows; missing sources were not substituted.

All **18 planned companies** downloaded successfully. The addition contains
**423 financial rows**, of which **319** meet the historical label/size/status
eligibility checks; 15 companies have eligible one-year positive rows. Sources,
checksums and row-level annual-filing evidence are retained separately under
`data/raw/development_expansion` and
`data/processed/validation/development_expansion`.

This deliberately enriches bankruptcy examples. It does not represent market
bankruptcy prevalence and does not establish complete outcomes for 2023–2026.
The original published research dataset and frozen experiment remain preserved.

## Candidate fitting and threshold selection

The base model is the previously development-selected XGBoost. Company roles
depend on a fixed hash, with separate chronological windows for fitting,
calibration and threshold tuning. Earlier-stage outcomes must be known before
the next stage's public feature dates. Imputation/classifier fitting uses only
training rows. Sigmoid calibration uses a frozen estimator and separate rows,
following the [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).

| Stage | Rows | Companies | Positive companies |
|---|---:|---:|---:|
| Training | 285 | 19 | 4 |
| Calibration | 63 | 14 | 6 |
| Threshold tuning | 73 | 16 | 8 |

The tuning policy maximizes F2 among thresholds satisfying precision and recall
floors of 0.5. It blocks when no threshold meets them. The floors are engineering
defaults, not a validated investment loss function.

**Development selection results only:** threshold **0.15601644**, precision
**64.5%**, recall **74.1%**, F1 **69.0%**. Confusion matrix: TN 35, FP 11, FN 7,
TP 20. These metrics are optimistically selected on the tuning set. They are
neither independent accuracy nor a comparable replacement for the previous
final result. Calibration fitting is complete, but probability calibration for
the intended population remains unvalidated. Brier loss alone cannot prove it.

The candidate artifact and its checksum/threshold are frozen under
`data/processed/validation/calibration_expanded/`. They are never written to the
serving registry. Subsequent identical runs verify the frozen artifact and
return saved diagnostics. Changed inputs require a separate experiment.

```bash
.venv/bin/python -m src.validation.development_expansion --download
.venv/bin/python -m src.validation.calibration_development --include-expansion
.venv/bin/python -m src.validation.production_readiness --as-of 2026-10-17
```

The first command uses the committed acquisition manifest and cached public
sources. The last command intentionally exits 2 while production is blocked.
The original final features still match their frozen checksum, and the consumed
evaluation marker remains present. No new final-test predictions were generated.

## Still required

Acquire a new independent evaluation cohort and representative calibration
evidence with completed outcome horizons; evaluate the frozen candidate once.
Validate news reliability and buy/hold/sell performance separately with historical
prices, delisted firms, publication timestamps, benchmark returns and costs.
Neither a bankruptcy probability nor an article search result alone establishes
a profitable recommendation. AWS activation remains deferred until validation.

**Verification:** 132 local tests passed; Python compilation, CloudFormation lint
and whitespace checks passed. Local AAPL smoke checks returned five cited
passages per bull/bear agent and an explicit abstention.
