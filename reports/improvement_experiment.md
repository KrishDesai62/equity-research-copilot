# Expanded US model improvement experiment

**Superseded performance diagnostics:** the subsequent source audit found
changes affecting 55 old training rows, and negative outcome coverage remains
unreviewed. The figures below are historical records, not current validated
accuracy. See [the current readiness audit](model_readiness.md).

**Outcome: data coverage improved; predictive improvement was not demonstrated. Production readiness remains false.**

## Step 1: expand sources

Expanded from 10 to 36 companies. Downloaded original SEC Company Facts histories and primary SEC disclosures for 21 cited bankruptcy dates. This is a purposive case-control sample, not a representative historical investment universe.

Extracted 1914 distinct original annual/quarterly filing periods; 1588 remain after post-event and outcome-follow-up exclusions. There are 78 positive rows across 21 companies. Those rows do not represent 78 independent events.

Automated source checks found the specified event date and Chapter 11 text in each cited disclosure. They do not establish complete negative-event coverage. Negative labels remain provisional, and the diagnostic models are excluded from the production registry.

## Step 2: financial history

Added cash/assets, liabilities/assets, liquidity, operating cash flow, margins, an explicitly named earnings proxy, interest coverage, and year-over-year changes. All sources use original accessions and public filing dates. Year-to-date flows are differenced into standalone quarters, then four consecutive quarters are summed. Annual flows are used directly. Missing quarters leave TTM values missing.

| Measure on the expanded sample | Missing fraction |
| --- | ---: |
| Operating-income / assets | 24.1% |
| Explicit earnings proxy / assets | 16.4% |
| Consolidated operating cash flow / assets | 38.1% |

The earnings proxy prefers operating profit; when unavailable it uses pretax income plus disclosed interest expense. The method is recorded. It is not silently relabeled as operating income. Continuing-operations cash flow is not silently substituted for consolidated cash flow, so some cash-flow values remain missing.

## Step 3: reserve the test set

Company IDs were assigned by a seeded hash, without using outcome labels or financial values. All history of reserved companies was excluded from development. Training and development labels had to be available before 2020-01-01. The final period starts on 2020-01-01.

The development sample contains 839 rows and 5 positive companies. Final test: 79 rows from 9 reserved companies, including 5 positive companies.

## Step 4: compare features and tune on development only

Used chronological yearly windows crossed with company holdout cohorts. Imputation, scaling, and class weights were fit on each training fold. Compared the four-ratio baselines with enhanced logistic regression, three prespecified XGBoost configurations, and a small neural network.

| Candidate | Development mean fold PR-AUC |
| --- | ---: |
| baseline_logistic | 0.3388 |
| baseline_xgboost | 0.2436 |
| enhanced_logistic | 0.1629 |
| enhanced_xgb_depth2 | 0.1971 |
| enhanced_xgb_depth3 | 0.2049 |
| enhanced_xgb_depth4 | 0.2052 |
| enhanced_neural_network | 0.1373 |

Selected **baseline_logistic** on development results. Extra features and a larger model did not help in this experiment. Model, feature, and threshold choices were saved before test scoring.

## Step 5: final holdout

| Model | Test PR-AUC | Precision | Recall | Brier score |
| --- | ---: | ---: | ---: | ---: |
| baseline_xgboost | 0.7391 | 1.000 | 0.222 | 0.0735 |
| baseline_logistic | 0.6722 | 0.444 | 0.444 | 0.0678 |

The test constant-score PR-AUC baseline is 0.1139. The paired company-bootstrap interval for selected-model PR-AUC minus baseline XGBoost is [-0.2460, 0.0244]. It includes zero; this is not reliable evidence of improvement.

The threshold sought 50% precision on development predictions. That target was not achieved, so the documented fallback was 0.5. Test thresholds were not optimized. Brier scores describe uncalibrated predictions on a selected sample; they do not establish deployable bankruptcy probabilities.

## Step 6: regression and source audits

Cross-checked 6126 trailing-flow values against the independent annual-plus-YTD-minus-prior-YTD identity: 0 mismatches. Short reorganization/transition periods are excluded from full-year comparisons.

Reconciled 45 raw financial values against the original Apple and Bed Bath & Beyond inline-XBRL documents; all matched.

A transition-period correction changed 3 historical rows but **0 modeling rows**. Every development and final-test input was checked unchanged, so the reported experiment remains the same. The held-out predictions were not recomputed.

The coverage-review worksheet is saved under `data/processed/validation/expanded_quality_check/outcome_coverage_review.csv`; pending reviews were not automatically approved.

## What remains

- Independently review complete event coverage before using negative labels for production training.
- Add independent bankruptcy events across more years; development has only five positive companies.
- Audit additional source tags and use explicit continuing-operations features where consolidated cash flow is unavailable.
- Reserve new, genuinely unseen companies before the next adaptive experiment. This test set is now exposed and cannot be reused as untouched evidence.
- Evaluate text-derived features only after their as-of extraction and incremental benefit can be checked.

## Files and reproduction

Detailed data, provenance, source checks, the committed split, development results, selected configuration, and final predictions are in `data/processed/validation/expanded/`.

```bash
.venv/bin/python -m src.validation.improve
.venv/bin/python -m src.validation.improve --audit-data
.venv/bin/python -m pytest -q
```

A completed experiment is reused rather than retuned against the exposed holdout. No diagnostic model was promoted into the risk agent. Source citations are in `config/validation_expanded.json`; raw documents are cached locally.
