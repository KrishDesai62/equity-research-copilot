# Detection improvement — October 19, 2026

A new research specialist improves development alert ranking and recall at a fixed precision floor. The retained probability model still has better probability loss. Neither is production-approved.

## What changed

The experiment compares native missing-value handling in gradient-boosted trees, full versus square-root class balancing, company-normalized training loss, recency weighting, stronger regularization, Extra Trees, XGBoost and two fixed probability ensembles. No additional public source was downloaded and no final companies were scored.
Company normalization reduces domination by companies with many historical snapshots. Recency weighting is a separate alternative, not an automatic policy change. Both affect training loss only: the actual dataset inclusion weights, development outcomes and evaluation weights remain unchanged.
The same 2,174 historical training snapshots and 303 development companies were used. All candidate settings were fixed before this run. The existing model’s saved cross-fitted probabilities reproduce exactly within numerical tolerance.

## Results

| Measure | Retained probability model | Research detection specialist |
|---|---:|---:|
| ROC-AUC | 0.87496 | 0.85755 |
| PR-AUC | 0.15394 | 0.22607 |
| Probability log loss | 0.06243 | 0.06370 |
| Brier loss | 0.01371 | 0.01377 |
| Maximum recall at precision ≥50% | 6.7% | 24.6% |

The specialist is **native_company_balanced**. At the development-selected diagnostic operating point, weighted precision is **79.8%** and weighted recall is **24.6%**. Raw counts: **7 bankruptcies detected, 22 missed, 1 false alert and 273 true negatives**.
These are repeatedly reused development companies, not independent final-test performance. The specialist and its diagnostic operating point were selected after inspecting this development experiment; the apparent gain may be optimistic. Raw and design-weighted precision differ because the study oversamples outcome proxies.
This improves a high-precision alerting diagnostic, not the required precision/recall pair. The unchanged production requirement remains precision ≥50% AND recall ≥50%. The specialist’s release threshold remains null.

## Full fixed candidate comparison

| Candidate | PR-AUC | ROC-AUC | Log loss | Recall at precision ≥50% | Precision at recall ≥50% |
|---|---:|---:|---:|---:|---:|
| native_design | 0.1423 | 0.8132 | 0.06865 | 13.7% | 8.1% |
| native_sqrt | 0.2231 | 0.8602 | 0.06399 | 24.6% | 11.7% |
| native_balanced | 0.2049 | 0.8638 | 0.06281 | 21.1% | 12.3% |
| native_company_balanced | 0.2261 | 0.8575 | 0.06370 | 24.6% | 10.1% |
| native_recent_balanced | 0.1195 | 0.8481 | 0.06464 | 6.9% | 6.9% |
| native_regularized_sqrt | 0.2054 | 0.8551 | 0.06461 | 21.1% | 8.0% |
| xgb_native_sqrt | 0.2087 | 0.8595 | 0.06433 | 20.6% | 11.1% |
| extra_trees_balanced | 0.0405 | 0.5505 | 0.07677 | 0.0% | 1.6% |
| frozen_current_reference | 0.1539 | 0.8750 | 0.06243 | 6.7% | 11.8% |
| native_vote | 0.1778 | 0.8721 | 0.06313 | 13.7% | 11.1% |
| diverse_vote | 0.1479 | 0.7682 | 0.07146 | 13.7% | 7.2% |

## Saved artifacts and controls

- Retained probability model: `data/processed/validation/market_context_models_20261019/expanded/candidate.joblib`.
- New research specialist: `data/processed/validation/detection_specialist_20261019/research_detection.joblib`.
- The specialist was refitted with the frozen settings; its predictions must reproduce the saved experiment before export. Its feature schema, source hashes and prediction target accompany the artifact.
- No production registry activation, release record, held-out prediction or AWS deployment was created.
- Independent outcome review, representative final probability validation, trading validation and AWS operation remain separate requirements.

## Reproduce locally

```bash
.venv/bin/python -m src.validation.detection_development
.venv/bin/python -m src.validation.detection_research_candidate
.venv/bin/python -m src.validation.detection_improvement_summary
.venv/bin/python -m pytest -q tests/test_detection_development.py
```
