# Production follow-through — October 19, 2026

No production model was approved and no held-out company was scored. This pass fixes source admission and inference contracts, and runs additional fixed development experiments.

## Dataset audit

The reserved cohort has **20 source-corroborated issuer petitions, 315 SEC-disclosure negatives and 80 unknown outcomes**. These are automated source decisions, not independently signed-off human labels or exhaustive court coverage.
The issuer rule now recognizes a direct legal-name subject following a long deliberation clause. This recovers DZS’s explicit parent Chapter 7 petition without admitting a parent merely named as a subsidiary’s owner. The same rule was applied uniformly to development and reserved outcome evidence.
Development data stayed byte-for-byte identical after this correction: 2,174 training snapshots from 89 positive companies; 303 recent development companies, including 29 positives. The original files, prior experiment hashes and prior packets are preserved.
The updated review packet is `data/processed/validation/final_outcome_audit_v2_20261019/pending_final_reviews.jsonl`. Review responses must reference its exact packet and source-checkpoint hashes. Current independent approvals: **0**.

## Additional features and comparisons

Context features distinguish concrete going-concern, liquidity, covenant, forbearance, restructuring and financing statements from generic audit boilerplate and negation. Original documents are cached with byte hashes; extraction now resumes per document and runs with four local worker processes.
The combined experiment corrects the liquidity-negation edge case: not having sufficient cash remains a warning. Its corrections are saved in `market_context_signals_20261019/liquidity_corrections.json`; the initial contextual experiment remains archived and is not the production candidate.
Additional features use only notices published before prediction: exchange/delisting notices (Item 3.01), unregistered equity sales (3.02), auditor changes (4.01), terminated agreements (1.02), exit costs (2.05), impairments (2.06), security rights (5.03), and prior annual-filing SIC industry context. Missing item metadata stays unknown.
All candidate lists were fixed before their runs. Development companies, labels, design weights, calibration folds and the original model control stay aligned. No additional data source or paid service was used.

| Experiment | Selected research candidate | ROC-AUC | PR-AUC | Log loss | Precision available at ≥50% recall | Meets 50% P/R |
|---|---|---:|---:|---:|---:|---|
| matched_models_20261019 | hist_gradient_balanced | 0.8630 | 0.1049 | 0.06461 | 10.3% | No |
| context_models_20261019 | hist_gradient_balanced | 0.8664 | 0.1158 | 0.06389 | 11.3% | No |
| market_context_models_20261019 | hist_gradient_balanced | 0.8750 | 0.1539 | 0.06243 | 11.8% | No |

These are weighted cross-fitted development results after selection, not independent final performance or percent accuracy. A higher ranking score does not establish that a usable alert threshold exists.

## Runtime fixes

- A model predicting accounting enforcement can no longer be loaded into the production bankruptcy track. Approved SEC-disclosure models expose their precise target and scope in the response.
- Missing required feature columns produce an explicit unavailable result instead of an exception or fabricated probability.
- Registry registration and inference share the same permitted bankruptcy-target contract. Release validation, training cutoff, freshness and integrity checks remain required.
- The readiness report binds review completion to the current evidence-packet hash, preventing approval of an older packet from satisfying the current gate.

## Current release blockers

- development precision recall floor
- independent review of current packet
- untouched final performance
- recent probability calibration
- dated trading strategy validation
- live aws end to end validation

AWS remains undeployed per the existing local-first instruction. This report does not certify autonomous trading. No permission, signature or independent review has been fabricated.

## Reproduce locally

```bash
.venv/bin/python -m src.validation.final_outcome_audit_v2
.venv/bin/python -m src.validation.current_outcome_reviews
.venv/bin/python -m src.validation.context_model_development
.venv/bin/python -m src.validation.market_context_model_development
.venv/bin/python -m src.validation.production_followthrough_summary
.venv/bin/python -m pytest -q
```
