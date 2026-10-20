# Label-coverage audit, simple baselines and frozen research snapshot

Completed public-source discovery, broader document review, review-ledger reconciliation, two fixed logistic baselines and a matched-coverage financial-score baseline. **The existing tree model was retained. No new labels were admitted, no production threshold was released, and neither reserved cohort was scored.**

## Negative-label coverage

Inventoried **4,031 proposed-negative rows**: 3,815 training, 133 development and 83 in the fresh reservation. The metadata check found a post-horizon financial-report filing for each, but that alone does not establish complete fiscal-period coverage or absence of bankruptcy.

The selected checkpoint ledgers omitted 48,777 unique current reports. Reconciliation across ten prior ledger directories found recorded primary-body reviews for 775, leaving **48,002 reports without a body-review record in the scanned ledgers**. This is a review backlog, not evidence that those reports contain bankruptcies or that every label is wrong. Historical ledgers may also be incomplete. Financial-report bodies and court records remain separate gaps.

For the five false-alert development companies, downloaded and checked **198 additional current reports** that were outside the earlier event-review set. Four contain bankruptcy-related terms and remain flagged for independent review. The automated issuer/date parser found **zero qualifying issuer petitions within their forecast horizons**. Every downloaded document passed nonempty/access-response checks. No label was changed. Absence of a parsed assertion cannot certify absence of an event.

Evidence:

- `data/processed/validation/negative_coverage_inventory_20261019/negative_rows.csv`
- `data/processed/validation/reconciled_coverage_20261019/remaining_current_reports.json`
- `data/processed/validation/negative_coverage_audit_20261019/report.json`

## Broader public case discovery

Searched SEC full-text current reports for Chapter 7/11 phrases without the Item 1.03 restriction. Windows were June 2022–May 2023 and April 2025–October 2026, bounded by the existing outcome protocols. Search pagination, filters, source bytes and hashes were checked.

Found **380 previously unselected companies** with non-Item-1.03 matching filings. Population filtering retained **147 company/date combinations**, requiring prior nonfinancial annual-report membership. Read **251 matching primary SEC documents**, verified official submission CIK/name metadata, and applied the issuer/date parser. Acquisition completed without errors. It found **zero new in-horizon qualifying issuer assertions**. No candidate was admitted as a positive or negative, no candidate entered training and no independent approvals were invented. Search phrases frequently concern hypothetical events, subsidiaries or counterparties; complete financial-feature availability and prior-event review would also be required for admission.

The case-discovery path is implemented and reproducible, but this pass did **not** increase verified positive-company coverage. It cannot solve the fresh reservation's one-positive support shortfall.

Candidate evidence and proposals:

- `data/processed/validation/broad_bankruptcy_discovery_20261019/candidates.csv`
- `data/processed/validation/broad_candidate_review_20261019/proposals.json`

## Fixed logistic comparisons

Used expanded historical training with 3,912 eligible snapshots and 95 positive companies, the same 167 development companies, unchanged design weights, the same five company partitions and nested calibration. Both logistic models use fixed C=0.1 and class balancing. Imputation, clipping and scaling fit only on training data. No hyperparameter grid or best-seed selection was run.

| Candidate | Median weighted log loss (lower is better) | Median PR-AUC |
| --- | ---: | ---: |
| frozen_reference | 0.07258 | 0.507 |
| logistic_financial | 0.11001 | 0.079 |
| logistic_all | 0.10331 | 0.124 |

Neither logistic model's fitting folds supplied thresholds meeting both 50% precision and recall, so their held-company policies abstained. This is failure to establish a useful policy, not a successful zero-false-positive result. The original model still fails stability in one partition and remains research only.

## Financial distress-score benchmark

Implemented the nonmanufacturer Z-double-prime coefficients from [Altman's July 2000 paper, page 26](https://www.textbiz.org/projects/defaultprediction/zscores.pdf), using book net assets relative to liabilities, without the emerging-market constant. **Operating income/assets is an EBIT proxy**, so this is an explicitly labeled approximation, not an exact published-model replication or a validated bankruptcy probability.

Missing components are not imputed: **143 of 167 companies, including 30 positives**, have complete admissible inputs. On that identical covered subset:

| Raw ranking | Weighted ROC-AUC | Weighted PR-AUC |
| --- | ---: | ---: |
| Altman operating-income proxy | 0.799 | 0.069 |
| Matched tree reference | 0.923 | 0.600 |

The subset metrics must not be compared directly to full-cohort PR-AUC. Score encodings used for threshold diagnostics are not reported as probabilities. The financial-score approximation did not outperform the reference here; no claim is made about its performance in other populations.

## Frozen artifacts and remaining work

Frozen research manifest: `data/processed/validation/frozen_coverage_snapshot_20261019/manifest.json`. It binds the retained model, source tables, calibration/policy code, fixed baseline results, primary-document audit, reconciled review queue and new case triage to exact hashes. It explicitly records research-only status, no release threshold and the unresolved evidence blockers. A future study must use a new version rather than editing the frozen files.

```bash
.venv/bin/python -m src.validation.simple_baseline_development
.venv/bin/python -m src.validation.altman_baseline
.venv/bin/python -m src.validation.negative_coverage_inventory
.venv/bin/python -m src.validation.reconcile_review_ledgers
.venv/bin/python -m src.validation.broad_bankruptcy_discovery
.venv/bin/python -m src.validation.broad_candidate_review
.venv/bin/python -m src.validation.freeze_coverage_snapshot
```

Cached runs do not need AWS or paid data. Uncached discovery/document downloads require public internet access. Independent review remains pending. The original final is untouched, and the fresh cohort still lacks the 20-positive minimum. Reused development comparisons do not establish independent validation. A frozen snapshot is reproducible evidence, not production approval.

Full verification: **288 tests passed**, with the existing toy neural-network convergence warning. Tests cover train-only preprocessing, missing financial-score inputs, source issuer identity, evidence tampering, review-ledger deduplication and prevention of automatic candidate label admission. Saved reference predictions and source/model seals reproduced; no reserved-company scoring occurred.
