# Fixed blends and held-out alert thresholds

The existing probability candidate was retained. Three fixed HGB/XGBoost blends did not improve median design-weighted log loss over the HGB model across the same five company partitions:

| Candidate | Median log loss (lower is better) |
| --- | ---: |
| Existing HGB | 0.07258 |
| XGBoost | 0.07364 |
| 75% HGB / 25% XGBoost | 0.07382 |
| Equal blend | 0.07442 |
| 25% HGB / 75% XGBoost | 0.07387 |

The original HGB predictions reproduced across all five partitions. No serving model or release threshold changed. Inputs, outputs, implementation and configuration hashes are saved under `data/processed/validation/recent_ensemble_20261019/selection.json`.

A separate policy diagnostic addresses an important distinction in prior reports. Their precision/recall operating limits were descriptive bounds computed using the scored outcomes. They do not establish that a threshold selected before seeing those outcomes would work. Moreover, pooled out-of-fold calibrated probabilities combine different fitted mappers; their common thresholds can distort raw-score ordering across folds.

The new diagnostic uses the existing classifier's **raw scores** for alert threshold selection and leaves calibrated probability reporting separate. In each of the same five partitions, each held company receives a threshold selected using only the other companies. Neither its label nor its calibration mapper chooses its own threshold. If the fitting companies supply no threshold meeting the fixed 50% precision/50% recall requirements, the policy abstains rather than silently calling the held companies negative.

| Company partition | Weighted precision | Weighted recall | Coverage |
| --- | ---: | ---: | ---: |
| 0 | 65.5% | 55.9% | 100% |
| 1 | 65.5% | 55.9% | 100% |
| 2 | 65.5% | 55.9% | 100% |
| 3 | 26.5% | 52.9% | 100% |
| 4 | 65.5% | 55.9% | 100% |

Four partitions satisfy both diagnostic floors, but the fifth fails badly on precision. These are overlapping partitions of the **same 167 development companies**, not five independent tests. The base classifier was also previously selected using this development cohort. These results therefore cannot establish production accuracy, erase selection exposure, or replace an independent final evaluation. The policy is research only; no deployment threshold is released. The reserved final remains untouched.

The apparent improvement over earlier calibrated-score threshold figures concerns the decision policy and score scale, not newly discovered outcomes or a better underlying classifier. No target, sampling weights, prediction horizon, reporting grace or precision/recall requirement changed.

Reproduce the saved studies locally:

```bash
.venv/bin/python -m src.validation.recent_ensemble
.venv/bin/python -m src.validation.recent_operating_policy
```

The policy tests explicitly verify that flipping held companies' labels cannot change their own thresholds or alerts, that infeasible fitting thresholds yield abstentions, and that repeated-company rows are rejected. Full-suite verification: 268 tests passed, with the existing toy neural-network convergence warning.
