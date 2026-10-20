# Local reliability check — completed, release still blocked

Implemented and ran a fixed bootstrap-consensus threshold experiment locally. Each fitting fold uses 200 company bootstrap samples, the 75th percentile of feasible thresholds, and requires at least 80% feasible bootstrap samples. Held-company outcomes never determine their own thresholds. Configuration and implementation hashes were sealed before running. The prediction target, one-year horizon, reporting grace, sample weights, and 50% precision/recall requirements are unchanged.

The experiment did **not** improve reliability and was rejected. The existing model and alert policy were retained. No production threshold was released.

| Partition | Precision on covered companies | Recall on covered companies | Weighted coverage | Abstained companies |
| --- | ---: | ---: | ---: | ---: |
| 0 | 47.4% | 36.0% | 74.2% | 43 |
| 1 | 44.4% | 44.4% | 70.8% | 54 |
| 2 | 100.0% | 12.5% | 38.2% | 108 |
| 3 | 23.1% | 44.1% | 100.0% | 0 |
| 4 | 100.0% | 40.9% | 64.4% | 59 |

A precision of 100% on a small covered subset is not a reliability certificate: the policy can miss or abstain on many positive companies. All five partitions reuse the same development population, and the classifier was previously selected on its labels. Bootstrap agreement does not supply independent evidence or guarantee future performance.

The new read-only local status command verifies the saved model, data, implementations and policy-output hashes, checks whether every partition meets the diagnostic requirements with full coverage, and reports the independent-validation blockers:

```bash
.venv/bin/python -m src.validation.local_reliability_status
```

Current result: evidence integrity verified; reliability gate **failed**; final evaluation **unstarted**; production approval **false**. This command performs no network requests, AWS actions, model fitting or final-test predictions. It does not manufacture independent reviewer approval or treat missing reports as successful checks.

Tests: **273 passed**, with one existing toy neural-network convergence warning. Added tests verify held-label independence, deterministic abstention, nonfinite-score rejection, coverage-aware assessment and rejection of stability claims when one partition fails. Existing sealed studies remain unchanged.

Local implementation and diagnostics can be run without external services. Reliable predictive performance remains unproven. The next evidence work is resolved independent outcome review and genuinely independent validation, alongside further useful training examples; repeating model selection on the same companies cannot replace them. AWS does not resolve those limitations.
