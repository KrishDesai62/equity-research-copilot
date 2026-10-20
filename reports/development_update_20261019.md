# Public-source development update — October 19, 2026

This report reads saved development results only. It does not rescore the reserved final test or approve a model for production.

## Sources and dataset changes

- SEC original nonfinancial 10-K annual-filer frames define the population. Companyfacts plus original 10-K/10-Q submission metadata supply point-in-time financial features; amended or later facts cannot replace the original prediction input.
- Primary SEC Item 1.03 search results identify candidate companies. A candidate is not a bankruptcy label: explicit issuer Chapter 7/11 petition evidence must fall inside the prediction horizon. Subsidiary-only, unclear and incomplete outcomes remain unknown.
- Negatives require continued original financial reporting across the full one-year horizon and a disclosure review through the additional 200-day reporting window. These are SEC-disclosure negatives, not proof that no bankruptcy occurred in any court.
- Training excludes all recent calibration and final company reservations. Sampling is fixed before financial acquisition; inverse inclusion weights apply to both positive and negative examples. Missing sources are not labeled zero.

Historical expansion: 703 additional companies sampled; **2,183 eligible training snapshots and 98 positive companies** in the combined dataset, compared with 1,282 snapshots and 44 positive companies before expansion. Acquisition errors: 0.

Recent development expansion: 299 new companies sampled; combined 33 positive and 274 negative companies, with 85 unknown outcomes. Acquisition errors: 0.

After the uniform issuer-identity correction, the usable expanded training dataset has **2,174 snapshots and 89 positive companies**. Corrected recent development has **303 resolved companies**, including **29 positives**. Unconfirmed former positives are unknown, never new negatives. Earlier counts above describe acquisition before this correction.

The correction rejects an issuer merely named as owner of a petitioning subsidiary, and avoids binding a prior Current Report filing date as the petition date. UpHealth, Iconic Brands and RVL subsidiary reports triggered conservative exclusions. The older final evidence packet must be audited and reissued under the same rule before final validation.

## Filing-text features

Eight features measure visible going-concern, substantial-doubt, covenant-breach, liquidity-shortfall, restructuring, auditor-change and financing-dependency language, plus document length. Counts are normalized by visible word count. Hidden XBRL, scripts and styles are excluded; access-denial pages and insufficient text fail acquisition instead of becoming zero counts.
Each document is the exact original financial filing available before the prediction date. Saved evidence includes the source URL, raw-byte hash, visible-text hash, matched terms and context proofs. Phrase frequency can reflect boilerplate and is not an asserted legal or audit conclusion.

## Completed model comparisons

All metrics below are weighted, cross-fitted DEVELOPMENT metrics after model selection. They are not untouched-test accuracy or guaranteed current-market probabilities. The two study versions use different calibration cohorts; compare phases within the same study.

| Study | Training phase | Selected candidate | ROC-AUC | PR-AUC | Log loss | Brier | Meets P≥50%, R≥50% |
|---|---|---|---:|---:|---:|---:|---|
| text_models_20261019 | base | hist_gradient_balanced | 0.7342 | 0.0533 | 0.08994 | 0.01853 | No |
| text_models_20261019 | expanded | hist_gradient_balanced | 0.8521 | 0.1722 | 0.08164 | 0.01797 | No |
| matched_models_20261019 | base | random_forest_balanced | 0.7711 | 0.0419 | 0.07177 | 0.01432 | No |
| matched_models_20261019 | expanded | hist_gradient_balanced | 0.8630 | 0.1049 | 0.06461 | 0.01399 | No |

On corrected expanded development data, the best precision available at recall of at least 50% is **11.9%** across the newly fitted candidate families. This remains below the 50% precision requirement. ROC-AUC is a ranking metric, not percent accuracy or evidence that an actionable threshold exists.

| Corrected expanded candidate | ROC-AUC | PR-AUC | Log loss | Converged |
|---|---:|---:|---:|---|
| logistic_regularized | 0.4720 | 0.0250 | 0.07774 | True |
| logistic_balanced | 0.6173 | 0.0635 | 0.07572 | True |
| xgb_regularized | 0.5812 | 0.0607 | 0.07581 | True |
| xgb_sqrt_balance | 0.8436 | 0.1166 | 0.06763 | True |
| neural_network | 0.3591 | 0.0111 | 0.07907 | True |
| logistic_c1_balanced | 0.6150 | 0.0632 | 0.07575 | True |
| hist_gradient_balanced | 0.8630 | 0.1049 | 0.06461 | True |
| random_forest_balanced | 0.8307 | 0.0642 | 0.06868 | True |
| previous_frozen_reference | 0.8081 | 0.0526 | 0.06809 | True |

Each phase compares the same nine candidate families, including logistic regression, boosted trees, random forest, a neural network and the previous frozen reference. Preprocessing is fitted on historical training data; sigmoid calibration is fitted outside each held company fold. The previous frozen reference retains its original 299-row base training history. It is diagnostic only in the corrected-label study and cannot be selected for release.

## Evaluation support and remaining gates

- Original calibration: 23 positive companies; positive effective sample size **4.62**. Largest positive-company weight share: 45.0%. This measures weight concentration, not independent outcome verification.
- Expanded calibration: 29 positive companies; positive effective sample size **28.95**. Largest positive-company weight share: 4.0%. This measures weight concentration, not independent outcome verification.
- Independent review: 0 current final units completed. Automated source corroboration is not an independent human sign-off.
- Threshold feasibility, fresh held-out performance, representative probability calibration, trading-agent profitability and live AWS operation are separate requirements. More rows or passing unit tests do not establish them.
- No production approval is granted by this update. AWS deployment remains deferred as requested.

## Local commands

```bash
.venv/bin/python -m pip install -r requirements-filing-text.lock
.venv/bin/python -m src.validation.historical_training_expansion
.venv/bin/python -m src.validation.text_model_development --phase base
.venv/bin/python -m src.validation.text_model_development --phase expanded
.venv/bin/python -m src.validation.text_model_development --phase select
.venv/bin/python -m src.validation.recent_development_expansion
.venv/bin/python -m src.validation.issuer_identity_admission_v2
.venv/bin/python -m src.validation.matched_model_development --phase base
.venv/bin/python -m src.validation.matched_model_development --phase expanded
.venv/bin/python -m src.validation.matched_model_development --phase select
.venv/bin/python -m src.validation.development_update_summary
```

Acquisition is resumable and frozen inputs are checked against hashes. Do not change a frozen plan, delete evaluation-consumption markers, fabricate review records or run two fits for the same phase concurrently.
