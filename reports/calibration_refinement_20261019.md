# Probability calibration refinement — October 19, 2026

This pass improves the research probability model. It does not improve every detection measure and does not approve production use. The existing detection specialist remains separate.

## Why this experiment

The previous probability model overpredicted the lowest development risk band and underpredicted the highest bands. Its sigmoid calibration used a fixed regularization value of 0.1. A flatter calibration slope can suppress score differences even when the base classifier ranks companies reasonably well.
The new method chooses among regularization values 0.01, 0.1, 1 and 10 using inner company folds. Each scored outer-fold company is excluded from the outcomes used to choose its calibration setting and fit its mapper. If an inner fit has insufficient class support, the declared fallback remains 0.1.
Both the existing probability classifier and detection specialist are fixed: no base classifier was retrained, no feature was changed, and no company or evaluation weight was removed. The previous fixed-calibration scores reproduce within numerical tolerance.
Five company-fold partitions were fixed before the run. Selection minimizes median weighted outer-fold log loss across all five, rather than choosing a favorable seed. The final research mapper’s regularization is chosen on the full development cohort by inner cross-validation after the method comparison.

## Results on the original comparison partition

| Model/calibration | ROC-AUC | PR-AUC | Log loss | Recall at precision ≥50% | Precision at recall ≥50% |
|---|---:|---:|---:|---:|---:|
| probability_reference__fixed | 0.8750 | 0.1539 | 0.06243 | 6.7% | 11.8% |
| probability_reference__nested | 0.8867 | 0.1800 | 0.05829 | 6.7% | 13.2% |
| detection_specialist__fixed | 0.8575 | 0.2261 | 0.06370 | 24.6% | 10.1% |
| detection_specialist__nested | 0.8611 | 0.1939 | 0.05955 | 7.5% | 11.0% |

## Sensitivity across all five fixed partitions

| Model/calibration | Median log loss | Median Brier | PR-AUC range |
|---|---:|---:|---:|
| probability_reference__fixed | 0.06101 | 0.01360 | 0.154–0.233 |
| probability_reference__nested | 0.05732 | 0.01313 | 0.180–0.228 |
| detection_specialist__fixed | 0.06272 | 0.01373 | 0.136–0.243 |
| detection_specialist__nested | 0.05929 | 0.01344 | 0.138–0.194 |

Selected research procedure: **probability_reference__nested**. Median probability log loss improved by **6.1%** relative to the same classifier’s fixed calibration. The final full-development mapper uses **C=1.0**.
The detector’s nested calibration reduces probability loss but hurts its original-partition high-precision recall. Its earlier fixed-calibration specialist is therefore retained for detection research; probability loss and alert quality are separate objectives.

## Limits and artifacts

- All 303 companies have already been used for development. Inner/outer separation prevents this calibration step from fitting a company’s label into its own score; it cannot undo previous model and feature selection exposure.
- None of the candidates meets both unchanged 50% precision and 50% recall requirements on the original comparison partition. Independent outcome review, final evaluation and operating validation remain outstanding.
- No final features, predictions, release record or AWS deployment were created.
- Research probability artifact: `data/processed/validation/calibration_refinement_20261019/research_candidate.joblib`.
- All partition predictions, fold-specific settings, fallback decisions, source hashes and reliability bins are saved alongside the artifact.

## Reproduce locally

```bash
.venv/bin/python -m src.validation.calibration_refinement
.venv/bin/python -m src.validation.calibration_refinement_summary
.venv/bin/python -m pytest -q tests/test_calibration_refinement.py
```
