# Current US model validation status

**Historical labels are available within reviewed coverage. The final cohort has been scored once: precision 0.688, recall 0.768. It is exposed and must not be reused for tuning. Evidence: data/processed/validation/independent_candidate_20261017/final_result.json. See reports/production_readiness.md; production release remains blocked. Previous accuracy figures remain superseded diagnostics.**

SEC submissions now filter raw inputs before historical features are computed, preventing amendments and conflicting dates from influencing later ratios.

Eligible one-year labels: 88 positive rows across 24 companies, 1630 reviewed negative rows. 721 rows remain unlabeled.

Negative outcomes are source-reviewed only within the BRD's large-public-company scope and 2010–2022 interval. Row-level annual filing and asset evidence is exported separately; other unknown outcomes stay unknown.

The audit in `data/processed/validation/model_readiness/report.json` lists affected old model inputs without rescoring the exposed holdout. Existing experimental artifacts and scores are preserved as historical records, not current validated performance.

Complete cited bankruptcy-event coverage in `data/processed/research/outcome_coverage_review.csv`: coverage_start, observed_through, reviewer, coverage_source_url, and event_coverage_reviewed. A citation to a financial filing alone does not establish complete bankruptcy follow-up. Missing required evidence fails validation.

After adequate reviewed examples of both classes exist, run company/time-separated development comparisons for logistic regression, XGBoost and the neural network. Reserve genuinely unseen firms before a final evaluation; the previous holdout cannot supply new independent evidence. Accounting enforcement and financial Q&A are separate evaluation tasks.
