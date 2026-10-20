# Newer public-data study — completed research comparison

This study is not a production approval or a new accuracy result. Source acquisition, exact historical filing feature extraction and all five candidate fits completed without recorded errors. A refitted balanced histogram gradient-boosting model was saved as a research candidate. Existing models and the reserved final evaluation remain unchanged.

The public sources are SEC financial-statement bulk archives, EDGAR company facts and submissions, original annual/quarterly reports, and issuer bankruptcy disclosures in current reports. SEC search results identify candidates; they do not establish a bankruptcy label by themselves. Source bytes, URLs, hashes, company reservations and sampling probabilities are saved locally.

The frozen plan adds 227 potential training companies to the existing 1,618 selected IDs. Previously selected companies receive an additional May 31, 2021 snapshot; newly selected companies can contribute earlier historical snapshots too. Acquisition produced 3,904 training rows: 3,005 eligible snapshots (91 positive companies) and 899 unknown rows. The eligible snapshot count increased from 2,174, but independent positive-company support increased only from 89 to 91. Earlier training rows receive the updated union inclusion probability. Repeated snapshots are not independent bankruptcy companies.

The newer development sample selects 286 of 2,368 eligible companies after training, legacy pilot and reserved-final exclusions. Acquisition completed without recorded errors: 207 rows comprise 34 positives, 133 disclosure negatives and 40 unknowns; other selected companies lacked usable input coverage. Of the 34 positive companies, 18 were also positive in the prior development cohort and 16 were not; these are overlapping development studies, not 63 independent positive companies. Prior development companies may recur at this later date. The sample includes a fixed random background and disclosure-based enrichment, with explicit sampling weights. It represents this conditional eligible population, not every listed company.

Dates preserve the complete one-year target and 200-day reporting grace:

| Stage | Forecast | Horizon ends | Evidence cutoff |
| --- | --- | --- | --- |
| Added training date | 2021-05-31 | 2022-05-31 | 2022-12-17 |
| New development | 2023-05-31 | 2024-05-31 | 2024-12-17 |
| Reserved final | 2024-12-31 | 2025-12-31 | 2026-07-19 |

Training outcomes mature before development forecasts; development outcomes mature before the reserved final forecast. Later 2026 filings cannot provide completed future one-year outcomes for 2026 forecasts.

The target remains an issuer Chapter 7/11 petition disclosed in SEC filings within one year. A disclosure negative requires completed follow-up under the stated protocol; incomplete evidence remains unknown. This does not prove absence of every court bankruptcy.

The fixed comparison tests two frozen classifiers and three refitted candidates using the same 63 financial, filing-text, filing-event and industry features on the same newer cohort. Nested calibration excludes scored companies from mapper fitting and regularization selection. Five fixed company partitions measure sensitivity. No final features or predictions are used. Candidate configuration and implementation hashes are committed locally before model comparison.

A completed descriptive audit of the existing 29-positive development sample reproduced 22 missed cases at the archived diagnostic threshold: 18 had contextual warning phrases and 16 had missing model features. These overlapping descriptions do not establish a causal explanation. The audit CSV is `data/processed/validation/missed_cases_20261019/positive_case_audit.csv`.

Resume the complete acquisition/feature/comparison workflow locally:

```bash
.venv/bin/python -m src.validation.recent_study_run
```

Do not run a second copy while acquisition is active. Network access is required for uncached public SEC files; AWS is not required. Checkpoints permit resumption. Source or feature errors stop model fitting; they are never silently converted to negative labels. Completed results are saved under `data/processed/validation/recent_study_models_20261019/selection.json`. No model is automatically promoted and no final test is scored.

## Same-cohort model results

All numbers below use the same 167 resolved May 2023 development companies, explicit design weights and the same nested company-fold calibration. The median combines five fixed partitions, rather than choosing the best seed.

| Candidate | Median log loss (lower is better) | Median PR-AUC | Primary ROC-AUC | Primary recall at precision ≥50% |
| --- | ---: | ---: | ---: | ---: |
| frozen_probability | 0.07724 | 0.351 | 0.916 | 35.3% |
| frozen_detection | 0.07583 | 0.405 | 0.919 | 20.6% |
| native_company_balanced | 0.07701 | 0.415 | 0.914 | 26.5% |
| native_balanced | 0.07258 | 0.507 | 0.926 | 35.3% |
| xgb_native_sqrt | 0.07364 | 0.401 | 0.920 | 44.1% |

The selected `native_balanced` refit reduced median probability loss by 6.0% versus the frozen probability classifier and 4.3% versus the frozen detector. Primary-partition PR-AUC increased from 0.287 to 0.507 versus the frozen probability classifier; median PR-AUC increased from 0.351 to 0.507. These are same-cohort development comparisons, not final-test accuracy.

The selected refit failed the combined 50% precision/recall requirement in all five partitions. At precision ≥50%, its maximum recall ranged from 32.4% to 47.1%; at recall ≥50%, maximum precision ranged from 30.6% to 47.4%. Two other candidates passed the floor on just one partition each, showing why a single favorable split is insufficient. No threshold was released. No production model was promoted.

The saved research artifact is `data/processed/validation/recent_study_models_20261019/research_candidate.joblib`. It uses the same 63 ordered features and SEC-disclosure target. The full calibration mapper uses only evidence mature by December 17, 2024, before the reserved final forecast. Model selection reused this development cohort; an independent, reviewed final evaluation remains necessary.

Validation: 265 tests passed; four newly added tests protect chronology, sampling-union probabilities, source-cache integrity and reserved-company separation. One existing neural-network toy test still emits its known convergence warning. All new model outputs, source tables, implementation/config files and required dependencies have saved hashes. Four spread-out reused cache records also reproduced fresh extraction within 1e-12.
