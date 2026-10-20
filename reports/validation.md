# Local validation results

**Current US-model status:** earlier US results are unverified diagnostics.
See [the current readiness audit](model_readiness.md) for source reconciliation
and reviewed-label availability. The UCI benchmark is a separate dataset and
does not establish US equity-research performance.

**Readiness: not production validated.**

The extraction and model-training paths executed successfully. The US pilot has provisional negative labels and too few independent events to establish prediction quality.

## Real-data classifier diagnostic

Source: [UCI Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data), Tomczak (2016), DOI 10.24432/C5F600, CC BY 4.0.

Used `5year.arff`, whose forecasting case predicts bankruptcy after one year. No horizons were mixed. Exact feature duplicates were removed before splitting. Fixed stratified 60/20/20 training/validation/test partitions were used; preprocessing was fit on training rows only.

| Model | Validation PR-AUC |
| --- | ---: |
| logistic | 0.2774 |
| xgboost | 0.6229 |
| neural_network | 0.3178 |

Selected on validation: **xgboost**. Held-out test PR-AUC: **0.6563**, compared with a constant-score baseline of **0.0699**.

Test observations: 1158; positive outcomes: 81.

The benchmark has no company IDs or filing timestamps. Company separation and chronological validity cannot be established. These results are diagnostic and do not demonstrate US-market generalization or calibrated bankruptcy probabilities.

## SEC extraction pilot

Downloaded Company Facts for 10 purposively selected companies, including historical bankrupt entities. Extracted **142 original annual filings**; **120 rows** remain after post-bankruptcy and incomplete-horizon exclusions.

| Feature | Missing fraction |
| --- | ---: |
| wc_to_ta | 3.3% |
| re_to_ta | 1.7% |
| ebit_to_ta | 20.0% |
| sales_to_ta | 6.7% |

Facts are selected by original accession, public filing date, fiscal end, USD unit, and annual duration. Cell-level source tags and values are retained in `fact_provenance.csv`.

Used four accounting ratios. Historical market capitalization was not sourced, so the market-equity ratio is excluded. This is not a complete Altman Z-score. Operating income is a documented EBIT proxy.

## Original-filing reconciliation and disclosure checks

Independently parsed the original non-dimensional inline-XBRL financial values in four SEC filings:

| Company | Original filing checks | Result |
| --- | ---: | --- |
| apple | 14 | All matched |
| bbby | 14 | All matched |

Apple 2023/2024 and Bed Bath & Beyond 2021/2022 disclosure comparisons both extracted Risk Factors and MD&A. Each produced two changed sections and zero unavailable sections. Reports retain original excerpts and SEC URLs; this verifies text extraction and differences, not semantic financial materiality.

## US labels and evaluation

The annual pilot contains **4 positive rows across 4 companies**. Five bankruptcy dates have primary SEC citations in the pilot configuration. A cited positive event does not prove complete historical event coverage for negative examples.

Negative pilot labels are explicitly provisional. Production training rejects those rows. The existing uncited/secondary-source bankruptcy CSV is not silently treated as a complete registry.

Company cohorts are held out across all their history; yearly test windows follow at least five initial training years. Training labels are purged until their availability dates. This fixes the earlier evaluator that erased the training set when companies recurred each quarter.

The exploratory US run is saved for pipeline inspection only:

| Model | Mixed-class test folds | Out-of-sample rows | Positive out-of-sample rows |
| --- | ---: | ---: | ---: |
| logistic | 2 | 18 | 2 |
| xgboost | 2 | 18 | 2 |
| neural_network | 2 | 18 | 2 |

These counts are too small for a trustworthy model ranking. Any exploratory PR-AUC in the JSON must not be represented as production validation. Some neural-network folds reached their iteration limit; warnings are retained in the report.

No diagnostic model was promoted into the risk agent's production model registry. Benchmark and smoke-test artifacts remain under `data/processed/validation/`.

## Remaining gates

- Review complete bankruptcy-event coverage before using any pilot negative labels in production.
- Expand a documented historical universe with more independent bankruptcy events and untouched final holdout companies.
- Source point-in-time market capitalization before using the full five-component Altman feature set.
- Quarterly/TTM features and independent final US holdout validation remain outstanding.

## Reproduce

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.validation.run
```

The SEC contact is configured in the gitignored `.env`; it is not included in this report. The second command reuses cached data. A fresh run requires network access. Detailed outputs and source checksums are in `data/processed/validation/`.
