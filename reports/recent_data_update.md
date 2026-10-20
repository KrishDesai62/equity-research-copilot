# Recent data refresh before AWS integration

**October 17 update:** historical negatives are now reviewed within 2010–2022
BRD scope, the date conflict is adjudicated, and the final cohort is reserved.
See [evaluation preparation](evaluation_preparation.md) for the current model
testing status. Recent 2023–2026 negative coverage remains unknown.

The local updater now uses a dated research configuration with an observation
endpoint of **2026-10-16**, adds three cited 2024–2025 bankruptcy events, and
imports the SEC's **2026 Q3 Financial Statement Data Set**. This financial
dataset supports the whole research project; it does not supply bankruptcy
outcome labels.

```bash
.venv/bin/python -m src.validation.update_recent
```

The configuration is `config/research_recent.json`. Existing historical
experiment configurations, splits, scores and model artifacts are preserved.
Before replacing the current research table, the updater copies its contents,
including human review work, into `data/processed/research_snapshots/`. Fresh
company histories and submission metadata are cached under
`data/raw/research_updates/2026-10-16/`, leaving earlier raw model inputs intact.
Repeated runs reuse this dated cache. Choose a new observation endpoint and
therefore a new snapshot cache when refreshing on a later date; merely rerunning
the same dated configuration is not a real-time feed.

## What is added

- SEC Company Facts and recent/archived submission histories for the cohort,
  filtered by public filing date before calculating historical features.
- Big Lots (2024-09-09), Spirit Airlines (2024-11-18) and Nikola (2025-02-19),
  with issuer SEC documents as event citations. These are additional documented
  positives, not proof of complete population-wide bankruptcy coverage.
- The most recently completed quarterly SEC Financial Statement dataset for the
  specified endpoint: 2026 Q3 for October 16, 2026. An unavailable quarterly
  download raises rather than silently substituting an older dataset.
- A memory-bounded numeric-fact reader that filters the research companies,
  excludes custom-taxonomy and segmented facts from consolidated US-GAAP
  comparisons, and withholds conflicting values instead of choosing one.
- Rebuilt filing chunks, local search index, cited Q&A checks and model-readiness
  checks using the refreshed dataset's own observation date.

Source datasets: [SEC Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets.html),
[Big Lots issuer filing](https://www.sec.gov/Archives/edgar/data/768835/000076883524000062/big-20240905.htm),
[Spirit issuer filing](https://www.sec.gov/Archives/edgar/data/1498710/000095010324016405/dp220813_8k.htm),
[Nikola issuer filing](https://www.sec.gov/Archives/edgar/data/1731289/000173128925000004/nkla-20250218.htm).

## Outputs

Current results remain under `data/processed/research/`. The new source exports
are `bulk_facts.csv`, `bulk_financial_table.csv`, `bulk_conflicts.csv` and
`bulk_dataset.json`. `recent_update.json` records the source checksums, cohort
counts, corroboration of the three new event dates, Q&A checks and model status.
Bulk financial values are an additional source, not silently merged into an
incompatible feature schema or treated as extra independent training examples.

Recent financial rows are not automatically negative outcomes. A filing made in
2026 may require follow-up into 2027 for a one-year target. Unknown or immature
outcomes remain unlabeled. The LoPucki release still ends in 2022, and the Toys R
Us date conflict remains quarantined. Current US accuracy remains unavailable
until adequate reviewed examples of both classes support valid evaluation.

## AWS order agreed with the user

1. Refresh and validate data locally.
2. Complete outcome-coverage review, compare logistic regression/XGBoost/NN on
   company/time-separated development folds, and evaluate a genuinely unseen
   final cohort.
3. Configure the AWS profile, region and private S3 bucket; run a live monitor
   test and then enable the schedule.

The existing AWS layer supports S3 caching and scheduled discovery of new SEC
filings, filing comparisons and durable checkpoints. It is **not deployed or
enabled**. Its default is a daily schedule, not an instantaneous streaming feed.
Refreshing filings would not automatically retrain or validate a model. This
updater runs locally with S3 caching explicitly disabled for the new quarterly
download, and does not create cloud resources or enable schedules.
