# AI Equity Research Copilot

**Coverage and simple baselines:** audited 198 additional current reports, reconciled a 48,002-report body-review backlog, and triaged 251 broader-search documents without finding a new qualifying in-horizon issuer assertion. Fixed logistic and Altman-style baselines did not beat the retained tree model. **288 tests pass.** The research pipeline and evidence are frozen; independent review, negative-label coverage and adequate untouched evaluation remain incomplete. See [the audit, benchmark results and reproduction commands](reports/label_coverage_and_baselines_20261019.md).

**Public outcome expansion:** acquired additional SEC evidence, expanding training to **3,912 eligible snapshots / 95 positive companies** and adding negative examples for 127 companies. Reserved new April 2025 companies: **1 positive, 83 disclosure negatives, 18 unknowns**, unscored and below the 20-positive minimum. Two fixed refits did not beat the reference, which was retained. **282 tests pass.** See [sources, dates, review packets and results](reports/outcome_expansion_20261019.md).

**Reliability follow-through:** audited the five false alerts, tested quarterly trends, and mined 176 hard-negative training snapshots from 115 companies using held-company training predictions. Both refits reduced probability loss but worsened alert precision, so the alert reference was retained. **278 tests pass; no production approval or final-test scoring.** See [the completed interventions and evidence limits](reports/reliability_followthrough_20261019.md).

**Quarterly-trend audit:** traced five false alerts to distressed disclosure-negative companies and added nine point-in-time quarterly trend/age features. The balanced refit reduced median development probability loss **2.2%**, but worsened alert precision, so the existing alert model was retained. Reliability remains unproven; no final scoring or deployment. See [the source audit, comparison and limits](reports/quarter_trends_20261019.md).

**Local reliability audit:** tested and rejected a bootstrap threshold policy because it worsened performance and reduced coverage. Retained the existing model. Run `.venv/bin/python -m src.validation.local_reliability_status` for an offline integrity and reliability audit. **273 tests pass; predictive reliability remains unproven.** See [the experiment and remaining evidence requirements](reports/local_reliability_20261019.md).

**Alert-policy diagnostic:** fixed HGB/XGBoost blends did not beat the existing probability candidate. A separate held-company threshold check on raw scores achieved **65.5% precision / 55.9% recall in four of five reused-development partitions**, but precision fell to **26.5%** in the fifth. This is not independent production validation; no threshold was released. See [the comparison, threshold checks and limitations](reports/ensemble_and_policy_20261019.md).

**Newer public-data study completed:** expanded training to **3,005 eligible snapshots and 91 positive companies**, and evaluated five candidates on **167 resolved May 2023 development companies, including 34 positives**. On that same newer cohort, the selected refit reduced median probability loss **6.0%** versus the frozen probability model across five fixed partitions; primary PR-AUC improved **0.287 → 0.507**. The combined precision/recall floor still fails; the final remains unscored. See [sources, sampling, dates, comparison and limits](reports/recent_study_20261019.md).

**Probability calibration update:** nested company-fold calibration improved original-partition development ROC-AUC from **0.875 to 0.887** and reduced median probability loss by **6.1% across five fixed partitions**. The separate detection specialist remains available; production precision/recall requirements are still unmet. See [the calibration procedure, sensitivity results and limits](reports/calibration_refinement_20261019.md).

**Detection research update:** tested native missing-value models, alternative training weights and ensembles. A separate research specialist improved development PR-AUC from **0.154 to 0.226** and recall at precision ≥50% from **6.7% to 24.6%**. The existing probability model retains lower probability loss; neither meets the production precision/recall requirement. See [the comparison and saved candidate](reports/detection_improvement_20261019.md).

**Production follow-through:** re-audited the reserved outcomes, fixed production target/schema checks, and tested contextual distress language plus exchange/auditor/industry signals. Development ROC-AUC improved to **0.875** and PR-AUC to **0.154**, but the precision/recall release floor still fails. See [the changes, results and current gates](reports/production_followthrough_20261019.md). No final predictions or AWS deployment were performed.

**October 19 development update:** after a source audit of issuer identity, historical training has **2,174 eligible snapshots and 89 positive companies**. Corrected recent development has **29 positives and 274 disclosure negatives**, with much lower sampling-weight concentration. Added eight original filing-text features. On the same corrected development cohort, expanded training improved selected-model ROC-AUC from **0.771 to 0.863**; no model met both 50% precision and recall. See [the source audit and saved results](reports/development_update_20261019.md). The reserved final remains unscored; its evidence packet also needs the updated issuer audit before validation. No production model has been approved.

**October 18 update:** expanded the reserved final dataset to **20
source-corroborated bankruptcies and 315 disclosure negatives**. Built **1,282 usable
historical training snapshots**, added seven public filing-event features and
compared nine models. No model met both precision/recall floors; the final remains
unscored and no production model was promoted. Generated a 335-unit evidence packet
for independent review. See [actual data, sources, model results and remaining
gates](reports/sec_disclosure_expansion_20261018.md). Earlier
[V1 results](reports/sec_disclosure_v1.md) remain archived.

Production-support additions now include reviewed-outcome admission, future-cohort
reservation/evaluation, disclosure features, development tuning, dated strategy
backtests, an IAM operator-status template and approved-model rollback. Real recent
outcome/strategy validation and live AWS tests remain incomplete. See
[the implementation status](reports/production_work_status.md).

See [the dataset inventory](reports/dataset_inventory.md) for all added sources,
saved counts and label changes. Export the verified bankruptcy training and
calibration tables locally with `.venv/bin/python -m src.validation.export_dataset`.

**Earlier held-out experiment — V3:** acquired a fixed 915-company cohort from
48 historical SEC archives, reconciled 1,985 source files and built **542 eligible
prediction snapshots**. All five model variants were fitted. The frozen selected
logistic model scored ROC-AUC **0.794** on 75 unseen companies, with five bankruptcies.
It remains **research only**: no threshold met both precision/recall requirements,
the test lacks the required 20 independent bankruptcies, and calibration is not
validated for the current market. See [the results](reports/cohort_v3_results.md)
and [the dataset and protocol](reports/cohort_v3_data_card.md).

Read the verified saved result locally; this does not rescore the test:

```bash
.venv/bin/python -m src.validation.cohort_v3 summary
```

**Earlier V2 experiment:** added a 435-issuer historical panel and compared
five model variants with two calibration options. Probability loss improved over
the previous candidate on 39 new companies, but the constant baseline was better
and the sample had no bankruptcies. **No model was promoted.** See
[the complete improvement report](reports/model_improvement_v2.md).

Run the current research workflow **locally without AWS**:

```bash
.venv/bin/python -m src.validation.local_workflow
```

Add `--refresh` to fetch public SEC data, or `--ticker AAPL --question "What are
the liquidity risks?"` for cited local filing passages. See
[the local run guide](reports/run_locally.md). Saved final tests are never rerun
by this command; production model release remains blocked.

**Calibration audit:** a separate natural-sample historical audit found a mean
score of 26.2% but no reviewed one-year events among 51 eligible companies.
Scores are **not validated market probabilities**. See
[the local calibration audit](reports/population_calibration.md). AWS is not required.

**Earlier case-control test:** the frozen research candidate achieved **68.8%
precision, 76.8% recall and 80.6% accuracy** on 206 rows from 47 unseen companies.
This historical case-control benchmark is not representative of market prevalence.
See [the independent evaluation](reports/independent_candidate_evaluation.md).

**Production status: blocked.** Representative probability calibration and
recommendation validation remain outstanding. V3 passes the company-count floor,
but its five final positive companies fall short of the 20-event floor, and it
has no qualifying alert threshold. Candidates remain outside the serving registry.
An earlier candidate's failed final result is preserved as a separate experiment.
See [production readiness](reports/production_readiness.md) and
[candidate development](reports/development_calibration.md).

For the newest local dataset, run:

```bash
.venv/bin/python -m src.validation.update_recent
```

This uses the 2026-10-16 research configuration and adds SEC 2026 Q3 financial
data plus cited 2024–2025 bankruptcy cases. See [the recent-data guide](reports/recent_data_update.md).
AWS activation remains deferred until model validation, as requested.

The legacy local command delegates to the same local-only workflow:

```bash
.venv/bin/python -m src.validation.finish_local
```

See [local completion and remaining evidence](reports/local_completion.md).
The actual LoPucki release ends in 2022 and does not automatically supply healthy
labels. Local cited Q&A works without an API key or embedding download. AWS
configuration and live deployment are deferred.

**Current US model status:** historical source review now supplies 1,630 negative
rows within the BRD's 2010–2022 large-public-company scope. Toys R Us uses the
adjudicated court filing date. The separate company/time-disjoint final cohort
has now been scored and is no longer untouched. See [the evaluation preparation](reports/evaluation_preparation.md).
Earlier accuracy figures remain superseded diagnostics.

```bash
.venv/bin/python -m src.validation.model_readiness --evaluate
```

This validates the coverage worksheet and compares all three model types only
when reviewed examples of both classes exist. It does not rescore the exposed
holdout or publish a production artifact.

The connected research dataset now combines SEC financial history, cited bankruptcy
outcomes, filing sections and changes, accounting-event candidates, sampled insider
transactions, and a separate FinQA development benchmark. See
[the dataset guide](reports/research_dataset.md) for output schemas and label limits.

```bash
.venv/bin/python -m src.data_layer.research_dataset
```

Research prototype with local model training, filing retrieval and optional S3 caching.
Some later-phase agents remain stubs; real-data MVP validation is still required.

## What changed from the original spec (read this first)

Four correctness fixes and two differentiation additions, folded into the
skeleton below so they're not easy to skip:

1. **Point-in-time universe, not today's constituent list.**
   `src/data_layer/point_in_time_universe.py` builds a ticker/CIK universe
   *as it existed at each historical date*, including companies later
   delisted or bankrupted. If you train Track A on "today's S&P 500," you
   have quietly excluded every company that ever went bankrupt — the model
   never sees a positive label. This file exists specifically to stop that.

2. **Filing-date lag, not fiscal-period-end date.**
   `src/data_layer/xbrl_bulk.py` and `src/ml/features_distress.py` both key
   off `filing_date` (when the 10-Q/10-K was actually made public), not
   `fiscal_period_end`. Joining on period-end date lets the model "see" Q3
   numbers before they existed — the single most common source of a
   suspiciously good backtest.

3. **Severity-split manipulation labels.**
   `src/data_layer/restatement_labels.py` keeps AAER enforcement actions and
   8-K Item 4.02 restatements in separate columns instead of one blended
   "manipulation" label. A typo-driven restatement and a fraud enforcement
   action are not the same event; don't let the model treat them as one.

4. **Insider trading + short interest as features.**
   `src/data_layer/insider_trading.py` pulls Form 4 filings (free, EDGAR)
   as an added feature source for both tracks. Cheap to add, and most
   RAG-over-filings tools on the market don't bother with it — it's one of
   your few genuinely differentiated inputs.

5. **Narrowed small-cap wedge.**
   Don't pitch "small/micro-cap coverage" broadly — pick one niche
   (`config/settings.yaml` has a `niche` field to fill in: e.g. OTC/pink
   sheet filers, or a single sector). Broad is what the incumbents already
   claim; narrow is defensible.

6. **MCP server wraps the retrieval backbone.**
   `src/mcp_server/retrieval_mcp_server.py` exposes `retrieve()` as an MCP
   tool once Phase 1 is stable. Build this *after* retrieval works, not
   before — it's a thin wrapper, not new logic.

## MVP — build this first, nothing else

The full spec is six phases. The MVP is a deliberately small slice that
proves the two riskiest assumptions before you invest in agents or a
backtest:

- **Riskiest assumption #1:** the point-in-time data pipeline is buildable
  without leakage (survivorship bias, filing-date lag).
- **Riskiest assumption #2:** the distress classifier beats a naive
  baseline on ~20-30 real S&P 500 names using free data end to end.

**MVP scope:**
- `data_layer`: EDGAR ingestion + XBRL bulk parse + point-in-time universe
  for ~20-30 S&P 500 tickers only. Skip small-cap universe for now.
- `rag`: chunking + embeddings + Chroma store + the single `retrieve()`
  function. No multi-agent debate yet — one plain Q&A function that calls
  `retrieve()` and answers with citations is enough to prove the backbone.
- `ml`: Track A (distress) only. Logistic regression baseline vs. XGBoost.
  Skip Track B (manipulation) and Track C (return signal) for the MVP —
  they reuse the same XBRL pipeline once it exists, so they're cheap to
  add after, not before.
- `backtest` / `agents`: **not in MVP.** Stubbed in `agents_full/` and
  `backtest/` so the shape exists, but don't build these until Track A is
  validated against the Polish/Taiwan bankruptcy benchmark datasets.

**MVP "done" criteria** (don't move to Phase 2 until all four are true):
- [ ] Point-in-time universe file produces a company list that includes at
      least a few historically delisted/bankrupt names, not just current
      constituents.
- [ ] `retrieve()` returns filing-section-level chunks with correct
      metadata filtering (ticker, filing_type, date_range).
- [ ] Track A model trained with walk-forward, split-by-company validation
      reports AUC-PR (not accuracy) and beats a "predict majority class"
      baseline.
- [ ] You can explain, in one sentence each, why filing-date lag and
      point-in-time universe matter — if you can't, re-read files 1-2
      above before continuing.

## Build order (post-MVP)

| Phase | What | File(s) |
|---|---|---|
| 1 (MVP) | RAG backbone + Track A on ~20-30 S&P names | `data_layer/`, `rag/`, `ml/features_distress.py`, `ml/train_distress.py` |
| 2 | Track B (manipulation) + Track C (return signal, exploratory only) | `ml/features_manipulation.py`, `ml/train_manipulation.py` |
| 3 | NN vs. XGBoost comparison on both tracks | `ml/train_distress.py`, `ml/train_manipulation.py` (extend) |
| 4 | Multi-agent research debate, 3-4 demo names | `agents_full/` |
| 5 | Quarterly portfolio agent + backtest vs. mechanical baseline & buy-and-hold | `backtest/` |
| 6 | Expand to narrowed small-cap niche, end to end | reconfigure `config/settings.yaml` universe toggle |
| — | MCP server (do any time after Phase 1 is stable) | `mcp_server/retrieval_mcp_server.py` |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in EDGAR user-agent string, any API keys
```

SEC EDGAR requires a descriptive `User-Agent` header on every request
(name + email) — see `.env.example`. It rate-limits at ~10 req/sec; the
ingestion stub in `data_layer/edgar_ingest.py` has a TODO for a rate
limiter, don't skip it.

## Directory map

```
config/               ticker universes, settings (niche, paths, rate limits)
data/raw/             untouched pulled filings / XBRL zips (gitignored)
data/processed/       cleaned tabular features (gitignored)
src/data_layer/       ingestion + label construction (the fixes live here)
src/rag/              chunking, embeddings, vector store, retrieve()
src/ml/               feature engineering + training + evaluation, Tracks A/B/C
src/agents/           MVP: single Q&A agent over retrieve()
src/agents_full/      post-MVP: bull/bear/risk/synthesis + portfolio agent
src/backtest/         post-MVP: quarterly backtest engine + mechanical baseline
src/mcp_server/       post-Phase-1: MCP wrapper around retrieve()
tests/                correctness tests for the leakage fixes — run these first
```


## Local neural network and risk inference

The shared neural network is a scikit-learn MLP with two hidden layers,
median imputation, scaling, and balanced sample weights. Training stays local.
Both tracks expose neural-network trainers alongside logistic regression and XGBoost.
A labeled CSV must contain your ordered numeric features, binary target, `cik`,
`filing_date`, and a label availability date. For negative outcomes, availability
must be the end of the fully observed prediction horizon; do not label censored
rows as negative. For positive outcomes it is when the event became public.

```bash
.venv/bin/python -m src.ml.train data/processed/features/distress.csv \
  --track distress --target target_distressed_1yr \
  --features wc_to_ta re_to_ta ebit_to_ta mve_to_tl sales_to_ta \
  --label-available-col label_available_date --cutoff 2025-12-31 \
  --model neural_network --compare
.venv/bin/python -m pytest -q
```

Comparison uses chronological test windows crossed with fixed company holdout
cohorts. Every held-out company is excluded from training across its full history;
training labels are purged until their availability dates. Recurring company panels
are supported. Use `--period Y --min-train-periods 5` for yearly test windows after
five initial years, and optionally `--test-companies` for a fixed holdout cohort.
No usable evaluation raises an error instead of silently returning an empty report.

`RiskAgentTool` reads `data/processed/features/risk.csv` (or an injected DataFrame)
and trusted `distress.joblib` / `manipulation.joblib` artifacts. It selects the
latest public feature row and rejects models trained after the requested date.
Absent models produce unavailable fields, never invented probabilities. Balanced
classifier scores are uncalibrated; calibrate on independent historical data
before interpreting them as bankruptcy frequencies.

## Optional AWS filing cache

Set a real `EDGAR_USER_AGENT` and `AWS_S3_BUCKET` in `.env`. Boto3 uses its standard
credential chain (including an AWS profile or role). Never commit credentials.
With no bucket configured, ingestion uses only the local cache. With a bucket,
filings and quarterly ZIPs check local disk, then S3, then SEC; new downloads
are uploaded encrypted. S3 access errors surface rather than masquerading as misses.
Existing Chroma collections must be rebuilt to add numeric filing-date metadata.

`infra/aws/cache.yaml` provisions a private, encrypted, versioned bucket with a
TLS-only policy and retains it if the stack is deleted. Nothing is deployed automatically.
To deploy after configuring your AWS CLI:

```bash
aws cloudformation deploy --template-file infra/aws/cache.yaml --stack-name equity-research-cache
aws cloudformation describe-stacks --stack-name equity-research-cache --query 'Stacks[0].Outputs'
```

Set `AWS_S3_BUCKET` to the returned bucket name. Your local principal needs
`s3:GetObject` and `s3:PutObject` on that bucket's `equity-research/*` prefix.
AWS only stores raw source data; no training services or cloud compute are used.

Remaining MVP work includes curated and fully observed outcome labels, validated
historical membership intervals (the Wikipedia importer alone is insufficient),
real SEC/RAG integration checks, and the Q&A agent, which is still a stub.

On macOS, XGBoost additionally requires the OpenMP runtime (`brew install libomp`).
Logistic and neural-network training do not load XGBoost.

Implementation references: [Boto3 S3 operations](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/put_object.html) and [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

## Filing change detection

**Automatic daily monitoring:** the Lambda handler in `src/aws/filing_monitor.py`
now checks an S3 watchlist, processes new filings, saves reports, and checkpoints
processed accession numbers. Deploy it with `infra/aws/monitor.yaml`; see
[the deployment guide](infra/aws/monitor.md). The schedule is disabled until you
configure and enable it. Example watchlist: `config/monitor_watchlist.example.json`.

Compare a company's two latest **10-Ks** or **10-Qs** to see what changed in
Risk Factors and Management's Discussion and Analysis (MD&A). This uses text
comparison, so no LLM, embedding download, trained classifier, or API fee is
required. Existing local/S3 filing caching is reused.

After setting your real `EDGAR_USER_AGENT` in `.env`, run:

```bash
.venv/bin/python -m src.data_layer.filing_changes \
  --cik 0000320193 --form 10-K \
  --output data/processed/filing_changes/apple_annual
```

This writes `apple_annual.md` (readable report) and `apple_annual.json` (structured
data for an agent or future alerts). Each changed block contains previous and
current excerpts, source links, and sentence ranges within the normalized section.
The JSON includes accession numbers, filing dates, input text hashes, a unified
diff, and a section-level text similarity measure. Similarity is not a risk or
materiality score. Running the same comparison again overwrites the chosen output
files; choose a distinct output name to retain each historical report.

Use `--form 10-Q` for sequential quarterly comparisons or `--as-of 2025-03-01`
to exclude later filings. Automatic selection uses the SEC's **recent submissions
list**, not its archived history; an older cutoff may require explicit local files.
Amendments are excluded to avoid comparing a partial amendment with a full filing.
Quarterly Risk Factors may describe only updates to the annual disclosure, so
text differences between quarters do not represent a complete risk inventory.

You can also compare files already on disk without AWS or SEC access:

```bash
.venv/bin/python -m src.data_layer.filing_changes --cik 0000320193 --form 10-K \
  --previous-file path/to/older.html --current-file path/to/newer.html \
  --previous-date YYYY-MM-DD --current-date YYYY-MM-DD \
  --previous-accession OLDER-ACCESSION --current-accession NEWER-ACCESSION \
  --previous-document older-document.htm --current-document newer-document.htm \
  --output data/processed/filing_changes/comparison
```

Replace the metadata placeholders with the actual filings' values (SEC accession
format: `0000320193-25-000079`). File paths can point to the cached `.txt` files;
they contain the original filing HTML. Metadata is required so citations identify
the actual documents.

Python callers can use `detect_latest_changes(cik, filing_type="10-K", as_of=None)`
or `compare_filings(previous, current, previous_text, current_text)` from
`src.data_layer.filing_changes`, constructing `Filing` metadata objects for the latter.

Formatting whitespace, common smart quotes, scripts, and hidden inline-XBRL
content are normalized away. The parser prefers substantive sections over short
table-of-contents entries and requires a subsequent Item heading as an end boundary.
Unrecognized/missing sections are marked **unavailable**, not unchanged or deleted.
Unusual SEC HTML layouts can still require manual review. Changes are exact textual
review signals, not a semantic assessment or an investment recommendation.

```bash
.venv/bin/python -m pytest tests/test_filing_changes.py -q
```


## Local real-data validation

The expanded improvement experiment is documented in
[reports/improvement_experiment.md](reports/improvement_experiment.md). It uses
36 companies, quarterly/annual as-filed features, trailing-year cash flows,
company/time-separated development folds, a committed company holdout, and a
small prespecified model search. The run did not demonstrate a reliable
prediction improvement; negative outcome coverage remains provisional.

```bash
.venv/bin/python -m src.validation.improve
.venv/bin/python -m src.validation.improve --audit-data
```

The first command reuses a completed experiment rather than retuning against an
exposed test set. The second checks source values and trailing-flow arithmetic
without recomputing holdout scores. Detailed outputs are under
`data/processed/validation/expanded/`; corrected-data audits and the pending
outcome-coverage worksheet are under `expanded_quality_check/`. A new hypothesis
needs genuinely unseen test companies, not just a renamed output directory.

Read [the validation results](reports/validation.md) before treating any model
score as validated. Reproduce the downloaded-data diagnostics with:

```bash
.venv/bin/python -m src.validation.run
```

The runner caches the UCI Polish Companies Bankruptcy benchmark, prepares an
annual ten-company SEC extraction pilot with cell-level provenance, independently
reconciles sample values against original inline-XBRL filings, and writes reports
under `data/processed/validation/`. SEC requests require your real contact header
in the gitignored `.env`; a fresh run needs network access.

The UCI run is a classifier diagnostic using a fixed stratified train/validation/test
split. It lacks company IDs and filing dates and does not validate US point-in-time
predictions. The SEC sample is deliberately small and selected for pipeline testing,
not a representative historical index universe. Its negative outcomes are marked
provisional; production training rejects unreviewed coverage. Diagnostic models are
stored outside the risk agent's production registry.

`build_distress_labels` now requires public `filing_date` anchors and an observation
endpoint. Negative labels additionally require `event_coverage_reviewed`,
`coverage_start`, and adequate follow-up through the entire horizon. Unknown or
censored targets remain missing. Calendar-year horizons replace approximate day
counts. No missing bankruptcy record is silently assumed to be a healthy company.
