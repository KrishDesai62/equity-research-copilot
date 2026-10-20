# Run the research workflow locally

No AWS account, bucket, access keys or model API key is needed. Run commands from
the repository root using the existing `.venv`. For a new environment, use
Python 3.14 and install `requirements-local.lock` into a virtual environment.
Downloaded/curated evidence and frozen research artifacts must also be present;
they are not all distributed in Git.

## Cached checks and saved model evidence

```bash
.venv/bin/python -m src.validation.local_workflow
```

This reads the local research snapshot, audits feature provenance/freshness,
verifies the candidate against saved independent/calibration evidence, and writes
a run report and log under `data/processed/validation/local_runs/`.
It does not download public sources, fit a model or predict on exposed tests.
When V3 evidence is present, it also verifies and includes the saved design-weighted
assessment. For the dataset specification and resumable acquisition commands, see
[the V3 data card](cohort_v3_data_card.md). For the actual completed evaluation,
see [the V3 results](cohort_v3_results.md).
The date defaults to today's date in America/Chicago; use `--as-of YYYY-MM-DD`
for an explicit assessment date.

## Cited filing questions

```bash
.venv/bin/python -m src.validation.local_workflow --ticker AAPL --question "What does the filing say about liquidity?"
```

This uses the existing local SQLite index and returns source passages with
citations, filtered to the requested date. It does not synthesize an investment
recommendation or validate numerical reasoning.

## Refresh public SEC data

Ensure `.env` contains a real `EDGAR_USER_AGENT` with an application name and
contact email. The current workspace already has it configured. Then run:

```bash
.venv/bin/python -m src.validation.local_workflow --refresh
```

This requires internet access to public sources, but no AWS credentials. It uses
`config/research_recent.json`, advances the observation endpoint to the run date,
and writes a separate effective configuration. The user's source configuration
is preserved. A completed, checked dataset is published atomically as the current
local snapshot; a failed refresh leaves the active pointer intact.

For a cached historical refresh use, for example:

```bash
.venv/bin/python -m src.validation.local_workflow --refresh --as-of 2026-10-16
```

Each newly published version preserves the exact effective configuration as
`research_config.json`, so later standalone label audits can identify its scope
by checksum. A completed calendar quarter may not yet have a published SEC bulk
archive; that refresh fails explicitly instead of silently claiming newer data.

## Existing development fitting

For the completed larger research experiment, use:

```bash
.venv/bin/python -m src.validation.local_workflow --train-v2
```

This verifies/reuses the separately frozen V2 candidate. The default local report
also includes its failed research assessment. See
[the experiment and its limitations](model_improvement_v2.md). It does not rerun
final tests or replace the serving registry.

```bash
.venv/bin/python -m src.validation.local_workflow --train-development
```

This delegates to the existing development calibration workflow. In this
workspace, its research candidate is already frozen, so the workflow verifies
and reuses it. It does not silently retune that model, use the final-test labels
for fitting or publish a serving artifact. Fitting from scratch still requires
the frozen development inputs and additional reviewed development-case data.

The old `src.validation.finish_local` entry point now delegates here;
`finish_local --download` requests a local refresh. It no longer rebuilds an
active snapshot in place or describes an exposed final cohort as untouched.

## Meaning of completion

Exit 0 means the requested research commands completed, not that the model
passed release gates. Missing local data returns exit 2 with file paths and a
refresh instruction. Execution failures return a nonzero exit and retain the
run's failure report. Production-readiness checks remain in the report even
when release is blocked.

`EQUITY_LOCAL_ONLY=1` disables automatic S3 cache initialization before inspecting
bucket settings. The workflow sets it temporarily and restores the previous
environment afterward. It does not alter `.env` or configure cloud resources.

This is a one-shot command, not a continuously running monitoring service.
Recent filings do not create reviewed recent non-bankruptcy outcomes. The current
candidate still needs population calibration evidence; the local runner does not
fix that statistical limitation. See [the calibration audit](population_calibration.md).

Earlier local checks covered a cached 2026-10-16 refresh, development-candidate
reuse, AAPL Q&A with three citations, and the legacy entry point. The V3 work adds
source hashes, sampling weights, company/time separation and a one-use final test;
its results remain research only. The current suite has 171 passing tests. The
neural-network toy fixture emits a convergence warning; the V3 neural candidate
also did not converge within its frozen budget and was not selected. No AWS
resources were used.
