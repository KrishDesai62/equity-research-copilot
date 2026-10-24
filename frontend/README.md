# Ledger frontend

A responsive filing-first research workspace with real public SEC data. No frontend build step, npm installation, external fonts, or API keys are required.

## Preview

From the repository root:

```bash
python3 -m http.server 8000 --directory frontend
```

Open **http://localhost:8000**. Serve the folder over HTTP; opening the HTML directly from disk will not load the JSON files.

## Host the standalone site

Publish the **contents of `frontend/`** to a static host. If the host asks for settings:

- Build command: leave empty.
- Publish/output directory: `frontend`.
- Entry point: `index.html`.

All routes use URL hashes, so no SPA rewrite rule is necessary. The checked-in `data/` directory is required: it contains the public snapshot of 46 companies, 138 filings, and 5,132 attributed passages. Files load one company at a time. No model artifacts, private reviewer records, environment variables, or credentials are included. The catalog, individual company payloads, CSS and JavaScript use relative URLs and work beneath a project subpath.

The static question box searches keywords in cached filing passages, with company and date filters. It returns excerpts with source links. It does not claim to run the Python RAG pipeline or generate an investment recommendation. The About the data screen explains the distinction.

## Connect the existing Python retrieval pipeline

From the repository root, using the existing project environment:

```bash
.venv/bin/python -m src.web.server --host 0.0.0.0 --port 8000
```

The server hosts the same frontend and enables `POST /api/answer` through `src.agents.qa_agent.answer_question`. It serves `/config.json` automatically with the connected mode enabled. The repository's saved filing index and Python dependencies must be available to this server; deploy them along with the application if choosing this option. This uses the existing extractive retrieval path, not a new language-model integration. A static host alone cannot run Python.

`frontend/config.json` defaults to `{"apiBase":null}` for standalone hosting. Prefer the included same-origin Python server for connected mode. A separate API origin would need its own hosting and CORS configuration.

## Refresh the public snapshot

```bash
.venv/bin/python scripts/export_frontend.py
```

The exporter reads `data/processed/research` and writes only public company identifiers, filing metadata, selected financial fields, and attributed passages. Dates in the exporter reflect the current frozen October 2026 research snapshot; change them deliberately when publishing a refreshed snapshot. Re-export after updating the underlying research data.

## Included interactions

- Search and switch between all 46 covered companies; `/` opens company search.
- Explore reported financials and revenue/cash-flow history with original filing links.
- Search passages by keyword, section, form and evidence cutoff.
- Open full passages and their original SEC reports.
- Follow companies, save per-company notes in browser storage, and export briefs/notes as text.
- Responsive desktop/mobile navigation and keyboard-accessible controls.

Browser storage is local to the current device and origin; it does not synchronize notes or watchlists. The interface displays no unreleased bankruptcy probability. Research snapshot dates are visible; hosting the site does not make the data live.

## Import a company by ticker

Run the Python server, open company search, and enter a ticker outside the saved collection, such as `TSLA` or `NFLX`. Choose **Import … from SEC** or press Enter. The dialog shows progress; it opens the company when indexing finishes. Existing companies open directly without another download. An import can continue while the user browses another company, and the current browser tab resumes polling after a page reload.

The importer resolves the current ticker through the SEC company directory and verifies the submissions and XBRL issuer CIK. It imports up to two annual, two quarterly, and two current reports, including supported amendments, and extracts financial history from the SEC company-facts API. Where comparable USD financials are unavailable, the company can still be researched through its filings; financial values stay unavailable and a coverage message explains the gap. An unknown or historical ticker absent from the current SEC directory returns an explicit failure.

Imports use the existing filing section extraction, chunking, financial-history extraction and SQLite FTS5 retrieval. They do not run embedding generation, change the ChromaDB collection, train a model, modify frozen research datasets, or publish bankruptcy scores. SEC sources and byte hashes are recorded. The selected reports' primary bodies are indexed; exhibits and exhaustive filing-history coverage are not promised.

The host needs outbound SEC access, project dependencies, writable storage, and `EDGAR_USER_AGENT` configured with an application name and real contact email. Keep that contact configuration on the server, out of frontend files. No paid API key is required. Requests use the existing shared SEC rate coordinator; a single worker processes a bounded import queue. Failed imports do not replace the published company. Cached raw files can be reused on retry.

For persistent hosting, mount a writable data volume and start the server with:

```bash
.venv/bin/python -m src.web.server --host 0.0.0.0 --port 8000 --import-root /path/to/persistent/company_imports
```

Imported payloads, independent indexes, source manifests and job states live in that directory. The catalog automatically combines these companies with the initial 46. Completed imports survive server restarts when the volume is retained; in-progress jobs report interruption and can be retried. There is no automatic background refresh of already imported companies.

Verified locally with live Tesla and Netflix SEC imports: six reports each, 135 and 108 indexed passages respectively, with financial histories and cited retrieval. Mocked tests cover invalid/unknown tickers, amended forms, date filters, duplicate jobs, failed-publication isolation and unchanged base data.

## Research agents and ML integration

The connected interface now includes **Research memo**, **Bankruptcy model**, and **Filing changes** company tabs. Bull and bear agents retrieve dated, issuer-checked evidence from either the shipped public corpus or an imported-company index. The synthesis layer assembles an extractive memo with deduplicated citations; it does not generate a validated buy/sell recommendation. Filing comparisons use the latest two comparable original 10-K or 10-Q reports available by the selected date.

**Refresh SEC data** creates a separate import version using fresh source-cache paths. An unsuccessful refresh leaves the previous published company and its citations intact. This is an explicit manual refresh; AWS scheduling is separate.

Run installation checks from the repository root:

```bash
.venv/bin/python -m src.web.preflight
```

Public mode prepares and displays all 63 model inputs, observed/missing counts, financial-source dates and model status. It withholds scores until the configured artifact has a validated release. This is the default:

```bash
.venv/bin/python -m src.web.server --host 127.0.0.1 --port 8000
```

For **local research only**, explicitly enable unvalidated scores:

```bash
.venv/bin/python -m src.web.server --host 127.0.0.1 --port 8000 --research-scores
```

Research scores are labeled experimental, have no high/low-risk category, and do not imply production approval. The final evaluation dates are blocked from interactive inference. Input gating rejects stale statements, excessive missingness, wrong/future source data, changed source bytes, incorrect feature order and mismatched model artifacts. Missing dated industry classifications remain unknown; current SIC codes are not silently used as historical facts.

The pinned model and feature-kernel hashes are configured in `config/serving_model.json`. The exact artifact is a trusted local file, not a user-uploaded model. Because model artifacts and raw research files are Git-ignored, they must be provisioned separately on a host. Original SEC sources are required for model-feature extraction and filing comparisons of the initial 46 companies; imported companies retain their own sources under `--import-root`. The shipped public passage corpus can bootstrap its own local retrieval index without the original research SQLite database.

For the local backend environment, install the repository’s existing `requirements-local.lock` and `requirements-filing-text.lock`. AWS deployment, authentication, TLS and cloud scheduling are not configured by this integration.

Endpoints (same origin):

- `GET /api/model`: verified artifact/release status, target and research mode.
- `POST /api/research`: `{ "ticker": "NFLX", "as_of": "2026-10-23" }`; cited growth/risk memo and model status.
- `POST /api/risk`: same body; ordered feature coverage and gated score.
- `POST /api/changes`: same body; original-report text changes and source references.
- `POST /api/imports`: `{ "ticker": "NFLX", "refresh": true }`; versioned manual refresh.

Current frozen research-model validation remains incomplete. Backend integration is not a new independent evaluation or a promotion of the model. No AWS deployment is needed to run these features locally.
