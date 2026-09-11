# AI Equity Research Copilot

Guided project skeleton. Every `src/` file is a stub with docstrings and
`# TODO` markers — you implement the logic, this just gives you the shape
so you don't spend the first three sessions on file layout and imports.

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
