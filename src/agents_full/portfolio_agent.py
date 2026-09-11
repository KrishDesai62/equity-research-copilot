"""
Phase 5 (post-MVP). Runs on a quarterly cadence aligned to filing dates,
not daily. At each decision point: retrieve latest filing -> pull Track
A/B/C model outputs as tool calls -> buy/hold/sell judgment.

What this is supposed to test (don't lose sight of this when building
it): whether the agent's judgment adds value over the raw signal, not
just whether it makes money. Always run it alongside
backtest/mechanical_baseline.py and a plain buy-and-hold comparison —
see backtest/backtest_engine.py.
"""
# TODO: build in Phase 5, after Phase 4's agents exist to reuse.
