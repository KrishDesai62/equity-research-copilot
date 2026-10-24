# Backend and ML integration — October 23, 2026

Implemented a local backend that connects both the original public corpus and on-demand SEC imports to bull/bear evidence collection, extractive research synthesis, filing comparisons, and the existing frozen bankruptcy-disclosure model. AWS resources and schedules are unchanged.

The model-input adapter uses the original as-filed financial-history extractor and the same financial, filing-item, text, contextual and market-event feature kernels as the retained 63-feature model. It chooses an exact original financial source filed strictly before the forecast, checks source hashes and issuer identity, and preserves missing inputs. Incomplete event-history coverage does not become zero event counts. Current SIC metadata is not backdated into historical industry features. Live checks for Apple, Microsoft, Tesla and Netflix produced 52 observed inputs out of 63: missing industry indicators and financial fields remain explicit.

Public mode assembles inputs but does not score an unapproved model. A local `--research-scores` option permits explicitly unvalidated diagnostic inference, with no released threshold, risk category or alert. Model/feature-kernel checksums, training chronology, source integrity, stale/missing input gates, class ordering, probability validity and protected final dates are checked. Default serving approval remains false. The operator must not enable experimental public predictions merely because the endpoint works.

The matched saved development benchmark reproduced without refitting: ROC-AUC 0.922569 for the tree model versus 0.798602 for the Altman operating-income approximation on the same 143 companies; tree PR-AUC 0.600454. This reproduces reused-development diagnostics, not independent final performance. The model artifact is unchanged. The final cohort remains unscored.

Manual refresh creates new source/version paths and atomically publishes only completed imports. Tests verify a failed refresh preserves the previous version. A host-local base index can be rebuilt from the frontend's public attributed passages. A preflight CLI checks writable persistence, SEC contact configuration, corpus indexing and model integrity without downloading SEC data or performing inference.

Remaining scientific work: supplemental exhibit interpretation, outcome admission under the chosen protocol, development-only threshold selection, and reserved independent evaluation/calibration. These are unresolved evidence requirements; the implementation does not manufacture reviewer attestations or mark them complete. Hosting and cloud operations remain with the user.

Run locally:

```bash
.venv/bin/python -m src.web.preflight
.venv/bin/python -m src.web.server --port 8000
```

For explicitly experimental local scores, add `--research-scores`. The server-backed frontend exposes Research memo, Bankruptcy model and Filing changes; static hosting alone does not run these APIs.

Validation: 326 tests passed in the full suite (one existing toy neural-network convergence warning); 33 final focused integration checks passed. Browser checks exercised the research memo, all 63 model-input rows, public-score withholding, filing comparisons and mobile navigation. Backend preflight passed. Frozen artifact and original outcome packet hashes are unchanged.
