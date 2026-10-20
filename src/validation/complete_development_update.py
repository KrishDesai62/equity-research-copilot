"""Finish the frozen October 19 study after its two active acquisitions finish.

Does not deploy, score final companies, change sampling, or approve a release.
"""
import json
from pathlib import Path
import time

from src.validation import text_model_development, matched_model_development
from src.validation.development_release_preflight import run as preflight
from src.validation.development_update_summary import run as summarize


def main(timeout_seconds=7200):
    start=time.monotonic()
    required=[Path('data/processed/validation/text_models_20261019/expanded/selection.json'),
              Path('data/processed/validation/recent_development_20261019/report.json')]
    while not all(p.exists() for p in required):
        if time.monotonic()-start > timeout_seconds:
            raise TimeoutError('Acquisitions are incomplete; no final evaluation or release was attempted')
        time.sleep(5)
    report=json.loads(required[1].read_text())
    if report['errors']:
        summarize()
        raise ValueError('Recent acquisition has errors; resolve sources before model fitting')
    from src.validation.issuer_identity_admission_v2 import run as audit
    audit()
    text_model_development.select_phase()
    for phase in ['base','expanded']:
        matched_model_development.run(phase)
    matched_model_development.select_phase()
    preflight()
    result=summarize()
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':main()
