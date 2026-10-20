"""Finish the fixed metadata experiment after the active context fit completes."""
from pathlib import Path
import time


def main(timeout_seconds=3600):
    source=Path('data/processed/validation/context_models_20261019/expanded/selection.json')
    start=time.monotonic()
    while not source.exists():
        if time.monotonic()-start>timeout_seconds:raise TimeoutError('Context experiment is incomplete; final was not scored')
        time.sleep(5)
    from src.validation.market_context_model_development import run
    result=run('expanded')
    from src.validation.current_production_readiness import run as readiness
    readiness()
    print(result['selected'],flush=True)


if __name__=='__main__':main()
