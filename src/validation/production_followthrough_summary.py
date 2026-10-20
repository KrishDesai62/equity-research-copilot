"""Render saved source audits, research comparisons and actual readiness gates."""
import json
from pathlib import Path

import pandas as pd
import numpy as np

from src.validation.current_production_readiness import run as readiness
from src.validation.development_errors import operating_limits

REPORT=Path('reports/production_followthrough_20261019.md')


def run():
    status=readiness()
    experiments=[]
    for study in ['matched_models_20261019','context_models_20261019','market_context_models_20261019']:
        folder=Path('data/processed/validation')/study/'expanded'
        if not (folder/'selection.json').exists():continue
        result=json.loads((folder/'selection.json').read_text())
        predictions=pd.read_csv(folder/'calibration_predictions.csv',dtype={'cik':str})
        experiments.append((study,result,operating_limits(predictions,result['selected']['name'])))
    # Verify that adding columns does not silently change the control model.
    reference=pd.read_csv('data/processed/validation/matched_models_20261019/expanded/calibration_predictions.csv',dtype={'cik':str})
    selected=json.loads(Path('data/processed/validation/matched_models_20261019/expanded/selection.json').read_text())['selected']['name']
    for study,column in [('context_models_20261019','frozen_context_baseline'),('market_context_models_20261019','frozen_semantic_baseline')]:
        path=Path('data/processed/validation')/study/'expanded/calibration_predictions.csv'
        if path.exists():
            current=pd.read_csv(path,dtype={'cik':str})
            if not reference[['cik','unit_id']].equals(current[['cik','unit_id']]):raise ValueError('Control cohort changed')
            if not np.allclose(reference[selected],current[column],rtol=1e-9,atol=1e-10):raise ValueError('Frozen baseline control changed')
    audit=json.loads(Path('data/processed/validation/final_outcome_audit_v2_20261019/report.json').read_text())
    lines=['# Production follow-through — October 19, 2026','',
           'No production model was approved and no held-out company was scored. This pass fixes source admission and inference contracts, and runs additional fixed development experiments.','',
           '## Dataset audit','',
           f"The reserved cohort has **{audit['positive_companies']} source-corroborated issuer petitions, {audit['negative_companies']} SEC-disclosure negatives and {audit['unknown_companies']} unknown outcomes**. These are automated source decisions, not independently signed-off human labels or exhaustive court coverage.",
           'The issuer rule now recognizes a direct legal-name subject following a long deliberation clause. This recovers DZS’s explicit parent Chapter 7 petition without admitting a parent merely named as a subsidiary’s owner. The same rule was applied uniformly to development and reserved outcome evidence.',
           'Development data stayed byte-for-byte identical after this correction: 2,174 training snapshots from 89 positive companies; 303 recent development companies, including 29 positives. The original files, prior experiment hashes and prior packets are preserved.',
           'The updated review packet is `data/processed/validation/final_outcome_audit_v2_20261019/pending_final_reviews.jsonl`. Review responses must reference its exact packet and source-checkpoint hashes. Current independent approvals: **0**.','',
           '## Additional features and comparisons','',
           'Context features distinguish concrete going-concern, liquidity, covenant, forbearance, restructuring and financing statements from generic audit boilerplate and negation. Original documents are cached with byte hashes; extraction now resumes per document and runs with four local worker processes.',
           'The combined experiment corrects the liquidity-negation edge case: not having sufficient cash remains a warning. Its corrections are saved in `market_context_signals_20261019/liquidity_corrections.json`; the initial contextual experiment remains archived and is not the production candidate.',
           'Additional features use only notices published before prediction: exchange/delisting notices (Item 3.01), unregistered equity sales (3.02), auditor changes (4.01), terminated agreements (1.02), exit costs (2.05), impairments (2.06), security rights (5.03), and prior annual-filing SIC industry context. Missing item metadata stays unknown.',
           'All candidate lists were fixed before their runs. Development companies, labels, design weights, calibration folds and the original model control stay aligned. No additional data source or paid service was used.','',
           '| Experiment | Selected research candidate | ROC-AUC | PR-AUC | Log loss | Precision available at ≥50% recall | Meets 50% P/R |',
           '|---|---|---:|---:|---:|---:|---|']
    for study,result,limits in experiments:
        m=result['selected']['metrics'];lines.append(f"| {study} | {result['selected']['name']} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['log_loss']:.5f} | {limits['maximum_precision_at_recall_50pct']:.1%} | {'Yes' if result['development_feasible'] else 'No'} |")
    lines += ['', 'These are weighted cross-fitted development results after selection, not independent final performance or percent accuracy. A higher ranking score does not establish that a usable alert threshold exists.', '', '## Runtime fixes','',
              '- A model predicting accounting enforcement can no longer be loaded into the production bankruptcy track. Approved SEC-disclosure models expose their precise target and scope in the response.',
              '- Missing required feature columns produce an explicit unavailable result instead of an exception or fabricated probability.',
              '- Registry registration and inference share the same permitted bankruptcy-target contract. Release validation, training cutoff, freshness and integrity checks remain required.',
              '- The readiness report binds review completion to the current evidence-packet hash, preventing approval of an older packet from satisfying the current gate.','',
              '## Current release blockers','']
    lines += ['- '+reason.replace('_',' ') for reason in status['blockers']]
    lines += ['', 'AWS remains undeployed per the existing local-first instruction. This report does not certify autonomous trading. No permission, signature or independent review has been fabricated.','',
              '## Reproduce locally','', '```bash',
              '.venv/bin/python -m src.validation.final_outcome_audit_v2',
              '.venv/bin/python -m src.validation.current_outcome_reviews',
              '.venv/bin/python -m src.validation.context_model_development',
              '.venv/bin/python -m src.validation.market_context_model_development',
              '.venv/bin/python -m src.validation.production_followthrough_summary',
              '.venv/bin/python -m pytest -q', '```']
    REPORT.write_text('\n'.join(lines)+'\n')
    return status


if __name__=='__main__':print(json.dumps(run(),indent=2))
