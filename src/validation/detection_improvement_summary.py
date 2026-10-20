"""Explain detection/probability trade-offs using saved, replay-checked results."""
import json
from pathlib import Path

from src.validation.detection_development import verify_saved
from src.validation.detection_research_candidate import run as verify_specialist

REPORT=Path('reports/detection_improvement_20261019.md')


def run():
    result=verify_saved();specialist=verify_specialist()
    current=next(c for c in result['candidates'] if c['name']=='frozen_current_reference')
    best=specialist['candidate'];operating=specialist['diagnostic_operating_point'];conf=operating['unweighted_confusion']
    lines=['# Detection improvement — October 19, 2026','',
        'A new research specialist improves development alert ranking and recall at a fixed precision floor. The retained probability model still has better probability loss. Neither is production-approved.','',
        '## What changed','',
        'The experiment compares native missing-value handling in gradient-boosted trees, full versus square-root class balancing, company-normalized training loss, recency weighting, stronger regularization, Extra Trees, XGBoost and two fixed probability ensembles. No additional public source was downloaded and no final companies were scored.',
        'Company normalization reduces domination by companies with many historical snapshots. Recency weighting is a separate alternative, not an automatic policy change. Both affect training loss only: the actual dataset inclusion weights, development outcomes and evaluation weights remain unchanged.',
        'The same 2,174 historical training snapshots and 303 development companies were used. All candidate settings were fixed before this run. The existing model’s saved cross-fitted probabilities reproduce exactly within numerical tolerance.','',
        '## Results','',
        '| Measure | Retained probability model | Research detection specialist |',
        '|---|---:|---:|']
    for key,label in [('roc_auc','ROC-AUC'),('average_precision','PR-AUC'),('log_loss','Probability log loss'),('brier','Brier loss')]:
        lines.append(f"| {label} | {current['metrics'][key]:.5f} | {best['metrics'][key]:.5f} |")
    lines.append(f"| Maximum recall at precision ≥50% | {current['operating_limits']['maximum_recall_at_precision_50pct']:.1%} | {best['operating_limits']['maximum_recall_at_precision_50pct']:.1%} |")
    lines += ['',f"The specialist is **{best['name']}**. At the development-selected diagnostic operating point, weighted precision is **{operating['metrics']['precision']:.1%}** and weighted recall is **{operating['metrics']['recall']:.1%}**. Raw counts: **{conf['tp']} bankruptcies detected, {conf['fn']} missed, {conf['fp']} false alert and {conf['tn']} true negatives**.",
        'These are repeatedly reused development companies, not independent final-test performance. The specialist and its diagnostic operating point were selected after inspecting this development experiment; the apparent gain may be optimistic. Raw and design-weighted precision differ because the study oversamples outcome proxies.',
        'This improves a high-precision alerting diagnostic, not the required precision/recall pair. The unchanged production requirement remains precision ≥50% AND recall ≥50%. The specialist’s release threshold remains null.','',
        '## Full fixed candidate comparison','',
        '| Candidate | PR-AUC | ROC-AUC | Log loss | Recall at precision ≥50% | Precision at recall ≥50% |',
        '|---|---:|---:|---:|---:|---:|']
    for candidate in result['candidates']:
        m=candidate['metrics'];limits=candidate['operating_limits']
        lines.append(f"| {candidate['name']} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} | {m['log_loss']:.5f} | {limits['maximum_recall_at_precision_50pct']:.1%} | {limits['maximum_precision_at_recall_50pct']:.1%} |")
    lines += ['', '## Saved artifacts and controls','',
        '- Retained probability model: `data/processed/validation/market_context_models_20261019/expanded/candidate.joblib`.',
        '- New research specialist: `data/processed/validation/detection_specialist_20261019/research_detection.joblib`.',
        '- The specialist was refitted with the frozen settings; its predictions must reproduce the saved experiment before export. Its feature schema, source hashes and prediction target accompany the artifact.',
        '- No production registry activation, release record, held-out prediction or AWS deployment was created.',
        '- Independent outcome review, representative final probability validation, trading validation and AWS operation remain separate requirements.','',
        '## Reproduce locally','', '```bash',
        '.venv/bin/python -m src.validation.detection_development',
        '.venv/bin/python -m src.validation.detection_research_candidate',
        '.venv/bin/python -m src.validation.detection_improvement_summary',
        '.venv/bin/python -m pytest -q tests/test_detection_development.py', '```']
    REPORT.write_text('\n'.join(lines)+'\n')
    return {'specialist':best['name'],'probability_model_retained':True,'production_approval':False}


if __name__=='__main__':print(json.dumps(run(),indent=2))
