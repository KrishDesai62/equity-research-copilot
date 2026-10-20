"""Saved-data comparison of nested calibration and unchanged controls."""
import json
from pathlib import Path

import numpy as np

from src.validation.calibration_refinement import verify_saved

REPORT=Path('reports/calibration_refinement_20261019.md')


def run():
    result=verify_saved();selected=result['selected']
    fixed=next(c for c in result['candidates'] if c['base_model']==selected['base_model'] and c['method']=='fixed')
    improvement=1-selected['median_log_loss']/fixed['median_log_loss']
    lines=['# Probability calibration refinement — October 19, 2026','',
           'This pass improves the research probability model. It does not improve every detection measure and does not approve production use. The existing detection specialist remains separate.','',
           '## Why this experiment','',
           'The previous probability model overpredicted the lowest development risk band and underpredicted the highest bands. Its sigmoid calibration used a fixed regularization value of 0.1. A flatter calibration slope can suppress score differences even when the base classifier ranks companies reasonably well.',
           'The new method chooses among regularization values 0.01, 0.1, 1 and 10 using inner company folds. Each scored outer-fold company is excluded from the outcomes used to choose its calibration setting and fit its mapper. If an inner fit has insufficient class support, the declared fallback remains 0.1.',
           'Both the existing probability classifier and detection specialist are fixed: no base classifier was retrained, no feature was changed, and no company or evaluation weight was removed. The previous fixed-calibration scores reproduce within numerical tolerance.',
           'Five company-fold partitions were fixed before the run. Selection minimizes median weighted outer-fold log loss across all five, rather than choosing a favorable seed. The final research mapper’s regularization is chosen on the full development cohort by inner cross-validation after the method comparison.','',
           '## Results on the original comparison partition','',
           '| Model/calibration | ROC-AUC | PR-AUC | Log loss | Recall at precision ≥50% | Precision at recall ≥50% |',
           '|---|---:|---:|---:|---:|---:|']
    for candidate in result['candidates']:
        primary=candidate['partitions'][0];m=primary['metrics'];limits=primary['operating_limits']
        lines.append(f"| {candidate['name']} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['log_loss']:.5f} | {limits['maximum_recall_at_precision_50pct']:.1%} | {limits['maximum_precision_at_recall_50pct']:.1%} |")
    lines += ['', '## Sensitivity across all five fixed partitions','',
              '| Model/calibration | Median log loss | Median Brier | PR-AUC range |',
              '|---|---:|---:|---:|']
    for c in result['candidates']:
        ap=[r['metrics']['average_precision'] for r in c['partitions']]
        lines.append(f"| {c['name']} | {c['median_log_loss']:.5f} | {c['median_brier']:.5f} | {min(ap):.3f}–{max(ap):.3f} |")
    lines += ['',f"Selected research procedure: **{selected['name']}**. Median probability log loss improved by **{improvement:.1%}** relative to the same classifier’s fixed calibration. The final full-development mapper uses **C={result['final_mapper_regularization']}**.",
              'The detector’s nested calibration reduces probability loss but hurts its original-partition high-precision recall. Its earlier fixed-calibration specialist is therefore retained for detection research; probability loss and alert quality are separate objectives.', '',
              '## Limits and artifacts','',
              '- All 303 companies have already been used for development. Inner/outer separation prevents this calibration step from fitting a company’s label into its own score; it cannot undo previous model and feature selection exposure.',
              '- None of the candidates meets both unchanged 50% precision and 50% recall requirements on the original comparison partition. Independent outcome review, final evaluation and operating validation remain outstanding.',
              '- No final features, predictions, release record or AWS deployment were created.',
              '- Research probability artifact: `data/processed/validation/calibration_refinement_20261019/research_candidate.joblib`.',
              '- All partition predictions, fold-specific settings, fallback decisions, source hashes and reliability bins are saved alongside the artifact.', '', '## Reproduce locally','', '```bash',
              '.venv/bin/python -m src.validation.calibration_refinement',
              '.venv/bin/python -m src.validation.calibration_refinement_summary',
              '.venv/bin/python -m pytest -q tests/test_calibration_refinement.py','```']
    REPORT.write_text('\n'.join(lines)+'\n')
    return {'selected':selected['name'],'median_log_loss_improvement':improvement,'production_approval':False}


if __name__=='__main__':print(json.dumps(run(),indent=2))
