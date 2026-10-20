"""Write a readable report from checksum-verified saved V3 evidence only."""
import json
from pathlib import Path

from src.validation.cohort_v3 import summary
from src.validation.cohort_v3_data import ROOT


def run():
    final = summary()
    data = json.loads((ROOT / 'dataset_manifest.json').read_text())
    development = json.loads((ROOT / 'development_report.json').read_text())
    quality = json.loads((ROOT / 'data_quality_report.json').read_text())
    selected = development['selected']
    score = final['design_weighted']
    constant = final['constant_baseline']
    lines = ['# V3 historical bankruptcy experiment', '',
        '**Production status: ' + ('ready' if final['production_ready'] else 'blocked; research only') + '.**', '',
        'The fixed acquisition, source audit, weighted model comparison and one-use final',
        'evaluation have completed locally. No serving model was replaced and AWS was not used.', '',
        '## Dataset', '',
        f"The 48 quarterly SEC archives define {data['frame_companies']:,} historical companies",
        f"after exclusions. The fixed plan selected {data['sampled_companies']:,} companies.",
        f"Acquisition produced {data['snapshot_rows']:,} calendar snapshots: {data['eligible_snapshot_rows']:,}",
        f"source-eligible rows and {data['unknown_rows']:,} unknown/outside-scope rows, which are not training labels.",
        f"Checkpoints: {quality['checkpoints']}. Raw source files verified: {quality['raw_sources']:,}.", '',
        'This includes all registry case companies in the defined frame plus a 10% random',
        'background sample. Inverse company inclusion weights apply to every snapshot.',
        'Coverage remains restricted by public-filer history, financial-source availability,',
        'the BRD interval through 2022 and large-public-company eligibility. It is not a',
        'complete current-market dataset. See the [data card](cohort_v3_data_card.md).', '',
        '| Stage | Rows | Companies | Positive companies |', '|---|---:|---:|---:|']
    for stage, item in [('Training', development['training']), ('Calibration / selection', development['calibration']), ('Final test', final['final'])]:
        lines.append(f"| {stage} | {item['rows']} | {item['companies']} | {item['positive_companies']} |")
    lines += ['', 'Companies and time periods are disjoint across stages. Calibration uses',
        'three company-disjoint folds. Preprocessing is fitted only on training rows.', '',
        '## Fixed candidate comparison', '',
        '| Candidate | Weighted out-of-company-fold log loss | Weighted Brier loss | Warnings |', '|---|---:|---:|---|']
    for item in development['candidates']:
        m = item['out_of_company_fold_metrics']
        lines.append(f"| {item['name']} | {m['log_loss']:.6f} | {m['brier']:.6f} | {'; '.join(item['warnings']) or 'none'} |")
    lines += ['', f"Selected before final predictions: **{selected['name']}**, using weighted calibration log loss",
        'with Brier loss as the tie-breaker. All other final candidate scores remain uncomputed.',
        f"Threshold: `{selected['threshold']}`. An absent threshold means no calibration operating point",
        'met both fixed 50% precision and 50% recall floors; it was not lowered afterward.', '',
        '## One-use final evaluation', '',
        '| Measure | Selected candidate | Constant calibration-rate baseline |', '|---|---:|---:|',
        f"| Brier loss (lower is better) | {score['brier']:.6f} | {constant['brier']:.6f} |",
        f"| Log loss (lower is better) | {score['log_loss']:.6f} | {constant['log_loss']:.6f} |",
        f"| Mean probability | {score['mean_probability']:.2%} | {constant['mean_probability']:.2%} |", '',
        f"Weighted observed event frequency: {score['weighted_prevalence']:.2%}."]
    for key, label in [('precision', 'Precision'), ('recall', 'Recall'), ('accuracy', 'Accuracy'), ('average_precision', 'PR-AUC'), ('roc_auc', 'ROC-AUC')]:
        if score.get(key) is not None:
            lines.append(f"{label}: **{score[key]:.2%}**.")
    if selected['threshold'] is None:
        lines += ['Classification precision, recall and accuracy are not reported because there is',
                  'no qualifying frozen operating threshold. Ranking scores do not replace this requirement.']
    lines += ['', 'These are design-weighted historical estimates, not verified 2026 market probabilities.',
        'Unweighted audit metrics, fixed-bin calibration diagnostics and company-cluster',
        'bootstrap intervals are retained in `final_evaluation.json`.', '',
        '| Measure | Exploratory company-cluster 95% interval |', '|---|---|']
    for key, interval in final['company_cluster_95_percent_intervals'].items():
        lines.append(f"| {key} | {interval['lower']:.6f} to {interval['upper']:.6f} |")
    lines += ['', 'Intervals are conditional on the observed companies and design weights. They do not',
        'correct source nonresponse, censoring/eligibility selection or imperfect registry identities.', '',
        '## Gates and remaining work', '']
    for key, passed in final['gates'].items():
        lines.append(f"- {'PASS' if passed else 'BLOCKED'}: {key.replace('_', ' ')}.")
    lines += ['', 'The final test is consumed. Its results cannot be used to retune this candidate',
        'and then reused as independent validation. A future revision needs newly reserved',
        'companies/outcomes. Current-population outcome coverage and external calibration',
        'review remain necessary before a production risk probability can be approved.',
        'News-driven buy/sell recommendations also require their own dated evidence and',
        'out-of-time strategy evaluation; this experiment does not validate trading returns.', '',
        '## Evidence and commands', '',
        f"- Dataset SHA-256: `{final['dataset_sha256']}`.",
        f"- Candidate SHA-256: `{final['artifact_sha256']}`.",
        '- Dataset, source seal, raw hashes, checkpoints, attrition report and frozen model:',
        '  `data/processed/validation/cohort_v3/`.',
        '- Frozen sampling/evaluation specification: `config/cohort_v3.json` and',
        '  `config/cohort_v3_acquisition.json`.', '',
        'Read verified saved evidence without downloading, retraining or rescoring:', '',
        '```bash', '.venv/bin/python -m src.validation.cohort_v3 summary',
        '.venv/bin/python -m src.validation.local_workflow', '```', '']
    path = Path('reports/cohort_v3_results.md')
    path.write_text('\n'.join(lines))
    print(str(path))
    return path


if __name__ == '__main__':
    run()
