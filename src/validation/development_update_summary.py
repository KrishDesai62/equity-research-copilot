"""Summarize saved development results; never fit, score, or read final rows."""
import json
from pathlib import Path

import pandas as pd

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.development_release_preflight import weight_support
from src.validation.sec_disclosure_data import TARGET
from src.validation.development_errors import operating_limits

ROOT=Path('data/processed/validation/development_summary_20261019')
REPORT=Path('reports/development_update_20261019.md')


def run():
    inputs={}; experiments=[]
    def read(path):
        path=Path(path)
        if not path.exists():return None
        inputs[str(path)]=digest(path)
        return json.loads(path.read_text())
    acquisition=read('data/processed/validation/historical_training_20261019/report.json')
    audit=read('data/processed/validation/issuer_identity_admission_v2_20261019/report.json')
    recent=read('data/processed/validation/recent_development_20261019/report.json')
    review=read('data/processed/validation/disclosure_review_status_20261019/status.json') or {}
    for study in ['text_models_20261019','matched_models_20261019']:
        for phase in ['base','expanded']:
            path=Path('data/processed/validation')/study/phase/'selection.json'
            record=read(path)
            if record:
                experiments.append({'study':study,'phase':phase,'selected':record['selected'],
                    'training':record['training'],'calibration':record['calibration'],
                    'development_feasible':record['development_feasible'],
                    'constant_baseline_oof':record['constant_baseline_oof'],'candidates':record['candidates']})
    supports={}
    for name,path in [('original','data/processed/validation/sec_filing_signals_v4/calibration.csv'),
                      ('expanded','data/processed/validation/issuer_identity_admission_v2_20261019/calibration.csv')]:
        path=Path(path)
        if path.exists():
            frame=pd.read_csv(path,dtype={'cik':str});inputs[str(path)]=digest(path)
            supports[name]=weight_support(frame.loc[frame[TARGET].notna()])
    tradeoffs={}
    prediction_path=Path('data/processed/validation/matched_models_20261019/expanded/calibration_predictions.csv')
    if prediction_path.exists():
        predictions=pd.read_csv(prediction_path,dtype={'cik':str})
        inputs[str(prediction_path)]=digest(prediction_path)
        current=next((e for e in experiments if e['study']=='matched_models_20261019' and e['phase']=='expanded'),None)
        if current:
            tradeoffs={c['name']:operating_limits(predictions,c['name']) for c in current['candidates']}
    markers=[p for p in [Path('data/processed/validation/sec_disclosure_v1/experiment/final_evaluation_started.json'),
                         Path('data/processed/validation/sec_disclosure_v2/final_evaluation_started.json')] if p.exists()]
    result={'issuer_identity_audit':None if not audit else {'counts':audit['counts'],'changes':audit['changes']},'historical_acquisition':None if not acquisition else {k:acquisition[k] for k in ['selected_companies','rows','eligible_rows','positive_companies','errors']},
            'recent_acquisition':None if not recent else {k:recent[k] for k in ['selected_companies','rows','positive_companies','negative_companies','unknown','errors']},
            'experiments':experiments,'development_weight_support':supports,'expanded_operating_limits':tradeoffs,
            'independent_reviewed_units':review.get('approved_current_units',0),
            'independent_review_complete':review.get('independent_reviews_complete',False),
            'known_final_consumption_markers':[str(p) for p in markers],
            'final_rows_loaded':False,'production_approval':False,'input_hashes':inputs}
    ROOT.mkdir(parents=True,exist_ok=True);write_json(ROOT/'summary.json',result)
    lines=['# Public-source development update — October 19, 2026','',
           'This report reads saved development results only. It does not rescore the reserved final test or approve a model for production.','',
           '## Sources and dataset changes','',
           '- SEC original nonfinancial 10-K annual-filer frames define the population. Companyfacts plus original 10-K/10-Q submission metadata supply point-in-time financial features; amended or later facts cannot replace the original prediction input.',
           '- Primary SEC Item 1.03 search results identify candidate companies. A candidate is not a bankruptcy label: explicit issuer Chapter 7/11 petition evidence must fall inside the prediction horizon. Subsidiary-only, unclear and incomplete outcomes remain unknown.',
           '- Negatives require continued original financial reporting across the full one-year horizon and a disclosure review through the additional 200-day reporting window. These are SEC-disclosure negatives, not proof that no bankruptcy occurred in any court.',
           '- Training excludes all recent calibration and final company reservations. Sampling is fixed before financial acquisition; inverse inclusion weights apply to both positive and negative examples. Missing sources are not labeled zero.','']
    if acquisition:
        lines += [f"Historical expansion: {acquisition['selected_companies']} additional companies sampled; **{acquisition['eligible_rows']:,} eligible training snapshots and {acquisition['positive_companies']} positive companies** in the combined dataset, compared with 1,282 snapshots and 44 positive companies before expansion. Acquisition errors: {len(acquisition['errors'])}.",'']
    if recent:
        lines += [f"Recent development expansion: {recent['selected_companies']} new companies sampled; combined {recent['positive_companies']} positive and {recent['negative_companies']} negative companies, with {recent['unknown']} unknown outcomes. Acquisition errors: {len(recent['errors'])}.",'']
    else:
        lines += ['A separate 299-company recent development expansion is frozen and acquiring public evidence. Its results are not yet treated as complete.','']
    if audit:
        a=audit['counts'];lines += [f"After the uniform issuer-identity correction, the usable expanded training dataset has **{a['expanded_training']['eligible_rows']:,} snapshots and {a['expanded_training']['positive_companies']} positive companies**. Corrected recent development has **{a['calibration']['eligible_rows']} resolved companies**, including **{a['calibration']['positive_companies']} positives**. Unconfirmed former positives are unknown, never new negatives. Earlier counts above describe acquisition before this correction.",'', 'The correction rejects an issuer merely named as owner of a petitioning subsidiary, and avoids binding a prior Current Report filing date as the petition date. UpHealth, Iconic Brands and RVL subsidiary reports triggered conservative exclusions. The older final evidence packet must be audited and reissued under the same rule before final validation.','']
    lines += ['## Filing-text features','',
              'Eight features measure visible going-concern, substantial-doubt, covenant-breach, liquidity-shortfall, restructuring, auditor-change and financing-dependency language, plus document length. Counts are normalized by visible word count. Hidden XBRL, scripts and styles are excluded; access-denial pages and insufficient text fail acquisition instead of becoming zero counts.',
              'Each document is the exact original financial filing available before the prediction date. Saved evidence includes the source URL, raw-byte hash, visible-text hash, matched terms and context proofs. Phrase frequency can reflect boilerplate and is not an asserted legal or audit conclusion.','',
              '## Completed model comparisons','',
              'All metrics below are weighted, cross-fitted DEVELOPMENT metrics after model selection. They are not untouched-test accuracy or guaranteed current-market probabilities. The two study versions use different calibration cohorts; compare phases within the same study.','',
              '| Study | Training phase | Selected candidate | ROC-AUC | PR-AUC | Log loss | Brier | Meets P≥50%, R≥50% |',
              '|---|---|---|---:|---:|---:|---:|---|']
    for e in experiments:
        m=e['selected']['metrics'];lines.append(f"| {e['study']} | {e['phase']} | {e['selected']['name']} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['log_loss']:.5f} | {m['brier']:.5f} | {'Yes' if e['development_feasible'] else 'No'} |")
    if tradeoffs:
        best=max(v['maximum_precision_at_recall_50pct'] for k,v in tradeoffs.items() if k!='previous_frozen_reference')
        lines += ['', f'On corrected expanded development data, the best precision available at recall of at least 50% is **{best:.1%}** across the newly fitted candidate families. This remains below the 50% precision requirement. ROC-AUC is a ranking metric, not percent accuracy or evidence that an actionable threshold exists.','',
                  '| Corrected expanded candidate | ROC-AUC | PR-AUC | Log loss | Converged |',
                  '|---|---:|---:|---:|---|']
        current=next(e for e in experiments if e['study']=='matched_models_20261019' and e['phase']=='expanded')
        for c in current['candidates']:
            m=c['metrics'];lines.append(f"| {c['name']} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['log_loss']:.5f} | {c['converged']} |")
    lines += ['', 'Each phase compares the same nine candidate families, including logistic regression, boosted trees, random forest, a neural network and the previous frozen reference. Preprocessing is fitted on historical training data; sigmoid calibration is fitted outside each held company fold. The previous frozen reference retains its original 299-row base training history. It is diagnostic only in the corrected-label study and cannot be selected for release.', '', '## Evaluation support and remaining gates','']
    for name,support in supports.items():
        positive=support['positive'];largest=positive['largest_company_weight_fraction']
        lines += [f"- {name.capitalize()} calibration: {positive['companies']} positive companies; positive effective sample size **{positive['effective_sample_size']:.2f}**. Largest positive-company weight share: {largest:.1%}. This measures weight concentration, not independent outcome verification."]
    lines += [f"- Independent review: {review.get('approved_current_units',0)} current final units completed. Automated source corroboration is not an independent human sign-off.",
              '- Threshold feasibility, fresh held-out performance, representative probability calibration, trading-agent profitability and live AWS operation are separate requirements. More rows or passing unit tests do not establish them.',
              '- No production approval is granted by this update. AWS deployment remains deferred as requested.', '', '## Local commands','', '```bash',
              '.venv/bin/python -m pip install -r requirements-filing-text.lock',
              '.venv/bin/python -m src.validation.historical_training_expansion',
              '.venv/bin/python -m src.validation.text_model_development --phase base',
              '.venv/bin/python -m src.validation.text_model_development --phase expanded',
              '.venv/bin/python -m src.validation.text_model_development --phase select',
              '.venv/bin/python -m src.validation.recent_development_expansion',
              '.venv/bin/python -m src.validation.issuer_identity_admission_v2',
              '.venv/bin/python -m src.validation.matched_model_development --phase base',
              '.venv/bin/python -m src.validation.matched_model_development --phase expanded',
              '.venv/bin/python -m src.validation.matched_model_development --phase select',
              '.venv/bin/python -m src.validation.development_update_summary', '```','',
              'Acquisition is resumable and frozen inputs are checked against hashes. Do not change a frozen plan, delete evaluation-consumption markers, fabricate review records or run two fits for the same phase concurrently.']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
