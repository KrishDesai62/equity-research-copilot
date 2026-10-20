"""Evidence-bound readiness report; never serves or scores held-out predictions."""
import json
from pathlib import Path

from src.data_layer.research_dataset import write_json
from src.validation.cohort_v3_data import digest
from src.validation.final_outcome_audit_v2 import run as verify_outcomes
from src.validation.matched_model_development import ROOT as BASE, verify_saved as verify_base

ROOT=Path('data/processed/validation/production_readiness_20261019')


def assess(model, outcomes, review, packet_sha):
    blockers=[]
    if model['selected']['threshold'] is None or not model['development_feasible']:
        blockers.append('development_precision_recall_floor')
    if outcomes['positive_companies']<20 or outcomes['resolved_companies']<50:
        blockers.append('reserved_outcome_support')
    if review.get('packet_sha256')!=packet_sha or review.get('independent_reviews_complete') is not True:
        blockers.append('independent_review_of_current_packet')
    # These are future evidence requirements, not toggles that this diagnostic
    # can set to true. Completing development never grants production approval.
    blockers.extend(['untouched_final_performance','recent_probability_calibration',
                     'dated_trading_strategy_validation','live_aws_end_to_end_validation'])
    return blockers


def verify_source_proofs(proofs):
    seen={}
    for source in proofs:
        path=source['path'];expected=source['sha256']
        if path in seen and seen[path]!=expected:raise ValueError('Conflicting source proof hashes')
        seen[path]=expected
    for path,expected in seen.items():
        if digest(path)!=expected:raise ValueError('Reserved source bytes changed after outcome audit')
    return len(seen)


def verify_reserved_sources(outcomes):
    proofs=[]
    for name in outcomes['inputs']:
        path=Path(name)
        if path.suffix=='.json' and path.stem.isdigit() and len(path.stem)==10:
            proofs.extend(json.loads(path.read_text())['sources'])
    return verify_source_proofs(proofs)


def run():
    outcomes=verify_outcomes()
    verified_sources=verify_reserved_sources(outcomes)
    folder=BASE/'expanded';model=verify_base(folder)
    selection=folder/'selection.json'
    context=Path('data/processed/validation/context_models_20261019/expanded/selection.json')
    context_result=None
    if context.exists():
        from src.validation.context_model_development import verify_saved
        context_result=verify_saved(context.parent)
        # Its initial liquidity-negation heuristic is archived for comparison;
        # the subsequent market/context experiment contains the correction.
    combined=Path('data/processed/validation/market_context_models_20261019/expanded/selection.json')
    combined_result=None
    if combined.exists():
        from src.validation.market_context_model_development import verify_saved
        combined_result=verify_saved(combined.parent)
        if combined_result['selected']['name']!='frozen_semantic_baseline':
            model=combined_result;selection=combined
    refinement_result=None
    refinement_path=Path('data/processed/validation/calibration_refinement_20261019/selection.json')
    if refinement_path.exists():
        from src.validation.calibration_refinement import verify_saved as verify_refinement
        refinement_result=verify_refinement()
        chosen=refinement_result['selected']
        control=next(c for c in refinement_result['candidates'] if c['base_model']==chosen['base_model'] and c['method']=='fixed')
        if chosen['method']=='nested' and chosen['median_log_loss']<control['median_log_loss']:
            model={'selected':{'name':chosen['name'],'threshold':refinement_result['release_threshold'],
                               'metrics':refinement_result['primary_partition']['metrics']},
                   'development_feasible':refinement_result['release_threshold'] is not None}
            selection=refinement_path
    detection_result=None
    detection_path=Path('data/processed/validation/detection_specialist_20261019/report.json')
    if detection_path.exists():
        from src.validation.detection_research_candidate import run as verify_specialist
        detection_result=verify_specialist()
    packet=Path('data/processed/validation/final_outcome_audit_v2_20261019/manifest.json')
    packet_sha=json.loads(packet.read_text())['packet_sha256']
    review_path=Path('data/processed/validation/current_review_status_20261019/status.json')
    review=json.loads(review_path.read_text()) if review_path.exists() else {}
    blockers=assess(model,outcomes,review,packet_sha)
    result={'candidate_selection_path':str(selection),'candidate_selection_sha256':digest(selection),
            'selected':model['selected'],'context_experiment_complete':context_result is not None,
            'combined_corrected_experiment_complete':combined_result is not None,
            'probability_refinement_complete':refinement_result is not None,
            'research_detection_candidate':None if detection_result is None else {
                'name':detection_result['candidate']['name'],
                'metrics':detection_result['candidate']['metrics'],
                'release_threshold':detection_result['release_threshold'],
                'production_approval':False,'report_sha256':digest(detection_path)},
            'source_corroborated_reserved_positives':outcomes['positive_companies'],
            'source_corroborated_reserved_negatives':outcomes['negative_companies'],
            'unknown_reserved_outcomes':outcomes['unknown_companies'],
            'verified_reserved_raw_sources':verified_sources,
            'independently_reviewed_units':review.get('approved_current_units',0),
            'current_packet_sha256':packet_sha,'blockers':blockers,
            'model_prediction_target':'target_sec_disclosed_petition_1yr',
            'scope':'SEC-disclosed issuer Chapter 7/11; not exhaustive court bankruptcies or profitable buy/sell decisions',
            'heldout_predictions_created':False,'production_approval':False,
            'source_hashes':{str(packet):digest(packet),str(review_path):digest(review_path) if review_path.exists() else None}}
    ROOT.mkdir(parents=True,exist_ok=True);write_json(ROOT/'status.json',result)
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
