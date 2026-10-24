"""Public implementation status; exposes no credentials or per-user information."""
from src.web.providers import capabilities


def describe(research):
    caps=capabilities()
    try:
        model=research.model_status()
        model_status='released' if model.get('production_approval') else 'gated' if model.get('release_status')=='not_approved' else 'unavailable'
    except Exception:model_status='unavailable'
    return dict(version=1,source='server_configuration',layers={
        'sec':{'status':'available','label':'Connected'},
        'retrieval':{'status':'available','label':'Hybrid retrieval'},
        'agents':{'status':'available','label':'Cited evidence'},
        'features':{'status':'available','label':'63 input schema'},
        'model':{'status':model_status,'label':{'released':'Released','gated':'Scores gated','unavailable':'Unavailable'}[model_status]},
        'report':{'status':'available','label':'Combined research'},
        'neural':{'status':'research_only','label':'Research comparison'},
        'chroma':{'status':'research_only','label':'Original research pipeline'},
    },ai_synthesis='configured' if caps['ai_synthesis'] else 'not_configured',
       market_data='configured' if caps['market_data'] else 'not_configured',
       limitation='Configuration status does not prove retrieval relevance, prediction accuracy or provider availability. No model inference runs for this page.')
