"""Cited research and versioned model inference, with explicit research/release modes."""
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.agents_full.bull_agent import BullAgent
from src.agents_full.bear_agent import BearAgent
from src.agents_full.synthesis_agent import synthesize_research
from src.data_layer.research_dataset import write_json
from src.ml.artifacts import load_model
from src.ml.release import validate_release
from src.ml.risk_contracts import target_definition
from src.rag.retrieval import retrieve
from src.web.model_inputs import imported_inputs

REPO=Path(__file__).resolve().parents[2]


class ResearchService:
    def __init__(self,imports,*,allow_research=False,config=None):
        self.imports=imports
        self.allow_research=allow_research
        self.config=config or json.loads((REPO/'config/serving_model.json').read_text())
        self.lock=Lock()
        self._artifact=None

    def company(self,ticker):
        payload=self.imports.payload(ticker)
        if payload:return payload['company']
        return next((c for c in self.imports.catalog()['companies'] if c['ticker']==ticker),None)

    def _load(self):
        config=self.config
        path=REPO/config['artifact']
        for name,expected in config.get('feature_kernel_sha256',{}).items():
            if sha256((REPO/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('A frozen feature extractor changed; inference is disabled.')
        if sha256(path.read_bytes()).hexdigest()!=config['artifact_sha256']:
            raise ValueError('Serving model checksum does not match the configured version.')
        if self._artifact is None:
            self._artifact=load_model(path,expected_sha256=config['artifact_sha256'])
        from src.ml.detection_models import FEATURES
        if self._artifact['feature_cols']!=FEATURES or self._artifact['target']!=config['target']:
            raise ValueError('Serving model target or ordered feature schema mismatch.')
        return self._artifact

    def model_status(self):
        result=dict(name='Histogram gradient boosting',target=self.config['target'],
                    target_definition=target_definition('distress',self.config['target']),
                    artifact_sha256=self.config['artifact_sha256'],expected_features=63,
                    release_status='not_approved',research_scores_enabled=self.allow_research,
                    threshold=None,production_approval=False)
        try:
            with self.lock:artifact=self._load()
            result['trained_through']=artifact['trained_through']
            release_path=REPO/self.config['release']
            if release_path.exists():
                release=validate_release(release_path,REPO/self.config['artifact'])
                result.update(release_status='approved',threshold=release['threshold'],production_approval=True,
                              released_on=release['released_on'])
            else:
                result['reason']='Research candidate; final outcome review, independent evaluation and probability calibration remain incomplete.'
        except (OSError,ValueError):
            result.update(release_status='unavailable',reason='The model artifact or release record could not be verified.')
        return result

    def risk(self,ticker,as_of):
        date.fromisoformat(as_of)
        if self.company(ticker) is None:
            return dict(status='unknown_company',ticker=ticker,as_of=as_of,score=None,reasons=['Import or select a saved company first.'])
        status=self.model_status()
        result=dict(ticker=ticker,as_of=as_of,status='unavailable',score=None,risk_category=None,
                    alert=None,threshold=None,production_approval=False,model=status,reasons=[])
        if as_of in self.config['protected_evaluation_dates']:
            result['reasons']=['This date is reserved for a separate evaluation. Interactive scoring is disabled.']
            return result
        if status['release_status']=='unavailable':
            result['reasons']=[status['reason']]
            return result
        if as_of<=status['trained_through']:
            result['reasons']=['The requested date precedes the model training cutoff.']
            return result
        if status['release_status']=='approved' and as_of<status['released_on']:
            result['reasons']=['The approved release was not available at the requested date.']
            return result
        try:
            features=imported_inputs(self.imports,ticker,as_of)
        except (OSError,ValueError) as exc:
            result['reasons']=[str(exc) if isinstance(exc,ValueError) else 'Required SEC feature sources are unavailable.']
            return result
        report_path=self.imports.root/'model_inputs'/ticker/f'{as_of}.json'
        write_json(report_path,features)
        result['inputs']={key:features[key] for key in ['feature_filing_date','source_url','ordered_features',
            'values','observed_features','missing_features','financial_missing_features','metadata_complete']}
        if not features['eligible']:
            result['reasons']=features['ineligible_reasons']
            return result
        if status['release_status']!='approved' and not self.allow_research:
            result.update(status='research_only',reasons=['Inputs are prepared. Public risk scores remain disabled until a validated model release is available.'])
            return result
        with self.lock:
            artifact=self._load()
            if artifact['feature_cols']!=features['ordered_features']:
                raise ValueError('Inference input order differs from the trained model.')
            X=pd.DataFrame([{k:np.nan if v is None else v for k,v in features['values'].items()}],columns=artifact['feature_cols'])
            with threadpool_limits(limits=2):
                model=artifact['model'];classes=list(model.classes_)
                if set(classes)!={0,1} or len(classes)!=2:raise ValueError('Invalid model classes.')
                probabilities=np.asarray(model.predict_proba(X),dtype=float)
            if probabilities.shape!=(1,2) or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any() or not np.allclose(probabilities.sum(axis=1),1):
                raise ValueError('Model returned an invalid probability distribution.')
            score=float(probabilities[0,classes.index(1)])
        approved=status['release_status']=='approved'
        result.update(status='validated_score' if approved else 'unvalidated_research_score',score=score,
            production_approval=approved,threshold=status['threshold'] if approved else None,
            alert=bool(score>=status['threshold']) if approved else None,
            limitation='Experimental model output, not a validated personal bankruptcy probability or investment recommendation.' if not approved else status['target_definition'])
        return result

    def memo(self,ticker,as_of):
        company=self.company(ticker)
        if company is None:raise ValueError('Company is not imported or covered.')
        retriever=self.imports.retriever(ticker) or self.imports.base_retriever()
        bull=BullAgent(k_results=3,retriever=retriever).analyze_bull_case(ticker,company['cik'],as_of_date=as_of)
        bear=BearAgent(k_results=3,retriever=retriever).analyze_bear_case(ticker,company['cik'],as_of_date=as_of)
        from src.agents_full.evidence import filing_evidence
        liquidity_sources=filing_evidence('liquidity cash flow debt maturities funding going concern',ticker,company['cik'],3,as_of,retriever=retriever)
        liquidity=dict(citations=liquidity_sources,evidence_status='available' if liquidity_sources else 'no_cited_evidence')
        result=synthesize_research(company,as_of,bull,bear,self.risk(ticker,as_of),liquidity)
        from src.web.providers import synthesize
        result['synthesis']=synthesize('Summarize growth opportunities and business risks, with balanced evidence and no buy/sell recommendation.',result['citations'],model_context=result['combined_assessment']['model'])
        return result

    def changes(self,ticker,as_of):
        from src.data_layer.filing_changes import Filing,select_filing_pair,compare_filings
        company=self.company(ticker)
        if company is None:raise ValueError('Company is not covered.')
        payload=self.imports.payload(ticker)
        if payload and (self.imports.root/'companies'/f'{ticker}.json').exists():
            pointer=json.loads((self.imports.root/'companies'/f'{ticker}.json').read_text())
            manifest=json.loads((self.imports.root/'versions'/pointer['version']/'manifest.json').read_text())
            files={s['url']:s for s in manifest['sources'] if '/Archives/' in s.get('url','')}
            filings=payload['filings']
        else:
            folder=Path('data/processed/research')
            if not (folder/'documents.jsonl').exists():
                return dict(ticker=ticker,as_of=as_of,status='source_refresh_needed',reports=[],note='Refresh SEC data to download original reports for comparison.')
            filings=[json.loads(s) for s in (folder/'documents.jsonl').read_text().splitlines() if s.strip()]
            filings=[f for f in filings if f['ticker']==ticker]
            files={f['source_url']:dict(path=f['local_path'],sha256=f['sha256']) for f in filings}
        reports=[]
        for form in ['10-K','10-Q']:
            try:previous,current=select_filing_pair(company['cik'],filings,filing_type=form,as_of=as_of)
            except ValueError:continue
            texts=[]
            for filing in [previous,current]:
                source=files.get(filing.url)
                if source is None:raise ValueError('Original comparison source is unavailable.')
                raw=Path(source['path']).read_bytes()
                if sha256(raw).hexdigest()!=source['sha256']:raise ValueError('A comparison source changed after acquisition.')
                texts.append(raw.decode('utf-8-sig',errors='replace'))
            reports.append(compare_filings(previous,current,*texts))
        return dict(ticker=ticker,as_of=as_of,status='available' if reports else 'insufficient_comparable_filings',
                    reports=reports,note='Text changes are review signals, not validated materiality or investment recommendations.')
