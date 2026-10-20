import numpy as np
import pandas as pd
from src.ml import hard_negative_model as module
from src.validation.sec_disclosure_data import TARGET


def test_mining_scores_are_generated_without_the_scored_companies(monkeypatch):
    frame=pd.DataFrame(0.,index=np.arange(60),columns=module.FEATURES)
    frame['cik']=['company_'+str(i//2) for i in range(60)]
    frame[TARGET]=np.arange(60)%2;frame['design_weight']=1.
    scored=[]
    class Spy:
        classes_=np.array([0,1])
        def fit(self,X,y,sample_weight):
            self.companies=set(frame.loc[X.index,'cik']);return self
        def predict_proba(self,X):
            assert not self.companies&set(frame.loc[X.index,'cik'])
            scored.extend(X.index.tolist());return np.tile([.2,.8],(len(X),1))
    monkeypatch.setattr(module,'classifier',Spy)
    _,audit=module.fit(frame,{'mining_seed':'fixed','mining_score_threshold':.5,'hard_negative_multiplier':3.})
    assert sorted(scored)==list(frame.index)
    assert audit['mined_negative_rows']==30
    assert audit['development_rows_used_for_mining'] is False
