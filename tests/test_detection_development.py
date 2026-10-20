import numpy as np
import pandas as pd
import pytest

from src.ml.detection_models import ProbabilityVote,loss_weights
from src.validation.sec_disclosure_data import TARGET


def training():
    return pd.DataFrame({'cik':['a','a','b','c'],TARGET:[0,0,0,1],
                         'design_weight':[2.,2.,1.,1.],
                         'prediction_date':['2014-12-31','2020-12-31','2020-12-31','2020-12-31']})


def test_alternative_loss_weights_do_not_change_sampling_weights():
    frame=training();before=frame.copy(deep=True);y=frame[TARGET].to_numpy()
    balanced=loss_weights(frame,'balanced')
    assert balanced[y==0].sum()==pytest.approx(balanced[y==1].sum())
    company=loss_weights(frame,'company_balanced')
    assert company[0]==pytest.approx(company[2])
    recent=loss_weights(frame,'recent_balanced')
    assert recent[1]/recent[0]==pytest.approx(4.)
    pd.testing.assert_frame_equal(frame,before)


class FixedMember:
    def __init__(self,classes,probabilities):self.classes_=classes;self.probabilities=probabilities
    def predict_proba(self,X):return np.tile(self.probabilities,(len(X),1))


def test_ensemble_aligns_positive_class_before_averaging():
    a=FixedMember([0,1],[.8,.2]);b=FixedMember([1,0],[.8,.2])
    predictions=ProbabilityVote([a,b]).predict_proba(pd.DataFrame({'x':[1.,2.]}))
    np.testing.assert_allclose(predictions,np.full((2,2),.5))


def test_ensemble_rejects_invalid_member_instead_of_clipping_scores():
    with pytest.raises(ValueError,match='Invalid ensemble'):
        ProbabilityVote([FixedMember([0,1],[.1,.1])]).predict_proba([1,2])
    with pytest.raises(ValueError):ProbabilityVote([])


def test_unknown_loss_policy_and_nonpositive_sampling_weights_are_rejected():
    with pytest.raises(ValueError):loss_weights(training(),'anything')
    frame=training();frame.loc[0,'design_weight']=0
    with pytest.raises(ValueError):loss_weights(frame,'design')


def test_specialist_selection_prioritizes_detection_without_approving_release():
    from src.validation.detection_research_candidate import choose_specialist
    def candidate(name,recall,pr):
        return {'name':name,'converged':True,'threshold':None,
                'operating_limits':{'maximum_recall_at_precision_50pct':recall},
                'metrics':{'average_precision':pr,'log_loss':.06}}
    frozen=candidate('frozen_current_reference',.9,.9)
    a=candidate('native_sqrt',.2,.3)
    b=candidate('native_balanced',.3,.2)
    chosen=choose_specialist([frozen,a,b])
    assert chosen['name']=='native_balanced' and chosen['threshold'] is None
