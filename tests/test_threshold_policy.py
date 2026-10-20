import numpy as np
import pytest
from src.validation.threshold_policy import held_out_policy,summarize_policy


def test_held_out_labels_cannot_change_own_alert_or_threshold():
    raw=np.array([.9,.7,.3,.1,.8,.6,.4,.2]);y=np.array([1,1,0,0,1,1,0,0]);folds=np.array([0,0,0,0,1,1,1,1]);companies=np.arange(8)
    a,d=held_out_policy(raw,y,np.ones(8),companies,folds)
    changed=y.copy();changed[folds==0]=1-changed[folds==0]
    b,e=held_out_policy(raw,changed,np.ones(8),companies,folds)
    assert d[0]['threshold']==e[0]['threshold']
    np.testing.assert_array_equal(a[folds==0],b[folds==0])


def test_no_feasible_fit_threshold_abstains_instead_of_calling_negative():
    a,d=held_out_policy([.1,.1,.1,.1],[1,0,1,0],np.ones(4),np.arange(4),[0,0,1,1],.9,.9)
    assert np.isnan(a).all() and all(x['abstained'] for x in d)
    result=summarize_policy([1,0,1,0],np.ones(4),a)
    assert result['abstained_companies']==4 and result['precision'] is None
    assert result['recall_among_covered'] is None and result['positive_weight_abstained']==2


def test_repeated_company_rows_are_rejected():
    with pytest.raises(ValueError,match='Distinct company'):
        held_out_policy([.9,.2],[1,0],[1,1],['a','a'],[0,1])
