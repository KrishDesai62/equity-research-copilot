import numpy as np
import pytest

from src.validation.calibration_refinement import nested_scores,select_regularization

CONFIG={'inner_folds':2,'regularization_candidates':[.01,.1,1.], 'fallback_regularization':.1}


def test_outer_held_labels_cannot_choose_their_own_calibration():
    raw=np.linspace(.01,.9,24)
    y=np.array([0,1,0,1]*6);w=np.ones(24)
    companies=np.array([str(i).zfill(10) for i in range(24)])
    folds=np.repeat([0,1,2],8)
    first,details=nested_scores(raw,y,w,companies,folds,'test',CONFIG,'nested')
    changed=y.copy();changed[folds==0]=1-changed[folds==0]
    second,new_details=nested_scores(raw,changed,w,companies,folds,'test',CONFIG,'nested')
    np.testing.assert_array_equal(first[folds==0],second[folds==0])
    assert details[0]==new_details[0]


def test_inner_insufficient_support_uses_declared_fallback():
    c,fallback=select_regularization(np.array([.1,.2]),np.array([0,1]),np.ones(2),np.array(['a','b']),'test',CONFIG)
    assert fallback and c==.1


def test_duplicate_company_units_are_rejected():
    with pytest.raises(ValueError,match='one calibration prediction per company'):
        nested_scores(np.array([.1,.2]),np.array([0,1]),np.ones(2),np.array(['same','same']),np.array([0,1]),'test',CONFIG,'nested')
