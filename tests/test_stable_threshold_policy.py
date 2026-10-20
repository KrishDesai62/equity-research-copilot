import numpy as np
import pytest
from src.validation.stable_threshold_policy import consensus_threshold,crossfit_consensus

CONFIG={'bootstrap_replicates':40,'seed':7,'threshold_quantile':.75,'minimum_feasible_fraction':.8,'minimum_precision':.5,'minimum_recall':.5}


def test_held_labels_do_not_influence_consensus_policy():
    raw=np.tile([.9,.8,.2,.1],5);y=np.tile([1,1,0,0],5);folds=np.arange(20)%3;companies=np.arange(20)
    a,d=crossfit_consensus(raw,y,np.ones(20),companies,folds,CONFIG)
    changed=y.copy();changed[folds==0]=1-changed[folds==0]
    b,e=crossfit_consensus(raw,changed,np.ones(20),companies,folds,CONFIG)
    assert d[0]==e[0]
    np.testing.assert_array_equal(a[folds==0],b[folds==0])


def test_infeasible_bootstraps_abstain_reproducibly():
    a=consensus_threshold([.1]*20,[0]*20,np.ones(20),CONFIG)
    assert a==consensus_threshold([.1]*20,[0]*20,np.ones(20),CONFIG)
    assert a[0] is None and a[1]['feasible_bootstrap_fraction']==0


def test_consensus_rejects_nonfinite_inputs():
    with pytest.raises(ValueError,match='Invalid raw scores'):
        consensus_threshold([np.inf,.1],[1,0],[1,1],CONFIG)
