from src.validation.local_reliability_status import policy_passes


def test_high_precision_on_abstaining_subset_is_not_full_coverage_success():
    assert not policy_passes([{'all_companies_covered':False,'precision':1.,'recall_among_covered':.9}])


def test_one_failed_partition_prevents_stability_claim():
    good={'all_companies_covered':True,'precision':.65,'recall_among_covered':.55}
    assert policy_passes([good]*5)
    assert not policy_passes([good]*4+[{**good,'precision':.26}])
    assert not policy_passes([])
