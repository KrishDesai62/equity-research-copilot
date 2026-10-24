import pytest
from src.web.server import validate_question


def test_question_contract_preserves_cutoff_and_limits_results():
    assert validate_question({'question':'  liquidity  ','ticker':'aapl','as_of':'2025-01-01'}) == {
        'question':'liquidity','ticker':'AAPL','as_of':'2025-01-01','k':3}


@pytest.mark.parametrize('payload', [None, {}, {'question':'x'*501,'ticker':'AAPL','as_of':'2025-01-01'},
    {'question':'liquidity','ticker':'AAPL','as_of':'not-a-date'},
    {'question':'liquidity','ticker':42,'as_of':'2025-01-01'}])
def test_malformed_question_is_rejected(payload):
    with pytest.raises(ValueError):
        validate_question(payload)


def test_cutoff_dates_are_canonical_and_future_dates_are_rejected():
    result=validate_question({'question':'liquidity','ticker':'AAPL','as_of':'20250101'})
    assert result['as_of']=='2025-01-01'
    with pytest.raises(ValueError):
        validate_question({'question':'liquidity','ticker':'AAPL','as_of':'2099-01-01'})
