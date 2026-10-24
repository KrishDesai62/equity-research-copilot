from src.web.architecture import describe


def test_architecture_separates_research_from_serving(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY',raising=False);monkeypatch.delenv('FINNHUB_API_KEY',raising=False)
    class Research:
        def model_status(self):return {'production_approval':False,'release_status':'not_approved'}
    result=describe(Research())
    assert result['layers']['model']['status']=='gated'
    assert result['layers']['neural']['status']=='research_only'
    assert result['layers']['chroma']['status']=='research_only'
    assert result['ai_synthesis']=='not_configured'
    assert result['market_data']=='not_configured'


def test_architecture_does_not_turn_model_failure_into_release():
    class Research:
        def model_status(self):raise ValueError('Private internal path')
    result=describe(Research())
    assert result['layers']['model']['status']=='unavailable'
    assert 'Private internal path' not in str(result)
