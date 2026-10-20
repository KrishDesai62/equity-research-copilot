import pandas as pd
import yaml
from src.data_layer.point_in_time_universe import universe_as_of, write_mvp_universe


def test_universe_includes_historically_delisted_companies(tmp_path):
    path = tmp_path / "universe.yaml"
    write_mvp_universe(["AAPL", "LEH"], path)
    assert "LEH" in yaml.safe_load(path.read_text())["tickers"]


def test_universe_as_of_date_excludes_future_information():
    history = pd.DataFrame([dict(ticker="A", added_date="2000-01-01", removed_date="2010-01-01"),
                            dict(ticker="A", added_date="2015-01-01", removed_date=None),
                            dict(ticker="B", added_date="2020-01-01", removed_date=None)])
    assert universe_as_of("2012-01-01", history) == []
    assert universe_as_of("2016-01-01", history) == ["A"]
