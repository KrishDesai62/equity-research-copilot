import io
import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from src.aws.s3_cache import S3Cache
from src.ml.evaluate import evaluate_classifier, walk_forward_splits
from src.ml.neural_network import train_neural_network
from src.ml.artifacts import save_model, load_model
from src.agents_full.risk_agent import RiskAgentTool


def test_metrics_threshold_probabilities():
    metrics = evaluate_classifier([0, 1, 0, 1], [.1, .8, .2, .9])
    assert metrics["f1"] == 1
    assert metrics["pr_auc"] == 1


def test_splits_preserve_original_positions():
    df = pd.DataFrame({"filing_date": ["2021-05-01", "2020-02-01", "2021-02-01"]})
    for train, test in walk_forward_splits(df, min_train_periods=1):
        assert pd.to_datetime(df.iloc[train].filing_date).max() < pd.to_datetime(df.iloc[test].filing_date).min()


def test_neural_network_artifact_and_risk(tmp_path):
    rng = np.random.default_rng(42)
    values = rng.normal(size=100)
    df = pd.DataFrame({"x": values, "label": (values > 0).astype(int)})
    model = train_neural_network(df, ["x"], "label", max_iter=1500)
    assert evaluate_classifier(df.label, model.predict_proba(df[["x"]])[:, 1])["pr_auc"] > .9
    path = tmp_path / "distress.joblib"
    save_model(model, ["x"], path, trained_through="2020-01-01", target="label")
    assert load_model(path)["feature_cols"] == ["x"]
    features = pd.DataFrame({"ticker": ["A", "A"], "filing_date": ["2020-02-01", "2030-01-01"], "x": [1., -1.]})
    agent = RiskAgentTool(tmp_path, require_release=False, max_feature_age_days=400)
    risk = agent.get_risk_profile("A", "2021-01-01", features_df=features)
    assert risk["distress_probability"] > .5
    assert risk["manipulation_flag"] is None
    with pytest.raises(ValueError, match="trained after"):
        agent.get_risk_profile("A", "2019-01-01", features_df=features.iloc[[0]].assign(filing_date="2018-01-01"))


def test_s3_roundtrip_and_missing_key(tmp_path):
    class Client:
        objects = {}
        def put_object(self, **kwargs):
            assert kwargs["ServerSideEncryption"] == "AES256"
            self.objects[kwargs["Key"]] = kwargs["Body"].read()
        def get_object(self, **kwargs):
            if kwargs["Key"] not in self.objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}
    cache = S3Cache("test", client=Client())
    source = tmp_path / "source"
    source.write_bytes(b"filing")
    cache.upload("filings/a", source)
    target = tmp_path / "copy"
    assert cache.restore("filings/a", target)
    assert target.read_bytes() == b"filing"
    assert not cache.restore("missing", target)
    with pytest.raises(ValueError):
        cache.key("../escape")


def test_walk_forward_purges_future_labels_and_company_overlap():
    from src.ml.train_distress import run_walk_forward_evaluation
    frame = pd.DataFrame({
        "cik": ["train0", "train1", "future", "overlap", "overlap", "test1"],
        "filing_date": ["2020-01-01"] * 4 + ["2020-04-01"] * 2,
        "available": ["2020-02-01", "2020-02-01", "2030-01-01", "2020-02-01", "2020-05-01", "2020-05-01"],
        "x": [0, 1, 1, 0, 0, 1], "y": [0, 1, 1, 0, 0, 1]})
    def trainer(train, features, target):
        assert set(train.cik) == {"train0", "train1"}
        class Model:
            def predict_proba(self, X):
                return np.column_stack([1-X.x, X.x])
        return Model()
    report = run_walk_forward_evaluation(frame, ["x"], "y", trainer=trainer,
                                         label_available_col="available", min_train_periods=1, test_companies=["overlap", "test1"])
    assert report["mean_pr_auc"] == 1


def test_s3_access_denied_is_not_cache_miss(tmp_path):
    class Client:
        def get_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    with pytest.raises(ClientError):
        S3Cache("bucket", client=Client()).restore("a", tmp_path / "a")


def test_filing_restored_without_sec_request(tmp_path, monkeypatch):
    from src.data_layer.edgar_ingest import download_filing
    class Cache:
        def restore(self, key, path):
            assert key == "filings/0000000001/0001-20-001/report.htm"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("cached filing")
            return True
    monkeypatch.setattr("src.data_layer.edgar_ingest.requests.get", lambda *a, **kw: pytest.fail("Unexpected SEC request"))
    assert download_filing("1", "0001-20-001", "report.htm", cache=Cache(), dest_dir=tmp_path) == "cached filing"


def test_beneish_same_year_ratios():
    from src.ml.features_manipulation import compute_beneish_components
    row = dict(Revenues=100, Assets=200, CostOfGoodsAndServicesSold=60,
               AccountsReceivableNetCurrent=10, AssetsCurrent=50,
               PropertyPlantAndEquipmentNet=100, Depreciation=10,
               SellingGeneralAndAdministrativeExpense=10, LiabilitiesCurrent=30,
               LongTermDebt=20, NetIncomeLoss=10, NetCashProvidedByUsedInOperatingActivities=10)
    result = compute_beneish_components(row, row)
    assert result["dsri"] == 1
    assert result["tata"] == 0
    assert np.isfinite(result["beneish_m_score"])


def test_xbrl_current_consolidated_usd_facts(tmp_path):
    import zipfile
    from src.data_layer.xbrl_bulk import parse_zip_to_dataframe
    path = tmp_path / "quarter.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sub.txt", "adsh\tcik\tname\tfiled\tperiod\tfy\tfp\tform\na\t1\tA\t20200501\t20200331\t2020\tQ1\t10-Q\n")
        archive.writestr("num.txt", "adsh\ttag\tvalue\tddate\tqtrs\tuom\tcoreg\n"
                          "a\tAssets\t100\t20200331\t0\tUSD\t\n"
                          "a\tAssets\t50\t20190331\t0\tUSD\t\n"
                          "a\tAssets\t70\t20200331\t0\tEUR\t\n"
                          "a\tAssets\t10\t20200331\t0\tUSD\tsubsidiary\n")
    frame = parse_zip_to_dataframe(path)
    assert frame.value.tolist() == [100]
    assert frame.filing_date.iloc[0] == pd.Timestamp("2020-05-01")


def test_pr_auc_preserves_probability_ranking():
    from sklearn.metrics import average_precision_score
    y, probabilities = [0, 1, 0, 1], [.1, .4, .3, .45]
    assert evaluate_classifier(y, probabilities)["pr_auc"] == average_precision_score(y, probabilities)
