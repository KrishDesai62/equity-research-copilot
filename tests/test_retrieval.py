from src.rag.retrieval import retrieve


class Store:
    def query(self, vector, k, where):
        assert where == {"ticker": "A"}
        return dict(documents=[["risk text"]], metadatas=[[dict(ticker="A", filing_type="10-K",
                    section_title="Item 1A", filing_date="2020-01-01")]], distances=[[0.2]])


def result(monkeypatch):
    monkeypatch.setattr("src.rag.embeddings.embed_texts", lambda *args: [[1.0]])
    return retrieve("risk", ticker="A", store=Store())[0]


def test_retrieve_filters_by_ticker(monkeypatch):
    assert result(monkeypatch)["ticker"] == "A"


def test_retrieve_returns_expected_metadata_shape(monkeypatch):
    assert {"text", "ticker", "filing_type", "section", "filing_date", "similarity_score"} <= result(monkeypatch).keys()
