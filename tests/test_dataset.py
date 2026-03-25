"""Tests for feast_mlflow.dataset."""

import pandas as pd

from feast_mlflow.dataset import FeastDataset, FeastDatasetSource


def test_feast_dataset_source():
    source = FeastDatasetSource(
        feature_service_name="fraud_svc",
        feast_project="demo",
        feature_views=["txn_features"],
        data_sources=["transactions.parquet"],
    )
    d = source.to_dict()
    assert d["source_type"] == "feast"
    assert d["feature_service"] == "fraud_svc"
    assert "transactions.parquet" in d["data_sources"]


def test_feast_dataset():
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "amount": [100.0, 200.0],
        "count": [1, 2],
    })
    source = FeastDatasetSource(
        feature_service_name="test_svc",
        feast_project="test",
        feature_views=["v1"],
        data_sources=["data.parquet"],
    )
    dataset = FeastDataset(df=df, source=source)

    assert dataset.name == "feast:test_svc"
    assert len(dataset.digest) == 8
    assert dataset.profile()["num_rows"] == 2
    assert "amount" in dataset.schema_dict()


def test_feast_dataset_digest_changes_with_data():
    source = FeastDatasetSource("svc", "proj", ["v"], ["d"])
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})

    d1 = FeastDataset(df=df1, source=source)
    d2 = FeastDataset(df=df2, source=source)

    assert d1.digest != d2.digest
