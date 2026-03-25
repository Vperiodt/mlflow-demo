"""Tests for feast_mlflow.contract."""

import json
import tempfile
from pathlib import Path

from feast_mlflow.contract import (
    CONTRACT_VERSION,
    FeatureContract,
    FeatureSpec,
    ValidationResult,
)


def test_contract_roundtrip():
    contract = FeatureContract(
        feature_service="test_service",
        feast_project="test_project",
        created_at="2026-01-01T00:00:00Z",
        mlflow_run_id="abc123",
        features=[
            FeatureSpec(name="amount", dtype="Float64", source_view="txn_fv"),
            FeatureSpec(name="count", dtype="Int64", source_view="txn_fv"),
        ],
        entity_keys=["user_id"],
        data_sources=["transactions.parquet"],
        statistics={"row_count": 100},
    )

    json_str = contract.to_json()
    restored = FeatureContract.from_json(json_str)

    assert restored.feature_service == "test_service"
    assert restored.feast_project == "test_project"
    assert len(restored.features) == 2
    assert restored.features[0].name == "amount"
    assert restored.entity_keys == ["user_id"]
    assert restored.statistics["row_count"] == 100


def test_contract_save_load():
    contract = FeatureContract(
        feature_service="svc",
        features=[FeatureSpec(name="f1", dtype="Float64", source_view="v1")],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = contract.save(tmpdir)
        assert path.exists()
        loaded = FeatureContract.from_file(path)
        assert loaded.feature_service == "svc"
        assert len(loaded.features) == 1


def test_validation_result():
    r = ValidationResult(
        feature_name="amount",
        dtype_expected="Float64",
        dtype_actual="Float64",
        online_available=True,
        schema_match=True,
    )
    assert r.valid is True

    r2 = ValidationResult(
        feature_name="count",
        dtype_expected="Int64",
        dtype_actual="Float64",
        online_available=True,
        schema_match=False,
    )
    assert r2.valid is False
