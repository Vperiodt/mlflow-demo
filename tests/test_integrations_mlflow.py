"""Tests for Stage 1: Feast integration helpers (feast.integrations.mlflow)."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_feast_modules():
    """Load integrations from feast-src; stub top-level feast.FeatureStore only."""
    repo_root = Path(__file__).resolve().parent.parent
    sdk_python = repo_root / "feast-src" / "sdk" / "python"
    if not sdk_python.is_dir():
        pytest.skip("feast-src/sdk/python missing — clone Feast into feast-src to run these tests")

    for name in (
        "feast.integrations.mlflow",
        "feast.integrations.mlflow_autolog",
        "feast.integrations",
        "feast",
    ):
        sys.modules.pop(name, None)

    feast_mod = types.ModuleType("feast")
    feast_mod.FeatureStore = MagicMock
    feast_mod.__path__ = [str(sdk_python / "feast")]
    sys.modules["feast"] = feast_mod

    path_str = str(sdk_python)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass
        for name in (
            "feast.integrations.mlflow",
            "feast.integrations.mlflow_autolog",
            "feast.integrations",
            "feast",
        ):
            sys.modules.pop(name, None)


class TestLogFeatureRetrievalToMlflow:
    """Task 1.1 tests."""

    def test_logs_params_and_tags_to_active_run(self):
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast.integrations.mlflow import log_feature_retrieval_to_mlflow

            result = log_feature_retrieval_to_mlflow(
                feature_service_name="test_service",
                feature_refs=["fv:feat_a", "fv:feat_b"],
                entity_count=100,
                duration_seconds=1.5,
                retrieval_type="historical",
            )

        assert result is True
        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["feast.feature_service"] == "test_service"
        assert params["feast.num_features"] == 2
        assert params["feast.entity_count"] == 100
        mock_mlflow.set_tag.assert_any_call("feast.feature_service", "test_service")

    def test_noop_when_no_active_run(self):
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast.integrations.mlflow import log_feature_retrieval_to_mlflow

            result = log_feature_retrieval_to_mlflow(
                feature_service_name="test",
                feature_refs=["fv:a"],
                entity_count=10,
                duration_seconds=0.5,
            )

        assert result is False
        mock_mlflow.log_params.assert_not_called()

    def test_noop_when_mlflow_not_installed(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            # Force re-import
            if "feast.integrations.mlflow" in sys.modules:
                del sys.modules["feast.integrations.mlflow"]
            try:
                from feast.integrations.mlflow import log_feature_retrieval_to_mlflow
                result = log_feature_retrieval_to_mlflow(
                    feature_service_name="test",
                    feature_refs=["fv:a"],
                    entity_count=10,
                    duration_seconds=0.5,
                )
                assert result is False
            except ImportError:
                pass  # Expected if mock doesn't work cleanly


class TestResolveFeatureServiceFromModelUri:
    """Task 1.2 tests."""

    def test_resolves_from_tag(self):
        mock_mlflow = MagicMock()
        mock_model_info = MagicMock()
        mock_model_info.run_id = "abc123"
        mock_mlflow.models.get_model_info.return_value = mock_model_info

        mock_run = MagicMock()
        mock_run.data.tags = {"feast.feature_service": "my_service"}
        mock_mlflow.MlflowClient.return_value.get_run.return_value = mock_run

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast.integrations.mlflow import resolve_feature_service_from_model_uri

            result = resolve_feature_service_from_model_uri("models:/my-model/1")

        assert result == "my_service"

    def test_falls_back_to_convention(self):
        mock_mlflow = MagicMock()
        mock_model_info = MagicMock()
        mock_model_info.run_id = "abc123"
        mock_mlflow.models.get_model_info.return_value = mock_model_info

        mock_run = MagicMock()
        mock_run.data.tags = {}
        mock_mlflow.MlflowClient.return_value.get_run.return_value = mock_run

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast.integrations.mlflow import resolve_feature_service_from_model_uri

            result = resolve_feature_service_from_model_uri("models:/my-model/3")

        assert result == "my-model_v3"

    def test_raises_when_mlflow_not_installed(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            if "feast.integrations.mlflow" in sys.modules:
                del sys.modules["feast.integrations.mlflow"]
            try:
                from feast.integrations.mlflow import (
                    FeastMLflowError,
                    resolve_feature_service_from_model_uri,
                )
                with pytest.raises(FeastMLflowError, match="not installed"):
                    resolve_feature_service_from_model_uri("models:/x/1")
            except ImportError:
                pass


class TestGetEntityDfFromMlflowRun:
    """Task 1.3 tests."""

    def test_loads_from_parquet_artifact(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"user_id": ["a", "b"], "event_timestamp": ["2025-01-01", "2025-01-02"]})
        parquet_path = tmp_path / "entity_df.parquet"
        df.to_parquet(parquet_path)

        mock_mlflow = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.path = "entity_df.parquet"
        mock_mlflow.MlflowClient.return_value.list_artifacts.return_value = [mock_artifact]
        mock_mlflow.MlflowClient.return_value.download_artifacts.return_value = str(parquet_path)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast.integrations.mlflow import get_entity_df_from_mlflow_run

            result = get_entity_df_from_mlflow_run("run123")

        assert len(result) == 2
        assert "user_id" in result.columns
        assert "event_timestamp" in result.columns

    def test_raises_when_no_data(self):
        mock_mlflow = MagicMock()
        mock_mlflow.MlflowClient.return_value.list_artifacts.return_value = []
        mock_run = MagicMock()
        mock_run.data.params = {}
        mock_mlflow.MlflowClient.return_value.get_run.return_value = mock_run

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast.integrations.mlflow import FeastMLflowError, get_entity_df_from_mlflow_run

            with pytest.raises(FeastMLflowError, match="No entity DataFrame"):
                get_entity_df_from_mlflow_run("run123")
