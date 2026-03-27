"""Tests for Stage 2 Task 2.1: Feast tags helpers."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestSetFeastTags:
    """Task 2.1 tests."""

    def test_sets_tags_on_active_run(self):
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast_mlflow.feast_tags import set_feast_tags

            result = set_feast_tags(
                feature_service_name="fraud_service",
                project="fraud_detection",
                feature_refs=["fv:amount", "fv:count"],
            )

        assert result is True
        mock_mlflow.set_tag.assert_any_call("feast.feature_service", "fraud_service")
        mock_mlflow.set_tag.assert_any_call("feast.project", "fraud_detection")
        mock_mlflow.set_tag.assert_any_call("feast.feature_refs", "fv:amount,fv:count")

    def test_returns_false_when_no_active_run(self):
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast_mlflow.feast_tags import set_feast_tags

            result = set_feast_tags(feature_service_name="test")

        assert result is False

    def test_returns_false_when_mlflow_not_installed(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            from feast_mlflow.feast_tags import set_feast_tags
            result = set_feast_tags(feature_service_name="test")
            assert result is False


class TestSetFeastModelVersionTags:
    """Task 2.1 - model registry tags."""

    def test_sets_tags_on_model_version(self):
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast_mlflow.feast_tags import set_feast_model_version_tags

            set_feast_model_version_tags(
                model_name="fraud-model",
                version="1",
                feature_service_name="fraud_service",
                project="fraud_detection",
            )

        mock_client.set_model_version_tag.assert_any_call(
            "fraud-model", "1", "feast.feature_service", "fraud_service"
        )
        mock_client.set_model_version_tag.assert_any_call(
            "fraud-model", "1", "feast.project", "fraud_detection"
        )
