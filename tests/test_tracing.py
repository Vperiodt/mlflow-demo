"""Tests for Stage 2 Task 2.3: Tracing wrapper."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestFeastSpan:
    """Task 2.3 tests."""

    def test_context_manager_creates_span(self):
        mock_mlflow = MagicMock()
        mock_span = MagicMock()
        mock_mlflow.start_span.return_value = mock_span

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from feast_mlflow.tracing import feast_span

            with feast_span("get_online_features", feature_service="my_service") as span:
                span.set_attribute("feast.entity_count", 10)

        mock_mlflow.start_span.assert_called_once_with(name="feast.get_online_features")
        mock_span.set_attribute.assert_any_call("feast.feature_service", "my_service")
        mock_span.end.assert_called()

    def test_context_manager_noop_without_mlflow(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            from feast_mlflow.tracing import feast_span

            with feast_span("test_op") as span:
                span.set_attribute("key", "value")
            # Should not raise
