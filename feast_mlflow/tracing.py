"""
Stage 2 Task 2.3: Tracing wrapper for Feast feature retrieval.

When using MLflow Tracing (e.g. for GenAI or serving observability), wrapping
Feast calls in spans makes feature retrieval visible in the trace with context
like feature service name and duration.

Usage::

    from feast import FeatureStore
    from feast_mlflow.tracing import traced_get_online_features

    store = FeatureStore(repo_path="feature_repo")
    response = traced_get_online_features(
        store=store,
        feature_service_name="fraud_feature_service",
        entity_rows=[{"user_id": "user_001"}],
    )

Or as a context manager::

    from feast_mlflow.tracing import feast_span

    with feast_span("get_online_features", feature_service="fraud_feature_service") as span:
        response = store.get_online_features(features=fs, entity_rows=[...])
        span.set_attribute("feast.entity_count", len(entity_rows))
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from feast import FeatureStore

logger = logging.getLogger("feast_mlflow.tracing")


def traced_get_online_features(
    store: "FeatureStore",
    feature_service_name: str,
    entity_rows: List[Dict[str, Any]],
    full_feature_names: bool = False,
) -> Any:
    """Fetch online features from Feast with an MLflow trace span.

    Creates an OpenTelemetry-compatible span (via MLflow Tracing) around
    the ``get_online_features`` call, recording feature service name,
    entity count, and retrieval duration.

    If MLflow Tracing is not active, falls through to a normal Feast call.

    Args:
        store: Feast FeatureStore instance.
        feature_service_name: Name of the feature service.
        entity_rows: List of entity key dicts.
        full_feature_names: Whether to use full feature names.

    Returns:
        Feast OnlineResponse.
    """
    feature_service = store.get_feature_service(feature_service_name)

    span = None
    try:
        import mlflow
        span = mlflow.start_span(name="feast.get_online_features")
        if span:
            span.set_attribute("feast.feature_service", feature_service_name)
            span.set_attribute("feast.entity_count", len(entity_rows))
    except Exception:
        pass

    start = time.time()
    try:
        response = store.get_online_features(
            features=feature_service,
            entity_rows=entity_rows,
            full_feature_names=full_feature_names,
        )
        duration = time.time() - start

        if span:
            try:
                span.set_attribute("feast.duration_ms", round(duration * 1000, 1))
                span.set_attribute("feast.status", "success")
                span.end()
            except Exception:
                pass

        return response
    except Exception as e:
        if span:
            try:
                span.set_attribute("feast.status", "error")
                span.set_attribute("feast.error", str(e))
                span.end()
            except Exception:
                pass
        raise


@contextmanager
def feast_span(operation: str, feature_service: str = "", **attributes: Any):
    """Context manager that creates an MLflow trace span for a Feast operation.

    Usage::

        with feast_span("get_online_features", feature_service="my_service") as span:
            response = store.get_online_features(features=fs, entity_rows=[...])
            span.set_attribute("feast.entity_count", 100)

    If MLflow Tracing is not available, yields a no-op object.
    """
    span = None
    try:
        import mlflow
        span = mlflow.start_span(name=f"feast.{operation}")
        if span and feature_service:
            span.set_attribute("feast.feature_service", feature_service)
        for k, v in attributes.items():
            if span:
                span.set_attribute(f"feast.{k}", v)
    except Exception:
        pass

    class NoOpSpan:
        def set_attribute(self, key: str, value: Any) -> None:
            pass
        def end(self) -> None:
            pass

    try:
        yield span if span else NoOpSpan()
    finally:
        if span:
            try:
                span.end()
            except Exception:
                pass
