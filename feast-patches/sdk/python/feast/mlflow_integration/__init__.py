"""
Feast MLflow integration: native experiment tracking for feature stores.

Activated via the ``mlflow:`` block in ``feature_store.yaml``.
"""

from feast.mlflow_integration.emitter import FeastMlflowEmitter

__all__ = ["FeastMlflowEmitter"]
