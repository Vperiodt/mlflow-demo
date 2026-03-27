"""
Stage 2 Task 2.1: Feast context helpers for MLflow.

Convention: when logging a model that uses Feast features, set these tags
on the MLflow run or model version so that serving can resolve the
feature service automatically.

Tags:
    feast.feature_service     — Name of the Feast feature service
    feast.project             — Feast project name (optional)

Usage::

    import mlflow
    from feast_mlflow.feast_tags import set_feast_tags, log_model_with_feast_context

    with mlflow.start_run():
        # Option A: Set tags explicitly
        set_feast_tags(feature_service_name="fraud_feature_service", project="fraud_detection")
        mlflow.pytorch.log_model(model, "model")

        # Option B: Log model + set tags in one call
        log_model_with_feast_context(
            model, "model",
            feature_service_name="fraud_feature_service",
            flavor="pytorch",
        )
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger("feast_mlflow.feast_tags")


def set_feast_tags(
    feature_service_name: str,
    project: str = "",
    feature_refs: Optional[List[str]] = None,
    feature_store_identifier: str = "",
) -> bool:
    """Set Feast convention tags on the active MLflow run.

    Call this during training so that downstream consumers (serving, evaluation,
    lineage) can resolve which feature service the model depends on.

    Args:
        feature_service_name: Feast feature service name.
        project: Feast project name (optional).
        feature_refs: List of feature references (optional).
        feature_store_identifier: Identifier for the feature store instance (optional).

    Returns:
        True if tags were set, False if no active run or MLflow not installed.
    """
    try:
        import mlflow
    except ImportError:
        return False

    if mlflow.active_run() is None:
        return False

    try:
        mlflow.set_tag("feast.feature_service", feature_service_name)
        if project:
            mlflow.set_tag("feast.project", project)
        if feature_refs:
            mlflow.set_tag("feast.feature_refs", ",".join(feature_refs))
        if feature_store_identifier:
            mlflow.set_tag("feast.feature_store_identifier", feature_store_identifier)
        return True
    except Exception as e:
        logger.warning("Failed to set Feast tags: %s", e)
        return False


def log_model_with_feast_context(
    model: Any,
    artifact_path: str,
    feature_service_name: str,
    flavor: str = "pytorch",
    project: str = "",
    feature_refs: Optional[List[str]] = None,
    **log_model_kwargs: Any,
) -> Any:
    """Log a model to MLflow and set Feast convention tags.

    Convenience wrapper that calls ``mlflow.<flavor>.log_model()`` and then
    sets the ``feast.feature_service`` tag on the active run.

    Args:
        model: The model object to log.
        artifact_path: Artifact path within the run.
        feature_service_name: Feast feature service name.
        flavor: MLflow model flavor (default: "pytorch").
        project: Feast project name (optional).
        feature_refs: Feature references (optional).
        **log_model_kwargs: Additional kwargs for ``mlflow.<flavor>.log_model()``.

    Returns:
        The result of ``mlflow.<flavor>.log_model()``.
    """
    import mlflow

    flavor_mod = getattr(mlflow, flavor)
    result = flavor_mod.log_model(model, artifact_path, **log_model_kwargs)

    set_feast_tags(
        feature_service_name=feature_service_name,
        project=project,
        feature_refs=feature_refs,
    )

    return result


def set_feast_model_version_tags(
    model_name: str,
    version: str,
    feature_service_name: str,
    project: str = "",
    tracking_uri: Optional[str] = None,
) -> None:
    """Set Feast tags on a registered model version in MLflow Model Registry.

    Use after promoting a model to a stage/alias so that serving can
    resolve the feature service from the model URI.

    Args:
        model_name: Registered model name.
        version: Model version number (as string).
        feature_service_name: Feast feature service name.
        project: Feast project name (optional).
        tracking_uri: MLflow tracking URI (optional).

    Example::

        set_feast_model_version_tags("fraud-model", "1", "fraud_feature_service")
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    client.set_model_version_tag(model_name, version, "feast.feature_service", feature_service_name)
    if project:
        client.set_model_version_tag(model_name, version, "feast.project", project)
