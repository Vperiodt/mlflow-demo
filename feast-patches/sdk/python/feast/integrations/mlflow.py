"""
Feast MLflow integration helpers.

Optional helpers that link Feast feature retrieval to MLflow experiment tracking.
All functions are safe to call even if MLflow is not installed — they will no-op
or raise clear exceptions.

These helpers do NOT modify Feast or MLflow behavior automatically. They are
explicit functions that users call in their training/serving code.

Usage::

    from feast import FeatureStore
    from feast.integrations.mlflow import (
        log_feature_retrieval_to_mlflow,
        resolve_feature_service_from_model_uri,
        get_entity_df_from_mlflow_run,
    )

    store = FeatureStore(repo_path="feature_repo")

    # Task 1.1: Log feature retrieval metadata to the active MLflow run
    import time
    start = time.time()
    training_df = store.get_historical_features(entity_df, features=feature_service).to_df()
    duration = time.time() - start
    log_feature_retrieval_to_mlflow(
        feature_service_name="fraud_feature_service",
        feature_refs=["txn_features:amount", "txn_features:count_24h"],
        entity_count=len(entity_df),
        duration_seconds=duration,
        retrieval_type="historical",
    )

    # Task 1.2: Resolve feature service from an MLflow model URI
    fs_name = resolve_feature_service_from_model_uri("models:/fraud-model/Production", store=store)
    feature_service = store.get_feature_service(fs_name)

    # Task 1.3: Build entity DataFrame from a previous MLflow run
    entity_df = get_entity_df_from_mlflow_run("abc123def456")
    training_df = store.get_historical_features(entity_df, features=feature_service).to_df()
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import pandas as pd
    from feast import FeatureStore

logger = logging.getLogger("feast.integrations.mlflow")


# ---------------------------------------------------------------------------
# Task 1.1: Log feature retrieval metadata to the active MLflow run
# ---------------------------------------------------------------------------

def log_feature_retrieval_to_mlflow(
    feature_service_name: str,
    feature_refs: List[str],
    entity_count: int,
    duration_seconds: float,
    retrieval_type: str = "historical",
) -> bool:
    """Log feature retrieval metadata to the currently active MLflow run.

    If MLflow is not installed or there is no active run, this is a no-op and
    returns False. It never raises and never has side effects outside of MLflow.

    Args:
        feature_service_name: Name of the Feast feature service used.
        feature_refs: List of feature references (e.g. ``["fv:feature_a", "fv:feature_b"]``).
        entity_count: Number of entity rows in the retrieval request.
        duration_seconds: Wall-clock time for the retrieval in seconds.
        retrieval_type: ``"historical"`` or ``"online"``.

    Returns:
        True if metadata was logged successfully, False otherwise.

    Example::

        import time
        start = time.time()
        df = store.get_historical_features(entity_df, features=fs).to_df()
        log_feature_retrieval_to_mlflow(
            feature_service_name="fraud_feature_service",
            feature_refs=["txn:amount", "txn:count_24h"],
            entity_count=len(entity_df),
            duration_seconds=time.time() - start,
        )
    """
    try:
        import mlflow
    except ImportError:
        logger.debug("mlflow not installed; skipping log_feature_retrieval_to_mlflow")
        return False

    active_run = mlflow.active_run()
    if active_run is None:
        logger.debug("No active MLflow run; skipping log_feature_retrieval_to_mlflow")
        return False

    try:
        mlflow.log_params({
            "feast.feature_service": feature_service_name,
            "feast.num_features": len(feature_refs),
            "feast.entity_count": entity_count,
            "feast.retrieval_type": retrieval_type,
        })
        mlflow.log_metric("feast.retrieval_duration_sec", round(duration_seconds, 3))

        mlflow.set_tag("feast.feature_service", feature_service_name)
        mlflow.set_tag("feast.feature_refs", ",".join(feature_refs))
        mlflow.set_tag("feast.retrieval_type", retrieval_type)

        return True
    except Exception as e:
        logger.warning("Failed to log feature retrieval to MLflow: %s", e)
        return False


# ---------------------------------------------------------------------------
# Task 1.2: Resolve feature service name from an MLflow model URI
# ---------------------------------------------------------------------------

class FeastMLflowError(Exception):
    """Raised when Feast-MLflow integration encounters an unrecoverable error."""


def resolve_feature_service_from_model_uri(
    model_uri: str,
    store: Optional["FeatureStore"] = None,
    tracking_uri: Optional[str] = None,
) -> str:
    """Resolve the Feast feature service name linked to an MLflow model.

    Resolution order:
      1. If the model/run has tag ``feast.feature_service``, return that value.
      2. Otherwise, derive a default: ``{model_name}_v{version}``.

    If ``store`` is provided, optionally validates that the feature service
    exists in the Feast registry.

    Args:
        model_uri: MLflow model URI (e.g. ``"models:/my-model/Production"``
            or ``"models:/my-model/1"``).
        store: Optional FeatureStore instance for validation.
        tracking_uri: Optional MLflow tracking URI. If not provided, uses
            the currently configured URI.

    Returns:
        Feature service name string.

    Raises:
        FeastMLflowError: If MLflow is not installed, the model_uri is
            invalid, or validation fails.

    Example::

        fs_name = resolve_feature_service_from_model_uri(
            "models:/fraud-model/Production",
            store=FeatureStore(repo_path="feature_repo"),
        )
    """
    try:
        import mlflow
    except ImportError:
        raise FeastMLflowError(
            "mlflow is not installed. Install it with: pip install mlflow"
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()

    try:
        model_info = mlflow.models.get_model_info(model_uri)
        run_id = model_info.run_id
    except Exception as e:
        raise FeastMLflowError(f"Could not resolve model URI '{model_uri}': {e}")

    # Check the run's tags for feast.feature_service
    try:
        run = client.get_run(run_id)
        tags = run.data.tags
        fs_name = tags.get("feast.feature_service")
        if fs_name:
            logger.info("Resolved feature service '%s' from tag on run %s", fs_name, run_id[:8])
        else:
            # Derive from model name + version
            model_name = _extract_model_name(model_uri)
            model_version = _extract_model_version(model_uri, client)
            fs_name = f"{model_name}_v{model_version}"
            logger.info(
                "No feast.feature_service tag; using convention: '%s'", fs_name
            )
    except FeastMLflowError:
        raise
    except Exception as e:
        raise FeastMLflowError(f"Could not read run {run_id}: {e}")

    # Optional validation against Feast registry
    if store is not None:
        try:
            store.get_feature_service(fs_name)
            logger.info("Feature service '%s' validated in Feast registry", fs_name)
        except Exception:
            raise FeastMLflowError(
                f"Feature service '{fs_name}' not found in Feast registry. "
                f"Ensure it exists or set tag 'feast.feature_service' on the model."
            )

    return fs_name


def _extract_model_name(model_uri: str) -> str:
    """Extract model name from URI like models:/name/version."""
    parts = model_uri.replace("models:/", "").split("/")
    return parts[0] if parts else "unknown"


def _extract_model_version(model_uri: str, client: Any) -> str:
    """Extract version from URI; resolve aliases like 'Production'."""
    parts = model_uri.replace("models:/", "").split("/")
    if len(parts) < 2:
        return "latest"
    version_or_alias = parts[1]
    if version_or_alias.isdigit():
        return version_or_alias
    # It's an alias — try to resolve
    try:
        mv = client.get_model_version_by_alias(_extract_model_name(model_uri), version_or_alias)
        return str(mv.version)
    except Exception:
        return version_or_alias


# ---------------------------------------------------------------------------
# Task 1.3: Build entity DataFrame from an MLflow run
# ---------------------------------------------------------------------------

def get_entity_df_from_mlflow_run(
    run_id: str,
    tracking_uri: Optional[str] = None,
    timestamp_column: str = "event_timestamp",
    artifact_name: str = "entity_df.parquet",
) -> "pd.DataFrame":
    """Reconstruct the entity DataFrame from a previous MLflow run.

    Looks for the entity DataFrame in this order:
      1. Run artifact named ``artifact_name`` (default: ``entity_df.parquet``).
      2. Run params ``feast.entity_df_path`` pointing to a local/remote file.

    Args:
        run_id: MLflow run ID to extract the entity DataFrame from.
        tracking_uri: Optional MLflow tracking URI.
        timestamp_column: Name of the timestamp column (default: ``event_timestamp``).
        artifact_name: Name of the artifact to look for (default: ``entity_df.parquet``).

    Returns:
        pandas DataFrame with entity keys and event_timestamp column.

    Raises:
        FeastMLflowError: If MLflow is not installed, the run has no
            suitable entity data, or the data cannot be loaded.

    Example::

        entity_df = get_entity_df_from_mlflow_run("abc123")
        training_df = store.get_historical_features(entity_df, features=fs).to_df()
    """
    try:
        import mlflow
    except ImportError:
        raise FeastMLflowError(
            "mlflow is not installed. Install it with: pip install mlflow"
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()

    # Try to download entity_df artifact
    try:
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]

        if artifact_name in artifact_paths:
            local_path = client.download_artifacts(run_id, artifact_name)
            import pandas as pd
            df = pd.read_parquet(local_path)
            if timestamp_column in df.columns:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            logger.info(
                "Loaded entity DataFrame from artifact '%s' (%d rows)",
                artifact_name, len(df),
            )
            return df

        # Try CSV variant
        csv_name = artifact_name.replace(".parquet", ".csv")
        if csv_name in artifact_paths:
            local_path = client.download_artifacts(run_id, csv_name)
            import pandas as pd
            df = pd.read_csv(local_path)
            if timestamp_column in df.columns:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            logger.info(
                "Loaded entity DataFrame from artifact '%s' (%d rows)",
                csv_name, len(df),
            )
            return df
    except FeastMLflowError:
        raise
    except Exception as e:
        logger.debug("Could not load entity DataFrame from artifacts: %s", e)

    # Fall back to run params
    try:
        run = client.get_run(run_id)
        params = run.data.params
        entity_df_path = params.get("feast.entity_df_path")
        if entity_df_path:
            import pandas as pd
            if entity_df_path.endswith(".parquet"):
                df = pd.read_parquet(entity_df_path)
            elif entity_df_path.endswith(".csv"):
                df = pd.read_csv(entity_df_path)
            else:
                raise FeastMLflowError(
                    f"Unsupported file format: {entity_df_path}"
                )
            if timestamp_column in df.columns:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            logger.info(
                "Loaded entity DataFrame from param path '%s' (%d rows)",
                entity_df_path, len(df),
            )
            return df
    except FeastMLflowError:
        raise
    except Exception as e:
        logger.debug("Could not load entity DataFrame from params: %s", e)

    raise FeastMLflowError(
        f"No entity DataFrame found for run {run_id}. "
        f"Expected artifact '{artifact_name}' or param 'feast.entity_df_path'. "
        f"To enable reproducibility, log the entity DataFrame as an artifact when training: "
        f"mlflow.log_artifact('entity_df.parquet')"
    )


# ---------------------------------------------------------------------------
# Inference helper: load FeatureContract for a model
# ---------------------------------------------------------------------------

def load_feast_contract_for_model(
    model_uri: str,
    tracking_uri: Optional[str] = None,
) -> dict:
    """Load the FeatureContract from an MLflow model's run artifacts.

    Call after ``mlflow.<flavor>.load_model()`` to get the feature schema
    the model was trained on. Use it to validate serving features or to
    know which feature service to call.

    Args:
        model_uri: MLflow model URI (e.g. ``"runs:/<run_id>/model"``).
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        FeatureContract dict with keys: feature_service, features, entity_keys, etc.

    Raises:
        FeastMLflowError: If MLflow is not installed, the run has no contract, etc.

    Example::

        model = mlflow.pytorch.load_model("runs:/abc123/model")
        contract = load_feast_contract_for_model("runs:/abc123/model")
        fs_name = contract["feature_service"]
        features = store.get_online_features(features=store.get_feature_service(fs_name), ...)
    """
    try:
        import mlflow
    except ImportError:
        raise FeastMLflowError("mlflow is not installed")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    import json
    import re
    from pathlib import Path

    match = re.search(r"runs:/([a-f0-9]+)/", model_uri)
    if not match:
        raise FeastMLflowError(f"Cannot extract run ID from model URI: {model_uri}")
    run_id = match.group(1)

    client = mlflow.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id)
        for art in artifacts:
            if art.path == "feature_contract.json":
                local = client.download_artifacts(run_id, art.path)
                return json.loads(Path(local).read_text())
    except Exception as e:
        raise FeastMLflowError(f"Could not load contract from run {run_id}: {e}")

    raise FeastMLflowError(
        f"No feature_contract.json found for run {run_id}. "
        f"Ensure the model was trained with Feast MLflow auto-logging enabled."
    )
