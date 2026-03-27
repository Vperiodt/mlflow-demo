"""
Stage 2 Task 2.2: Evaluate model with features from Feast.

End-to-end evaluation: entity DataFrame → Feast historical features → merge
with labels → mlflow.evaluate(). Results appear in the MLflow UI with metrics,
confusion matrix, and feature context.

Usage::

    from feast import FeatureStore
    from feast_mlflow.evaluation import evaluate_with_feast

    store = FeatureStore(repo_path="feature_repo")
    results = evaluate_with_feast(
        model_uri="runs:/abc123/model",
        store=store,
        feature_service_name="fraud_feature_service",
        entity_df=eval_entity_df,
        label_column="is_fraud",
    )
    print(results.metrics)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import pandas as pd
    from feast import FeatureStore

logger = logging.getLogger("feast_mlflow.evaluation")


def evaluate_with_feast(
    model_uri: str,
    store: "FeatureStore",
    feature_service_name: str,
    entity_df: "pd.DataFrame",
    label_column: str,
    tracking_uri: Optional[str] = None,
    evaluator_config: Optional[Dict[str, Any]] = None,
    model_type: str = "classifier",
) -> Any:
    """Evaluate a model using features from Feast.

    Fetches historical features from Feast, merges with labels from
    ``entity_df``, and runs ``mlflow.evaluate()`` on the result.

    Args:
        model_uri: MLflow model URI (e.g. ``"runs:/abc123/model"``).
        store: Feast FeatureStore instance.
        feature_service_name: Feast feature service to fetch features from.
        entity_df: DataFrame with entity keys, event_timestamp, and label column.
        label_column: Name of the label/target column in ``entity_df``.
        tracking_uri: Optional MLflow tracking URI.
        evaluator_config: Optional config dict for ``mlflow.evaluate()``.
        model_type: Model type for evaluation (default: ``"classifier"``).

    Returns:
        ``mlflow.models.EvaluationResult`` with metrics and artifacts.

    Example::

        results = evaluate_with_feast(
            model_uri="runs:/abc123/model",
            store=store,
            feature_service_name="fraud_feature_service",
            entity_df=eval_df,  # must have entity keys + event_timestamp + label
            label_column="is_fraud",
        )
        print(f"AUC: {results.metrics['auc']}")
    """
    import mlflow
    import pandas as pd

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    feature_service = store.get_feature_service(feature_service_name)

    entity_cols = [c for c in entity_df.columns if c not in (label_column, "event_timestamp")]
    retrieval_df = entity_df[entity_cols + ["event_timestamp"]].copy()
    retrieval_df["event_timestamp"] = pd.to_datetime(retrieval_df["event_timestamp"])

    logger.info(
        "Fetching features from '%s' for %d entities...",
        feature_service_name, len(retrieval_df),
    )
    features_df = store.get_historical_features(
        entity_df=retrieval_df,
        features=feature_service,
    ).to_df()

    labels_df = entity_df[entity_cols + [label_column]].copy()
    eval_df = features_df.merge(labels_df, on=entity_cols, how="left")

    logger.info(
        "Evaluating model '%s' on %d rows with %d features...",
        model_uri, len(eval_df), len(features_df.columns) - len(entity_cols) - 1,
    )

    eval_config = evaluator_config or {}

    result = mlflow.evaluate(
        model=model_uri,
        data=eval_df,
        targets=label_column,
        model_type=model_type,
        **eval_config,
    )

    return result
