"""
feast-mlflow: Invisible bridge between feature stores and experiment tracking.

Activation modes (pick one):
    1. Config: add ``mlflow:`` block to ``feature_store.yaml``
    2. Env var: ``FEAST_MLFLOW=1``
    3. Explicit: ``feast_mlflow.autolog()``  (MLflow-style API)
    4. Explicit: ``feast_mlflow.enable()``
"""

from feast_mlflow.bridge import enable, disable, is_active

__version__ = "0.1.0"
__all__ = ["autolog", "enable", "disable", "is_active"]


def autolog() -> None:
    """MLflow-style autologging entry point.

    Patches Feast's ``get_historical_features`` and MLflow's ``log_model``
    so that feature metadata flows automatically into experiment tracking.

    Usage::

        import feast_mlflow
        feast_mlflow.autolog()  # one-liner, MLflow convention
    """
    enable()
