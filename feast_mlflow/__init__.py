"""
feast_mlflow: Stage 2 MLflow PR — convenience helpers for Feast integration.

These helpers would live in mlflow/integrations/feast/ if accepted upstream.
They use only standard Feast APIs and work with stock pip-installed Feast.

- feast_tags.py:  set_feast_tags, log_model_with_feast_context
- evaluation.py:  evaluate_with_feast
- tracing.py:     traced_get_online_features, feast_span
"""

__version__ = "0.1.0"
