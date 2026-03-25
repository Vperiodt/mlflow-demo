"""
FeastDataset and FeastDatasetSource: custom MLflow Dataset types
so Feast features appear natively in MLflow's dataset tracking UI.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("feast_mlflow")


class FeastDatasetSource:
    """MLflow-compatible dataset source pointing to a Feast feature service."""

    def __init__(
        self,
        feature_service_name: str,
        feast_project: str,
        feature_views: list[str],
        data_sources: list[str],
    ):
        self.feature_service_name = feature_service_name
        self.feast_project = feast_project
        self.feature_views = feature_views
        self.data_sources = data_sources

    @staticmethod
    def _get_source_type() -> str:
        return "feast"

    def to_dict(self) -> dict:
        return {
            "source_type": "feast",
            "feature_service": self.feature_service_name,
            "feast_project": self.feast_project,
            "feature_views": self.feature_views,
            "data_sources": self.data_sources,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class FeastDataset:
    """Lightweight dataset wrapper that can be logged via mlflow.log_input()."""

    def __init__(
        self,
        df: pd.DataFrame,
        source: FeastDatasetSource,
        name: str | None = None,
        targets: str | None = None,
    ):
        self._df = df
        self._source = source
        self._name = name or f"feast:{source.feature_service_name}"
        self._targets = targets
        self._digest = self._compute_digest()

    @property
    def name(self) -> str:
        return self._name

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def source(self) -> FeastDatasetSource:
        return self._source

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _compute_digest(self) -> str:
        sig = (
            f"{self._source.feature_service_name}:"
            f"{len(self._df)}:"
            f"{','.join(sorted(self._df.columns))}"
        )
        return hashlib.md5(sig.encode()).hexdigest()[:8]

    def profile(self) -> dict[str, Any]:
        return {
            "num_rows": len(self._df),
            "num_columns": len(self._df.columns),
            "feature_service": self._source.feature_service_name,
            "feast_project": self._source.feast_project,
        }

    def schema_dict(self) -> dict[str, str]:
        return {col: str(self._df[col].dtype) for col in self._df.columns}


def log_feast_dataset(
    dataset: FeastDataset,
    context: str = "training",
) -> None:
    """Log a FeastDataset to the active MLflow run using mlflow.data APIs if available,
    falling back to tag-based logging."""
    import mlflow

    try:
        mlflow_dataset = mlflow.data.from_pandas(
            dataset.df,
            source=dataset.source.to_json(),
            name=dataset.name,
            targets=dataset._targets,
        )
        mlflow.log_input(mlflow_dataset, context=context)
    except Exception:
        logger.debug("mlflow.data.from_pandas unavailable, falling back to tags", exc_info=True)
        mlflow.set_tag("feast_mlflow.dataset.name", dataset.name)
        mlflow.set_tag("feast_mlflow.dataset.digest", dataset.digest)
        mlflow.set_tag("feast_mlflow.dataset.source", dataset.source.to_json())
