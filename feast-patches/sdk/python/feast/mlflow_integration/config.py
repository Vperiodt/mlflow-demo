"""Runtime configuration for Feast MLflow integration."""

import os
from dataclasses import dataclass


@dataclass
class MlflowIntegrationConfig:
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    auto_log: bool = True
    lineage: bool = True

    @classmethod
    def from_env(cls) -> "MlflowIntegrationConfig":
        flag = os.environ.get("FEAST_MLFLOW", "").strip().lower()
        if flag not in ("1", "true", "yes"):
            return cls(enabled=False)

        return cls(
            enabled=True,
            tracking_uri=os.environ.get(
                "MLFLOW_TRACKING_URI", "http://localhost:5000"
            ),
            auto_log=True,
            lineage=True,
        )
