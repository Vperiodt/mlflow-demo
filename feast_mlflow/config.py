"""
Read the optional `mlflow:` config block from feature_store.yaml
and environment variables to determine bridge activation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BridgeConfig:
    tracking_uri: str = ""
    auto_log: bool = False
    lineage: bool = False
    enabled: bool = False

    @classmethod
    def from_yaml(cls, feast_repo_path: str | Path) -> "BridgeConfig":
        yaml_path = Path(feast_repo_path) / "feature_store.yaml"
        if not yaml_path.exists():
            return cls()

        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}

        mlflow_block = raw.get("mlflow")
        if not mlflow_block or not isinstance(mlflow_block, dict):
            return cls()

        return cls(
            tracking_uri=str(mlflow_block.get("tracking_uri", "")),
            auto_log=bool(mlflow_block.get("auto_log", False)),
            lineage=bool(mlflow_block.get("lineage", False)),
            enabled=True,
        )

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        flag = os.environ.get("FEAST_MLFLOW", "").strip()
        if flag not in ("1", "true", "yes"):
            return cls()

        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            auto_log=True,
            lineage=True,
            enabled=True,
        )

    @classmethod
    def resolve(cls, feast_repo_path: str | Path | None = None) -> "BridgeConfig":
        env_cfg = cls.from_env()
        if env_cfg.enabled:
            return env_cfg

        if feast_repo_path:
            yaml_cfg = cls.from_yaml(feast_repo_path)
            if yaml_cfg.enabled:
                return yaml_cfg

        return cls()
