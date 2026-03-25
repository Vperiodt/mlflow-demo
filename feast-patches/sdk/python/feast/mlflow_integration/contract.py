"""
FeatureContract: schema snapshot for training-serving consistency validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional


CONTRACT_FILENAME = "feature_contract.json"
CONTRACT_VERSION = "1.0"


@dataclass
class FeatureSpec:
    name: str
    dtype: str
    source_view: str


@dataclass
class FeatureContract:
    contract_version: str = CONTRACT_VERSION
    feature_service: str = ""
    feast_project: str = ""
    created_at: str = ""
    mlflow_run_id: str = ""
    features: List[FeatureSpec] = field(default_factory=list)
    entity_keys: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = [asdict(f) for f in self.features]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / CONTRACT_FILENAME
        path.write_text(self.to_json())
        return path

    @classmethod
    def from_dict(cls, data: dict) -> FeatureContract:
        features = [FeatureSpec(**f) for f in data.get("features", [])]
        return cls(
            contract_version=data.get("contract_version", CONTRACT_VERSION),
            feature_service=data.get("feature_service", ""),
            feast_project=data.get("feast_project", ""),
            created_at=data.get("created_at", ""),
            mlflow_run_id=data.get("mlflow_run_id", ""),
            features=features,
            entity_keys=data.get("entity_keys", []),
            data_sources=data.get("data_sources", []),
            statistics=data.get("statistics", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> FeatureContract:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: str | Path) -> FeatureContract:
        return cls.from_json(Path(path).read_text())

    @classmethod
    def build(
        cls,
        *,
        feature_service_name: str,
        project: str,
        run_id: str,
        feature_refs: List[str],
        entity_keys: List[str],
        data_sources: List[str],
        feature_views: Optional[list] = None,
        df: Any = None,
    ) -> FeatureContract:
        features = []
        if feature_views:
            for fv in feature_views:
                fv_name = fv.name if hasattr(fv, "name") else str(fv)
                if hasattr(fv, "features"):
                    for feat in fv.features:
                        features.append(FeatureSpec(
                            name=feat.name,
                            dtype=str(feat.dtype),
                            source_view=fv_name,
                        ))

        stats: dict = {}
        if df is not None:
            try:
                stats["row_count"] = len(df)
            except Exception:
                pass

        return cls(
            feature_service=feature_service_name,
            feast_project=project,
            created_at=datetime.now(timezone.utc).isoformat(),
            mlflow_run_id=run_id,
            features=features,
            entity_keys=entity_keys,
            data_sources=data_sources,
            statistics=stats,
        )
