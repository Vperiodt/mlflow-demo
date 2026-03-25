"""
FeatureContract: schema snapshot for training-serving consistency validation.

Saved as an MLflow artifact during training; loaded and validated at inference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("feast_mlflow")

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
    features: list[FeatureSpec] = field(default_factory=list)
    entity_keys: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

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
    def from_dict(cls, data: dict) -> "FeatureContract":
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
    def from_json(cls, text: str) -> "FeatureContract":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: str | Path) -> "FeatureContract":
        return cls.from_json(Path(path).read_text())

    @classmethod
    def build(
        cls,
        metadata,  # FeatureMetadata from provider
        run_id: str,
        df=None,  # optional pandas DataFrame for statistics
    ) -> "FeatureContract":
        from feast_mlflow.providers.base import FeatureMetadata

        stats: dict[str, Any] = {}
        if df is not None:
            stats["row_count"] = len(df)
            null_counts = {}
            feature_means = {}
            for col in df.columns:
                if col in ("event_timestamp",) or col in metadata.entity_keys:
                    continue
                nulls = int(df[col].isnull().sum())
                if nulls > 0:
                    null_counts[col] = nulls
                if df[col].dtype.kind in ("f", "i", "u"):
                    feature_means[col] = round(float(df[col].mean()), 4)
            if null_counts:
                stats["null_counts"] = null_counts
            if feature_means:
                stats["feature_means"] = feature_means

        return cls(
            feature_service=metadata.feature_service_name,
            feast_project=metadata.project,
            created_at=datetime.now(timezone.utc).isoformat(),
            mlflow_run_id=run_id,
            features=[
                FeatureSpec(name=s.name, dtype=s.dtype, source_view=s.source_view)
                for s in metadata.schema
            ],
            entity_keys=metadata.entity_keys,
            data_sources=metadata.data_sources,
            statistics=stats,
        )


@dataclass
class ValidationResult:
    feature_name: str
    dtype_expected: str
    dtype_actual: str
    online_available: bool
    schema_match: bool

    @property
    def valid(self) -> bool:
        return self.online_available and self.schema_match


def validate_contract(
    contract: FeatureContract,
    provider,  # FeatureStoreProvider
) -> list[ValidationResult]:
    current_schema = provider.get_feature_schema(contract.feature_service)
    online_avail = provider.check_online_availability(contract.feature_service)

    current_by_name = {s.name: s for s in current_schema}
    results: list[ValidationResult] = []

    for feat in contract.features:
        current = current_by_name.get(feat.name)
        if current is None:
            results.append(ValidationResult(
                feature_name=feat.name,
                dtype_expected=feat.dtype,
                dtype_actual="MISSING",
                online_available=False,
                schema_match=False,
            ))
            continue

        schema_match = feat.dtype == current.dtype
        results.append(ValidationResult(
            feature_name=feat.name,
            dtype_expected=feat.dtype,
            dtype_actual=current.dtype,
            online_available=online_avail.get(feat.name, False),
            schema_match=schema_match,
        ))

    return results
