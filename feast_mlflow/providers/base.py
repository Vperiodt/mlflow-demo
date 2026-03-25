"""Abstract interface for feature store providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FeatureSchema:
    name: str
    dtype: str
    source_view: str


@dataclass
class FeatureMetadata:
    feature_service_name: str
    project: str
    feature_refs: list[str]
    data_sources: list[str]
    entity_keys: list[str]
    schema: list[FeatureSchema]


@dataclass
class LineageNode:
    id: str
    node_type: str  # datasource | featureview | featureservice | mlflow_run | model
    label: str
    metadata: dict


@dataclass
class LineageEdge:
    source: str
    target: str


@dataclass
class LineageGraph:
    nodes: list[LineageNode]
    edges: list[LineageEdge]

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n.id, "type": n.node_type, "label": n.label, "metadata": n.metadata}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target}
                for e in self.edges
            ],
        }


class FeatureStoreProvider(ABC):
    @abstractmethod
    def get_feature_metadata(self, features) -> FeatureMetadata:
        """Extract metadata from a feature service or feature list."""
        ...

    @abstractmethod
    def get_feature_schema(self, feature_service_name: str) -> list[FeatureSchema]:
        """Return the schema of all features in a feature service."""
        ...

    @abstractmethod
    def check_online_availability(self, feature_service_name: str) -> dict[str, bool]:
        """Check which features are available in the online store."""
        ...

    @abstractmethod
    def get_registry_lineage(self) -> LineageGraph:
        """Build a lineage graph from the feature store registry."""
        ...
