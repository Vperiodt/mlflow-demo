"""Feast implementation of FeatureStoreProvider."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from feast_mlflow.providers.base import (
    FeatureMetadata,
    FeatureSchema,
    FeatureStoreProvider,
    LineageEdge,
    LineageGraph,
    LineageNode,
)

if TYPE_CHECKING:
    from feast import FeatureStore

logger = logging.getLogger("feast_mlflow")


class FeastProvider(FeatureStoreProvider):
    def __init__(self, store: "FeatureStore"):
        self._store = store

    def get_feature_metadata(self, features) -> FeatureMetadata:
        from feast import FeatureService

        if isinstance(features, FeatureService):
            fs = features
        elif isinstance(features, str):
            fs = self._store.get_feature_service(features)
        else:
            return FeatureMetadata(
                feature_service_name="unknown",
                project=self._store.project,
                feature_refs=[],
                data_sources=[],
                entity_keys=[],
                schema=[],
            )

        refs: list[str] = []
        schemas: list[FeatureSchema] = []
        data_sources: list[str] = []
        entity_keys: set[str] = set()

        for projection in fs.feature_view_projections:
            fv_name = projection.name
            for feat in projection.features:
                ref = f"{fv_name}__{feat.name}"
                refs.append(ref)
                schemas.append(FeatureSchema(
                    name=feat.name, dtype=str(feat.dtype), source_view=fv_name,
                ))

            try:
                fv = self._store.get_feature_view(fv_name)
                for entity_col in fv.entity_columns:
                    entity_keys.add(entity_col.name)
                if hasattr(fv, "entities"):
                    for ent_name in fv.entities:
                        entity_keys.add(ent_name)
                if hasattr(fv.batch_source, "path"):
                    data_sources.append(fv.batch_source.path)
                elif hasattr(fv.batch_source, "name"):
                    data_sources.append(fv.batch_source.name)
            except Exception:
                logger.debug("Could not resolve feature view %s", fv_name, exc_info=True)

        return FeatureMetadata(
            feature_service_name=fs.name,
            project=self._store.project,
            feature_refs=sorted(refs),
            data_sources=list(dict.fromkeys(data_sources)),
            entity_keys=sorted(entity_keys),
            schema=schemas,
        )

    def get_feature_schema(self, feature_service_name: str) -> list[FeatureSchema]:
        meta = self.get_feature_metadata(feature_service_name)
        return meta.schema

    def check_online_availability(self, feature_service_name: str) -> dict[str, bool]:
        meta = self.get_feature_metadata(feature_service_name)
        result: dict[str, bool] = {}
        for s in meta.schema:
            try:
                fv = self._store.get_feature_view(s.source_view)
                result[s.name] = getattr(fv, "online", False)
            except Exception:
                result[s.name] = False
        return result

    def get_registry_lineage(self) -> LineageGraph:
        nodes: list[LineageNode] = []
        edges: list[LineageEdge] = []
        seen_ids: set[str] = set()

        for fs in self._store.list_feature_services():
            fs_id = f"fs:{fs.name}"
            if fs_id not in seen_ids:
                nodes.append(LineageNode(
                    id=fs_id, node_type="featureservice",
                    label=fs.name, metadata={"description": fs.description or ""},
                ))
                seen_ids.add(fs_id)

            for proj in fs.feature_view_projections:
                fv_id = f"fv:{proj.name}"
                if fv_id not in seen_ids:
                    feat_count = len(proj.features)
                    nodes.append(LineageNode(
                        id=fv_id, node_type="featureview",
                        label=proj.name, metadata={"feature_count": feat_count},
                    ))
                    seen_ids.add(fv_id)
                edges.append(LineageEdge(source=fv_id, target=fs_id))

                try:
                    fv = self._store.get_feature_view(proj.name)
                    src = fv.batch_source
                    src_path = getattr(src, "path", None) or getattr(src, "name", "unknown")
                    ds_id = f"ds:{src_path}"
                    if ds_id not in seen_ids:
                        nodes.append(LineageNode(
                            id=ds_id, node_type="datasource",
                            label=str(src_path), metadata={"type": type(src).__name__},
                        ))
                        seen_ids.add(ds_id)
                    edges.append(LineageEdge(source=ds_id, target=fv_id))
                except Exception:
                    pass

        return LineageGraph(nodes=nodes, edges=edges)
