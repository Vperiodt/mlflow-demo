"""
Bidirectional lineage tracker.

Joins two data sources to build a complete graph:
  - Feast registry: data source -> feature view -> feature service
  - MLflow tracking: feature service -> MLflow run -> model

Queryable from both directions:
  - get_models_for_feature_service() -- for Feast UI
  - get_features_for_run()           -- for MLflow UI
  - get_full_lineage()               -- for CLI / visualization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from feast_mlflow.providers.base import (
    LineageEdge,
    LineageGraph,
    LineageNode,
)

if TYPE_CHECKING:
    from feast import FeatureStore

logger = logging.getLogger("feast_mlflow")

FEAST_TAG_PREFIX = "feast.feature_service"


class LineageTracker:
    def __init__(
        self,
        feast_store: "FeatureStore",
        mlflow_tracking_uri: str | None = None,
    ):
        self._store = feast_store
        self._tracking_uri = mlflow_tracking_uri

    def _get_mlflow_client(self):
        import mlflow
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        return mlflow.MlflowClient()

    def _feast_graph(self) -> LineageGraph:
        from feast_mlflow.providers.feast_provider import FeastProvider
        provider = FeastProvider(self._store)
        return provider.get_registry_lineage()

    def _mlflow_runs_for_feature_service(self, feature_service_name: str) -> list[dict]:
        client = self._get_mlflow_client()
        results: list[dict] = []

        try:
            experiments = client.search_experiments()
        except Exception:
            logger.debug("Could not list MLflow experiments", exc_info=True)
            return results

        for exp in experiments:
            try:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"tags.`feast.feature_service` = '{feature_service_name}'",
                    max_results=50,
                )
                for run in runs:
                    metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}
                    results.append({
                        "run_id": run.info.run_id,
                        "experiment_name": exp.name,
                        "experiment_id": exp.experiment_id,
                        "status": run.info.status,
                        "start_time": run.info.start_time,
                        "metrics": metrics,
                        "model_flavor": run.data.tags.get("feast.model_flavor", ""),
                    })
            except Exception:
                logger.debug("Error searching runs in experiment %s", exp.name, exc_info=True)

        return results

    def get_models_for_feature_service(self, feature_service_name: str) -> list[dict]:
        """Return all MLflow runs/models that consumed a given feature service."""
        return self._mlflow_runs_for_feature_service(feature_service_name)

    def get_features_for_run(self, run_id: str) -> dict:
        """Return Feast feature metadata for a given MLflow run."""
        client = self._get_mlflow_client()
        try:
            run = client.get_run(run_id)
        except Exception:
            return {}

        tags = run.data.tags
        return {
            "feature_service": tags.get("feast.feature_service", ""),
            "feature_refs": [r.strip() for r in tags.get("feast.feature_refs", "").split(",") if r.strip()],
            "data_sources": [s.strip() for s in tags.get("feast.data_sources", "").split(",") if s.strip()],
            "entity_keys": [k.strip() for k in tags.get("feast.entity_keys", "").split(",") if k.strip()],
            "project": tags.get("feast.project", ""),
        }

    def get_full_lineage(self, feature_service_name: str | None = None) -> LineageGraph:
        """Build the complete bidirectional graph."""
        feast_graph = self._feast_graph()
        nodes = list(feast_graph.nodes)
        edges = list(feast_graph.edges)
        seen_ids = {n.id for n in nodes}

        target_services: list[str] = []
        if feature_service_name:
            target_services = [feature_service_name]
        else:
            target_services = [
                n.label for n in nodes if n.node_type == "featureservice"
            ]

        for fs_name in target_services:
            fs_id = f"fs:{fs_name}"
            runs = self._mlflow_runs_for_feature_service(fs_name)

            for run_info in runs:
                run_id = run_info["run_id"]
                run_node_id = f"run:{run_id}"

                if run_node_id not in seen_ids:
                    metrics_str = ", ".join(
                        f"{k}={v}" for k, v in list(run_info["metrics"].items())[:3]
                    )
                    nodes.append(LineageNode(
                        id=run_node_id,
                        node_type="mlflow_run",
                        label=f"Run {run_id[:8]}",
                        metadata={
                            "run_id": run_id,
                            "experiment": run_info["experiment_name"],
                            "status": run_info["status"],
                            "metrics": run_info["metrics"],
                            "metrics_summary": metrics_str,
                            "model_flavor": run_info["model_flavor"],
                        },
                    ))
                    seen_ids.add(run_node_id)

                edges.append(LineageEdge(source=fs_id, target=run_node_id))

        return LineageGraph(nodes=nodes, edges=edges)
