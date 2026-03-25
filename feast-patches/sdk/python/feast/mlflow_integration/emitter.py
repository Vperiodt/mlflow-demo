"""
FeastMlflowEmitter: logs feature metadata to MLflow and records training runs
in the Feast registry for lineage tracking.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from threading import local as ThreadLocal
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from feast.mlflow_integration.config import MlflowIntegrationConfig

if TYPE_CHECKING:
    from feast.infra.registry.base_registry import BaseRegistry

logger = logging.getLogger("feast.mlflow_integration")

_context = ThreadLocal()


class FeastMlflowEmitter:
    """Logs Feast feature metadata to MLflow and records lineage in the registry."""

    def __init__(self, config: MlflowIntegrationConfig):
        self._config = config
        self._mlflow = None

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    @property
    def _mlflow_mod(self):
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self._config.tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                logger.debug("mlflow not installed; MLflow integration disabled")
                return None
        return self._mlflow

    def emit_historical_features(
        self,
        *,
        feature_service_name: str,
        project: str,
        feature_refs: List[str],
        feature_views: List[Any],
        entity_keys: List[str],
        data_sources: List[str],
        retrieval_job: Any = None,
    ) -> None:
        """Called after get_historical_features. Logs to MLflow or caches context."""
        mlflow = self._mlflow_mod
        if mlflow is None:
            return

        active_run = mlflow.active_run()
        if active_run is not None:
            self._log_to_mlflow(
                mlflow=mlflow,
                run_id=active_run.info.run_id,
                feature_service_name=feature_service_name,
                project=project,
                feature_refs=feature_refs,
                feature_views=feature_views,
                entity_keys=entity_keys,
                data_sources=data_sources,
            )
        else:
            self._cache_context(
                feature_service_name=feature_service_name,
                project=project,
                feature_refs=feature_refs,
                feature_views=feature_views,
                entity_keys=entity_keys,
                data_sources=data_sources,
            )

    def flush_deferred(self) -> None:
        """Flush cached context to the current active MLflow run."""
        mlflow = self._mlflow_mod
        if mlflow is None:
            return
        active_run = mlflow.active_run()
        ctx = self._get_context()
        if active_run and ctx and not ctx.get("logged"):
            self._log_to_mlflow(
                mlflow=mlflow,
                run_id=active_run.info.run_id,
                **{k: v for k, v in ctx.items() if k != "logged"},
            )

    def record_training_run(
        self,
        registry: "BaseRegistry",
        *,
        run_id: str,
        experiment_name: str,
        feature_service_name: str,
        project: str,
        feature_refs: List[str],
    ) -> None:
        """Record a training run in the Feast registry for lineage."""
        if not self._config.lineage:
            return
        try:
            registry.record_training_run(
                project=project,
                run_id=run_id,
                experiment_name=experiment_name,
                tracking_uri=self._config.tracking_uri,
                feature_service_name=feature_service_name,
                feature_refs=feature_refs,
            )
        except Exception:
            logger.debug("Could not record training run in registry", exc_info=True)

    def _log_to_mlflow(
        self,
        *,
        mlflow: Any,
        run_id: str,
        feature_service_name: str,
        project: str,
        feature_refs: List[str],
        feature_views: List[Any],
        entity_keys: List[str],
        data_sources: List[str],
    ) -> None:
        try:
            mlflow.set_tag("feast.feature_service", feature_service_name)
            mlflow.set_tag("feast.project", project)
            mlflow.set_tag("feast.feature_refs", ",".join(feature_refs))
            mlflow.set_tag("feast.entity_keys", ",".join(entity_keys))
            mlflow.set_tag("feast.data_sources", ",".join(data_sources))
            mlflow.set_tag("feast_mlflow.version", "native")

            mlflow.log_params({
                "feast.feature_service": feature_service_name,
                "feast.num_features": len(feature_refs),
                "feast.num_entities": len(entity_keys),
            })
        except Exception:
            logger.debug("Could not log Feast tags/params to MLflow", exc_info=True)

        try:
            from feast.mlflow_integration.contract import FeatureContract

            contract = FeatureContract.build(
                feature_service_name=feature_service_name,
                project=project,
                run_id=run_id,
                feature_refs=feature_refs,
                entity_keys=entity_keys,
                data_sources=data_sources,
                feature_views=feature_views,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                contract_path = contract.save(tmpdir)
                mlflow.log_artifact(str(contract_path))
        except Exception:
            logger.warning("Could not log FeatureContract to MLflow", exc_info=True)

        try:
            schema_data = []
            for fv in feature_views:
                fv_name = fv.name if hasattr(fv, "name") else str(fv)
                if hasattr(fv, "features"):
                    for feat in fv.features:
                        schema_data.append({
                            "name": feat.name,
                            "dtype": str(feat.dtype),
                            "source_view": fv_name,
                        })
            if schema_data:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, prefix="feast_schema_"
                ) as f:
                    json.dump(schema_data, f, indent=2)
                    schema_path = f.name
                mlflow.log_artifact(schema_path, artifact_path="feast")
                Path(schema_path).unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not log feature schema artifact", exc_info=True)

        self._set_context_logged()

    def _cache_context(self, **kwargs: Any) -> None:
        _context.feast_ctx = {**kwargs, "logged": False}

    def _get_context(self) -> Optional[Dict[str, Any]]:
        return getattr(_context, "feast_ctx", None)

    def _set_context_logged(self) -> None:
        ctx = self._get_context()
        if ctx:
            ctx["logged"] = True

    def clear_context(self) -> None:
        _context.feast_ctx = None

    # -- Online serving: automatic contract validation -------------------------

    def validate_online_features(
        self,
        *,
        feature_service_name: str,
        project: str,
        registry: "BaseRegistry",
    ) -> Optional[Dict[str, Any]]:
        """Validate that the online features match the most recent training contract.

        Called automatically by Feast's get_online_features when mlflow config is
        enabled. Downloads the FeatureContract from the latest MLflow run that
        used this feature service and checks every feature for dtype and online
        availability.

        Returns a dict with validation results, or None if no contract is found.
        """
        if not self._config.auto_log:
            return None

        cache_key = f"contract_validation:{feature_service_name}"
        cached = getattr(_context, cache_key, None)
        if cached is not None:
            return cached

        mlflow = self._mlflow_mod
        if mlflow is None:
            return None

        try:
            client = mlflow.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[],
                filter_string=f"tags.`feast.feature_service` = '{feature_service_name}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs:
                return None

            run = runs[0]
            artifacts = client.list_artifacts(run.info.run_id)
            contract_path = None
            for art in artifacts:
                if art.path == "feature_contract.json":
                    contract_path = client.download_artifacts(
                        run.info.run_id, art.path
                    )
                    break

            if contract_path is None:
                return None

            from feast.mlflow_integration.contract import FeatureContract
            contract = FeatureContract.from_file(contract_path)

            current_features = set()
            for fv in registry.list_feature_views(project=project, allow_cache=True):
                for feat in fv.features:
                    current_features.add(feat.name)

            missing = []
            for f in contract.features:
                if f.name not in current_features:
                    missing.append(f.name)

            result = {
                "valid": len(missing) == 0,
                "contract_run_id": run.info.run_id,
                "feature_service": feature_service_name,
                "total_features": len(contract.features),
                "missing_features": missing,
            }

            setattr(_context, cache_key, result)

            if missing:
                logger.warning(
                    "Feast-MLflow contract validation: %d missing features for %s: %s",
                    len(missing), feature_service_name, missing,
                )

            return result
        except Exception:
            logger.debug("Online contract validation failed", exc_info=True)
            return None
