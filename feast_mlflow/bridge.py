"""
The invisible bridge: patches Feast and MLflow at the Python level so that
feature store metadata flows into experiment tracking automatically.

Activation:
    1. `mlflow:` block in feature_store.yaml  (auto-detected on FeatureStore init)
    2. FEAST_MLFLOW=1 env var
    3. feast_mlflow.enable()

Design principle: every intercepted call falls through to the original
implementation. If the bridge encounters an error it logs a warning and
lets the original call succeed. The bridge never breaks user code.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from threading import local as ThreadLocal
from typing import Any

logger = logging.getLogger("feast_mlflow")

_state = ThreadLocal()
_originals: dict[str, Any] = {}
_active = False


def is_active() -> bool:
    return _active


def enable(feast_repo_path: str | Path | None = None) -> None:
    """Activate the bridge. Safe to call multiple times."""
    global _active
    if _active:
        return
    _install_patches(feast_repo_path)
    _active = True
    logger.info("feast-mlflow bridge activated")


def disable() -> None:
    """Deactivate the bridge and restore original functions."""
    global _active
    if not _active:
        return
    _uninstall_patches()
    _active = False
    logger.info("feast-mlflow bridge deactivated")


# -- Thread-local context for tracking Feast calls within an MLflow run ------

def _get_feast_context() -> dict:
    if not hasattr(_state, "feast_context"):
        _state.feast_context = {}
    return _state.feast_context


def _set_feast_context(ctx: dict) -> None:
    _state.feast_context = ctx


def _clear_feast_context() -> None:
    _state.feast_context = {}


# -- Patch installation / removal -------------------------------------------

def _install_patches(feast_repo_path: str | Path | None = None) -> None:
    import feast
    import mlflow

    # 1. Patch FeatureStore.__init__
    if "fs_init" not in _originals:
        _originals["fs_init"] = feast.FeatureStore.__init__

    original_init = _originals["fs_init"]

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _on_store_init(self)

    feast.FeatureStore.__init__ = _patched_init

    # 2. Patch get_historical_features
    if "get_historical" not in _originals:
        _originals["get_historical"] = feast.FeatureStore.get_historical_features

    original_get_historical = _originals["get_historical"]

    def _patched_get_historical(self, entity_df, features, *args, **kwargs):
        result = original_get_historical(self, entity_df, features, *args, **kwargs)
        try:
            _on_get_historical_features(self, entity_df, features, result)
        except Exception:
            logger.warning("feast-mlflow bridge: error in get_historical_features hook", exc_info=True)
        return result

    feast.FeatureStore.get_historical_features = _patched_get_historical

    # 3. Patch get_online_features
    if "get_online" not in _originals:
        _originals["get_online"] = feast.FeatureStore.get_online_features

    original_get_online = _originals["get_online"]

    def _patched_get_online(self, features, entity_rows, *args, **kwargs):
        result = original_get_online(self, features, entity_rows, *args, **kwargs)
        try:
            _on_get_online_features(self, features, entity_rows, result)
        except Exception:
            logger.warning("feast-mlflow bridge: error in get_online_features hook", exc_info=True)
        return result

    feast.FeatureStore.get_online_features = _patched_get_online

    # 4. Patch mlflow log_model functions to attach FeatureContract
    _patch_mlflow_log_model(mlflow)

    # 5. Patch mlflow load_model functions to auto-attach FeatureContract at inference
    _patch_mlflow_load_model(mlflow)

    # Auto-activate from env var on import
    if not feast_repo_path:
        from feast_mlflow.config import BridgeConfig
        env_cfg = BridgeConfig.from_env()
        if env_cfg.enabled and env_cfg.tracking_uri:
            os.environ.setdefault("MLFLOW_TRACKING_URI", env_cfg.tracking_uri)


def _uninstall_patches() -> None:
    import feast
    import mlflow

    if "fs_init" in _originals:
        feast.FeatureStore.__init__ = _originals.pop("fs_init")
    if "get_historical" in _originals:
        feast.FeatureStore.get_historical_features = _originals.pop("get_historical")
    if "get_online" in _originals:
        feast.FeatureStore.get_online_features = _originals.pop("get_online")

    import sys

    for key in list(_originals.keys()):
        if key.startswith(("log_model:", "load_model:")):
            func_name, flavor_name = key.split(":", 1)
            real_mod = sys.modules.get(f"mlflow.{flavor_name}")
            if real_mod and hasattr(real_mod, func_name):
                setattr(real_mod, func_name, _originals.pop(key))

    _clear_feast_context()


# -- Hook implementations ---------------------------------------------------

def _on_store_init(store) -> None:
    """Called after FeatureStore.__init__. Reads config and sets tracking URI."""
    from feast_mlflow.config import BridgeConfig

    repo_path = getattr(store, "repo_path", None) or getattr(store, "_repo_path", None)
    if repo_path:
        cfg = BridgeConfig.from_yaml(repo_path)
        if cfg.enabled and cfg.tracking_uri:
            os.environ.setdefault("MLFLOW_TRACKING_URI", cfg.tracking_uri)
            import mlflow
            mlflow.set_tracking_uri(cfg.tracking_uri)


def _on_get_historical_features(store, entity_df, features, result) -> None:
    """Called after get_historical_features succeeds. Auto-logs to MLflow."""
    import mlflow

    active_run = mlflow.active_run()
    if active_run is None:
        _cache_feast_context(store, features, result)
        return

    _log_feast_metadata(store, features, result, active_run.info.run_id)


def _cache_feast_context(store, features, result) -> None:
    """Store context for deferred logging when get_historical_features is
    called before mlflow.start_run()."""
    from feast_mlflow.providers.feast_provider import FeastProvider

    provider = FeastProvider(store)
    metadata = provider.get_feature_metadata(features)

    _set_feast_context({
        "store": store,
        "features": features,
        "metadata": metadata,
        "result": result,
        "logged": False,
    })


def _materialize_historical_result(result) -> Any:
    """Retrieval jobs may only allow one ``to_df()``; training often calls it before
    ``log_model`` triggers deferred logging. Return None if already consumed."""
    if result is None:
        return None
    try:
        if hasattr(result, "to_df"):
            return result.to_df()
        return result
    except Exception:
        logger.debug(
            "feast-mlflow: historical result not materializable (likely already finalized)",
            exc_info=True,
        )
        return None


def _log_feast_metadata(store, features, result, run_id: str) -> None:
    """Perform all auto-logging for a training-time feature fetch."""
    import mlflow

    from feast_mlflow.contract import FeatureContract
    from feast_mlflow.dataset import FeastDataset, FeastDatasetSource, log_feast_dataset
    from feast_mlflow.providers.feast_provider import FeastProvider

    provider = FeastProvider(store)
    metadata = provider.get_feature_metadata(features)

    # 1. Log tags
    mlflow.set_tag("feast.feature_service", metadata.feature_service_name)
    mlflow.set_tag("feast.feature_refs", ",".join(metadata.feature_refs))
    mlflow.set_tag("feast.data_sources", ",".join(metadata.data_sources))
    mlflow.set_tag("feast.entity_keys", ",".join(metadata.entity_keys))
    mlflow.set_tag("feast.project", metadata.project)
    mlflow.set_tag("feast_mlflow.version", "0.1.0")

    # 2. Log params
    mlflow.log_params({
        "feast.feature_service": metadata.feature_service_name,
        "feast.num_features": len(metadata.feature_refs),
        "feast.num_entities": len(metadata.entity_keys),
    })

    df = _materialize_historical_result(result)

    # 3. Build and log dataset (optional; needs a materialized frame)
    if df is not None:
        try:
            source = FeastDatasetSource(
                feature_service_name=metadata.feature_service_name,
                feast_project=metadata.project,
                feature_views=list({s.source_view for s in metadata.schema}),
                data_sources=metadata.data_sources,
            )
            dataset = FeastDataset(df=df, source=source)
            log_feast_dataset(dataset, context="training")
        except Exception:
            logger.debug("Could not log FeastDataset", exc_info=True)

    # 4. Build and log FeatureContract (schema from metadata; df only enriches stats)
    try:
        contract = FeatureContract.build(metadata=metadata, run_id=run_id, df=df)
        with tempfile.TemporaryDirectory() as tmpdir:
            contract_path = contract.save(tmpdir)
            mlflow.log_artifact(str(contract_path))
    except Exception:
        logger.warning("feast-mlflow: could not log FeatureContract", exc_info=True)

    # 5. Log feature schema as artifact
    try:
        import json
        schema_data = [{"name": s.name, "dtype": s.dtype, "source": s.source_view} for s in metadata.schema]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="feast_schema_") as f:
            json.dump(schema_data, f, indent=2)
            schema_path = f.name
        mlflow.log_artifact(schema_path, artifact_path="feast")
        Path(schema_path).unlink(missing_ok=True)
    except Exception:
        logger.debug("Could not log feature schema", exc_info=True)

    # 6. Log rich artifacts (HTML lineage, Markdown summary, run note)
    try:
        from feast_mlflow.artifacts import log_feast_artifacts

        log_feast_artifacts(
            mlflow_module=mlflow,
            feature_service_name=metadata.feature_service_name,
            feature_refs=metadata.feature_refs,
            data_sources=metadata.data_sources,
            entity_keys=metadata.entity_keys,
            run_id=run_id,
            project=metadata.project,
        )
    except Exception:
        logger.debug("Could not log rich Feast artifacts", exc_info=True)

    _set_feast_context({
        "store": store,
        "features": features,
        "metadata": metadata,
        "logged": True,
    })


def _on_get_online_features(store, features, entity_rows, result) -> None:
    """Called after get_online_features. Auto-creates trace span and validates."""
    import mlflow

    try:
        from feast_mlflow.providers.feast_provider import FeastProvider
        provider = FeastProvider(store)
        metadata = provider.get_feature_metadata(features)

        try:
            span = mlflow.get_current_active_span()
            if span is not None:
                span.set_attribute("feast.feature_service", metadata.feature_service_name)
                span.set_attribute("feast.feature_count", len(metadata.feature_refs))
        except Exception:
            pass

    except Exception:
        logger.debug("Online features hook: could not extract metadata", exc_info=True)


def _patch_mlflow_log_model(mlflow_module) -> None:
    """Patch common mlflow.<flavor>.log_model() to attach FeatureContract.

    MLflow uses LazyLoader proxies for flavor submodules (e.g. mlflow.pytorch).
    setattr on the proxy doesn't stick because __getattr__ delegates to the real
    module. We force-load the real module via the proxy, then patch it in
    sys.modules directly.
    """
    import sys

    flavors = ["sklearn", "pytorch", "xgboost", "lightgbm", "tensorflow", "keras"]

    for flavor_name in flavors:
        try:
            _proxy = getattr(mlflow_module, flavor_name, None)
            if _proxy is None:
                continue
            # Trigger lazy load so the real module lands in sys.modules
            getattr(_proxy, "log_model", None)
            real_mod = sys.modules.get(f"mlflow.{flavor_name}")
            if real_mod is None or not hasattr(real_mod, "log_model"):
                continue
        except Exception:
            continue

        key = f"log_model:{flavor_name}"
        if key in _originals:
            continue

        original_log_model = real_mod.log_model
        _originals[key] = original_log_model

        def _make_patched(orig, fname):
            def _patched_log_model(*args, **kwargs):
                result = orig(*args, **kwargs)
                try:
                    _on_log_model(fname)
                except Exception:
                    logger.debug("feast-mlflow: error in log_model hook for %s", fname, exc_info=True)
                return result
            return _patched_log_model

        real_mod.log_model = _make_patched(original_log_model, flavor_name)


def _on_log_model(flavor_name: str) -> None:
    """Called after any mlflow.<flavor>.log_model(). If there is Feast context
    from a prior get_historical_features call, log additional lineage."""
    import mlflow

    ctx = _get_feast_context()
    if not ctx or not ctx.get("metadata"):
        return

    active_run = mlflow.active_run()
    if active_run is None:
        return

    metadata = ctx["metadata"]

    if not ctx.get("logged"):
        _log_feast_metadata(
            ctx["store"], ctx["features"], ctx.get("result"), active_run.info.run_id,
        )
        ctx["logged"] = True

    mlflow.set_tag("feast.model_flavor", flavor_name)


# -- load_model patching: auto-attach FeatureContract -----------------------

def _patch_mlflow_load_model(mlflow_module) -> None:
    """Patch mlflow.<flavor>.load_model() to auto-attach FeatureContract.

    When a model is loaded, the bridge:
    1. Looks up the FeatureContract artifact from the model's run
    2. Attaches it as model._feast_contract
    3. Logs a validation summary
    """
    import sys

    flavors = ["sklearn", "pytorch", "xgboost", "lightgbm", "tensorflow", "keras"]

    for flavor_name in flavors:
        try:
            _proxy = getattr(mlflow_module, flavor_name, None)
            if _proxy is None:
                continue
            getattr(_proxy, "load_model", None)
            real_mod = sys.modules.get(f"mlflow.{flavor_name}")
            if real_mod is None or not hasattr(real_mod, "load_model"):
                continue
        except Exception:
            continue

        key = f"load_model:{flavor_name}"
        if key in _originals:
            continue

        original_load_model = real_mod.load_model
        _originals[key] = original_load_model

        def _make_patched_load(orig, fname):
            def _patched_load_model(*args, **kwargs):
                model = orig(*args, **kwargs)
                try:
                    _on_load_model(model, args, kwargs, fname)
                except Exception:
                    logger.debug("feast-mlflow: error in load_model hook for %s", fname, exc_info=True)
                return model
            return _patched_load_model

        real_mod.load_model = _make_patched_load(original_load_model, flavor_name)


def _on_load_model(model, args, kwargs, flavor_name: str) -> None:
    """Called after mlflow.<flavor>.load_model(). Attaches FeatureContract."""
    import mlflow
    import re

    model_uri = args[0] if args else kwargs.get("model_uri", "")
    if not model_uri:
        return

    run_id = None
    match = re.search(r"runs:/([a-f0-9]+)/", str(model_uri))
    if match:
        run_id = match.group(1)

    if not run_id:
        return

    client = mlflow.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id)
    except Exception:
        return

    contract_path = None
    for art in artifacts:
        if art.path == "feature_contract.json":
            contract_path = client.download_artifacts(run_id, art.path)
            break

    if contract_path is None:
        return

    from feast_mlflow.contract import FeatureContract
    contract = FeatureContract.from_file(contract_path)

    model._feast_contract = contract
    model._feast_run_id = run_id

    logger.info(
        "feast-mlflow: attached FeatureContract to model (run=%s, service=%s, %d features)",
        run_id[:8], contract.feature_service, len(contract.features),
    )

    tags = {}
    try:
        run_data = client.get_run(run_id)
        tags = run_data.data.tags
    except Exception:
        pass

    model._feast_feature_service = tags.get("feast.feature_service", contract.feature_service)
    feature_refs_str = tags.get("feast.feature_refs", "")
    model._feast_feature_cols = sorted(r.strip() for r in feature_refs_str.split(",") if r.strip())


# -- Auto-activation on env var ---------------------------------------------

def _auto_activate() -> None:
    from feast_mlflow.config import BridgeConfig
    cfg = BridgeConfig.from_env()
    if cfg.enabled:
        enable()


_auto_activate()
