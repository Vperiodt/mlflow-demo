"""
Lineage REST endpoints for the Feast UI lineage tab.

Can run standalone or be mounted into Feast's ui_server.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("feast_mlflow.ui")

app = FastAPI(title="feast-mlflow Lineage API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tracker = None


def _get_tracker():
    global _tracker
    if _tracker is not None:
        return _tracker

    from feast import FeatureStore
    from feast_mlflow.lineage import LineageTracker

    feast_repo = os.environ.get("FEAST_REPO_PATH", "feast_repo")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    store = FeatureStore(repo_path=feast_repo)
    _tracker = LineageTracker(feast_store=store, mlflow_tracking_uri=mlflow_uri)
    return _tracker


@app.get("/api/lineage")
def get_lineage(feature_service: str | None = Query(default=None)):
    tracker = _get_tracker()
    graph = tracker.get_full_lineage(feature_service_name=feature_service)
    return graph.to_dict()


@app.get("/api/lineage/models")
def get_models(feature_service: str = Query(...)):
    tracker = _get_tracker()
    return tracker.get_models_for_feature_service(feature_service)


@app.get("/api/lineage/features")
def get_features(run_id: str = Query(...)):
    tracker = _get_tracker()
    return tracker.get_features_for_run(run_id)


@app.get("/api/lineage/validate")
def validate_contract(run_id: str = Query(...)):
    import mlflow
    from feast import FeatureStore
    from feast_mlflow.contract import FeatureContract, validate_contract as _validate
    from feast_mlflow.providers.feast_provider import FeastProvider

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.MlflowClient()

    artifacts = client.list_artifacts(run_id)
    contract_path = None
    for art in artifacts:
        if art.path == "feature_contract.json":
            contract_path = client.download_artifacts(run_id, art.path)
            break

    if contract_path is None:
        return {"error": "No feature_contract.json found for this run"}

    contract = FeatureContract.from_file(contract_path)

    feast_repo = os.environ.get("FEAST_REPO_PATH", "feast_repo")
    store = FeatureStore(repo_path=feast_repo)
    provider = FeastProvider(store)

    results = _validate(contract, provider)
    return {
        "feature_service": contract.feature_service,
        "run_id": run_id,
        "all_valid": all(r.valid for r in results),
        "features": [
            {
                "name": r.feature_name,
                "dtype_expected": r.dtype_expected,
                "dtype_actual": r.dtype_actual,
                "online_available": r.online_available,
                "schema_match": r.schema_match,
                "valid": r.valid,
            }
            for r in results
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "feast-mlflow-lineage"}
