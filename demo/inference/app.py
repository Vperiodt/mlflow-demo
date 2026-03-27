"""
FastAPI inference app: FraudNet predicts using Feast online features.

ZERO special imports. ZERO monkey-patching. Standard Feast + MLflow code.
After loading the model, one Feast helper resolves the feature contract.

Run from demo/ directory:
    uvicorn inference.app:app --port 9000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import pandas as pd
import torch
from feast import FeatureStore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

FEAST_REPO = os.environ.get("FEAST_REPO", str(PROJECT_ROOT / "feast_repo"))

_state: dict = {}


class PredictRequest(BaseModel):
    user_id: str


class PredictResponse(BaseModel):
    user_id: str
    prediction: int
    fraud_probability: float
    features: dict = {}
    run_id: str
    feature_service: str


def _features_to_tensor(features: dict, feature_cols: list[str]) -> torch.Tensor:
    row = {}
    for col in feature_cols:
        val = features.get(col)
        if val is None:
            row[col] = 0.0
        elif isinstance(val, str):
            row[col] = float(pd.Categorical([val]).codes[0])
        else:
            row[col] = float(val)
    return torch.FloatTensor([[row[c] for c in feature_cols]])


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("fraud-detection")
    if exp is None:
        raise RuntimeError("No 'fraud-detection' experiment found in MLflow")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs in 'fraud-detection' experiment")

    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/model"

    # Standard MLflow model loading
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    # One Feast helper call to get the feature contract
    from feast.integrations.mlflow import load_feast_contract_for_model
    contract = load_feast_contract_for_model(model_uri, tracking_uri=tracking_uri)

    feature_service = contract["feature_service"]
    feature_cols = sorted(f["name"] for f in contract["features"])

    store = FeatureStore(repo_path=FEAST_REPO)

    _state["model"] = model
    _state["run_id"] = run.info.run_id
    _state["store"] = store
    _state["feature_service"] = store.get_feature_service(feature_service)
    _state["feature_service_name"] = feature_service
    _state["feature_cols"] = feature_cols
    _state["contract"] = contract

    print(f"Model loaded: run {run.info.run_id}")
    print(f"Feature service: {feature_service} ({len(feature_cols)} features)")
    print(f"FeatureContract: attached ({len(contract['features'])} features)")

    yield
    _state.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Standard Feast + MLflow code. One Feast helper for contract loading.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "run_id": _state.get("run_id"),
        "feature_service": _state.get("feature_service_name"),
        "contract_attached": _state.get("contract") is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    store = _state["store"]
    feature_service = _state["feature_service"]
    model = _state["model"]

    try:
        online_features = store.get_online_features(
            features=feature_service,
            entity_rows=[{"user_id": request.user_id}],
        ).to_dict()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Feast/Redis unavailable: {e}")

    features = {k: v[0] for k, v in online_features.items() if k != "user_id"}
    if all(v is None for v in features.values()):
        raise HTTPException(status_code=404, detail=f"No online features for {request.user_id}")

    tensor = _features_to_tensor(features, _state["feature_cols"])
    with torch.no_grad():
        probability = model(tensor).squeeze().item()

    serialized = {k: v.item() if hasattr(v, "item") else v for k, v in features.items()}

    return PredictResponse(
        user_id=request.user_id,
        prediction=1 if probability >= 0.5 else 0,
        fraud_probability=round(probability, 4),
        features=serialized,
        run_id=_state["run_id"],
        feature_service=_state["feature_service_name"],
    )
