# Feast + MLflow Integration

Native integration between Feast and MLflow. Feast auto-logs feature metadata to MLflow during training and provides helpers for inference — using only standard, public MLflow APIs.

## What It Does

Add `mlflow:` to your `feature_store.yaml`:

```yaml
mlflow:
  enabled: true
  tracking_uri: http://localhost:5000
  auto_log: true
```

Configure `mlflow:` in `feature_store.yaml`, then use standard Feast APIs plus explicit Stage-1 helpers (no separate `feast_mlflow` package in this repo):

```python
# Training — call Feast's autolog helper inside an active MLflow run
from feast.integrations.mlflow_autolog import auto_log_historical_features
# after get_historical_features(...): auto_log_historical_features(...)

# Inference — one helper call
from feast.integrations.mlflow import load_feast_contract_for_model
contract = load_feast_contract_for_model("runs:/abc123/model")
```

**Feast UI** shows MLflow training runs as nodes in the lineage graph.
**MLflow UI** shows feature service, schema, and interactive lineage in artifacts.

## Quick Start

```bash
cd feast-src && docker buildx build -t feast-mlflow:dev -f Dockerfile.mlflow .
cd ../demo && bash scripts/demo.sh
```

Open http://localhost:8888 (Feast UI) and http://localhost:5000 (MLflow UI).

