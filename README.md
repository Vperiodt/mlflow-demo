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

Then write standard Feast + MLflow code. Feast handles the rest:

```python
# Training — zero extra code
store = FeatureStore(repo_path="feast_repo")
df = store.get_historical_features(entity_df, features=fs).to_df()
# ^ auto-logs to MLflow: tags, params, feature_contract.json, lineage HTML

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

