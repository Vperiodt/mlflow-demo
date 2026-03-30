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

Then use standard Feast APIs plus explicit helpers:

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

### Prerequisites

- **Docker** running on your machine
- **Python 3.10+**

### Setup and Run

```bash
git clone <repo-url> && cd mlflow-demo

# Build the custom Feast Docker image
cd feast-patches && docker buildx build -t feast-mlflow:dev -f Dockerfile.mlflow . && cd ..

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[demo]"

# Run the demo
cd demo
bash scripts/demo.sh
```

### Endpoints

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| Feast UI | http://localhost:8888 |
| Inference API | http://localhost:9000 |

### Test a Prediction

```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_42"}'
```

### Tear Down

```bash
cd demo && docker compose down
```

## 7 Capabilities Demonstrated

1. **Feature-to-Model Lineage** — MLflow tags link runs to Feast feature services
2. **Reproducibility** — Entity DataFrame + feature service logged as artifacts for exact replay
3. **Training/Serving Skew Prevention** — Feature contract validates serving features match training
4. **Experiment Comparison by Feature Set** — MLflow params enable filtering runs by features
5. **Observability** — Feature retrieval duration logged as metrics; inference API returns timing
6. **Dataset Versioning** — Training data logged via `mlflow.log_input()` with digest
7. **Feast UI Lineage** — Feast UI shows MLflow runs connected to feature views
