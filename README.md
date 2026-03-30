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

The demo script handles everything automatically:
1. Generates synthetic fraud transaction data
2. Starts 6 containers (Redis, MLflow, Feast registry/offline/online servers, Feast UI)
3. Registers feature definitions (`feast apply`)
4. Trains a FraudNet model with full MLflow tracking
5. Materializes features to Redis for real-time serving
6. Verifies all 7 integration capabilities (lineage, reproducibility, skew prevention, etc.)
7. Starts an inference API on port 9000

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

## Repository Structure

```
mlflow-demo/
├── feast-patches/           # Patched Feast source files (MLflow integration + UI lineage)
│   ├── Dockerfile.mlflow    # Builds the feast-mlflow:dev Docker image
│   └── sdk/python/feast/
│       ├── integrations/    # MLflow autolog + contract helpers
│       ├── lineage/         # Registry lineage for Feast UI
│       └── ...              # Other patched Feast modules
├── demo/
│   ├── docker-compose.yml   # 6-container stack
│   ├── scripts/
│   │   ├── demo.sh          # One-command demo runner
│   │   └── generate_data.py # Synthetic data generator
│   ├── training/
│   │   ├── train.py         # FraudNet training with Feast + MLflow
│   │   └── model.py         # FraudNet architecture
│   ├── inference/
│   │   └── app.py           # FastAPI serving with skew prevention
│   ├── feast_repo/          # Feast config for local machine (remote store types)
│   └── feast_repo_docker/   # Feast config for inside containers (direct store access)
└── pyproject.toml
```

## Architecture

```
YOUR LOCAL MACHINE                         DOCKER CONTAINERS

                                           ┌─────────────────────┐
train.py ──── gRPC :6570 ────────────────→ │  feast-registry      │
app.py   ──── gRPC :6570 ────────────────→ │  (feature metadata)  │
                                           └─────────────────────┘
                                           ┌─────────────────────┐
train.py ──── Arrow Flight :8815 ────────→ │  feast-offline       │
                                           │  (historical features)│
                                           └─────────────────────┘
                                           ┌─────────────────────┐
app.py   ──── HTTP :6566 ────────────────→ │  feast-online        │──→ Redis :6379
                                           │  (real-time features) │
                                           └─────────────────────┘
                                           ┌─────────────────────┐
train.py ──── HTTP :5000 ────────────────→ │  MLflow              │
app.py   ──── HTTP :5000 ────────────────→ │  (experiment tracking)│
                                           └─────────────────────┘
```

## 7 Capabilities Demonstrated

1. **Feature-to-Model Lineage** — MLflow tags link runs to Feast feature services
2. **Reproducibility** — Entity DataFrame + feature service logged as artifacts for exact replay
3. **Training/Serving Skew Prevention** — Feature contract validates serving features match training
4. **Experiment Comparison by Feature Set** — MLflow params enable filtering runs by features
5. **Observability** — Feature retrieval duration logged as metrics; inference API returns timing
6. **Dataset Versioning** — Training data logged via `mlflow.log_input()` with digest
7. **Feast UI Lineage** — Feast UI shows MLflow runs connected to feature views
