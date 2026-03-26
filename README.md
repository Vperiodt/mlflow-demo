# Feast + MLflow: Native Integration

## The Problem

[Feast](https://feast.dev) manages features. [MLflow](https://mlflow.org) tracks experiments. Today they don't talk to each other. Data scientists manually copy feature metadata into MLflow tags, write glue scripts for lineage, and have no way to validate that the features a model was trained on still match what's being served.

Databricks solved this for their platform with `FeatureEngineeringClient`, but it requires rewriting your code and locks you into Databricks.

## The Solution

This project makes Feast and MLflow natively aware of each other — **in both directions** — with zero code changes to training or inference scripts.

**From the Feast side** (modified Feast SDK):
- Add `mlflow:` to `feature_store.yaml` and Feast auto-logs feature metadata to MLflow
- MLflow training runs appear as nodes in the Feast UI lineage graph
- `get_online_features()` auto-validates that served features match the training contract

**From the MLflow side** (`feast_mlflow` pip package):
- Call `feast_mlflow.autolog()` and standard Feast calls automatically log to MLflow
- `load_model()` auto-attaches the feature schema to the model
- Rich HTML lineage graphs and Markdown summaries render in MLflow's artifact viewer

## What the Demo Shows

Run `bash scripts/demo.sh` from `demo/` and you get:

**Feast UI** (http://localhost:8888) — open the Lineage page:

```
Data Source  →  Feature View  →  Feature Service  →  MLflow Run (blue node)
```

Each MLflow run that used a feature service appears as a clickable blue node in the lineage graph. Click it to jump to the run in MLflow.

**MLflow UI** (http://localhost:5000) — open any run:

- **Overview**: Feast tags (`feast.feature_service`, `feast.feature_refs`, etc.), params (`feast.num_features`), and a description note showing feature service + project
- **Artifacts**: `feast_lineage.html` (interactive graph), `feast_lineage.md` (Markdown summary), `feature_contract.json` (full schema snapshot for validation)
- **Datasets**: `feast:fraud_feature_service (Training)` — logged automatically

**Inference API** (http://localhost:9000):

```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001"}'
```

The model loads from MLflow with the FeatureContract auto-attached. Online features are fetched from Feast with the contract auto-validated. The app code has zero awareness of the integration — it's pure business logic.

## Quick Start

```bash
# 1. Build the custom Feast image (includes UI with MLflow lineage nodes)
cd feast-src
docker buildx build -t feast-mlflow:dev -f Dockerfile.mlflow .

# 2. Install the MLflow plugin
cd ..
pip install -e ".[demo]"

# 3. Run the demo (starts 6 Docker services, trains model, runs full pipeline)
cd demo
bash scripts/demo.sh
```

## Architecture

Six Docker services, one image:

| Service | Port | Role |
|---------|------|------|
| **Registry Server** | 6570 | Stores Feast metadata (entities, feature views, feature services). gRPC. |
| **Offline Server** | 8815 | Historical feature retrieval (training). Arrow Flight. |
| **Online Server** | 6566 | Low-latency feature serving (inference). HTTP, backed by Redis. |
| **Feast UI** | 8888 | Lineage graph, feature exploration. Queries MLflow for training runs. |
| **Redis** | 6379 | Online feature store backend. |
| **MLflow** | 5000 | Experiment tracking, model registry, artifact store. |

Training scripts run on the host and talk to the Docker servers as a Feast client (`feature_store.yaml` with `type: remote`).

## How It Works

### During Training

```python
# Standard code — no special imports or APIs beyond activating the bridge
store = FeatureStore(repo_path="feast_repo")
training_df = store.get_historical_features(entity_df, features=feature_service).to_df()
#                    ↑ bridge intercepts: logs tags, params, schema to MLflow

with mlflow.start_run():
    model = train(training_df)
    mlflow.pytorch.log_model(model, "model")
    #               ↑ bridge intercepts: logs FeatureContract, HTML lineage, Markdown
```

### During Inference

```python
model = mlflow.pytorch.load_model(model_uri)
#        ↑ bridge intercepts: downloads FeatureContract from run, attaches to model

features = store.get_online_features(features=feature_service, entity_rows=[...])
#           ↑ Feast intercepts: validates contract against current registry
```

### In the Feast UI

The Feast UI server (`ui_server.py`) queries MLflow at startup for any runs tagged with `feast.feature_service`. It injects these as `TrainingRunMetadata` into the registry protobuf before serving it to the browser. The React UI (rebuilt from modified TypeScript) renders them as blue "MLflow Run" nodes in the lineage graph.

### The Plugin Mechanism

The `feast_mlflow` package works by **monkey-patching** — replacing function references in Python's memory at runtime. When you `import feast_mlflow`, it swaps `feast.FeatureStore.get_historical_features`, `mlflow.pytorch.log_model`, and `mlflow.pytorch.load_model` with thin wrappers that call the original function first, then log metadata. No MLflow or Feast source files are modified on disk.

The Feast fork (`feast-patches/`) takes the native approach — the same logic is built directly into the Feast SDK, activated by a `mlflow:` config block. This is the code intended for upstream contribution.

## Project Structure

```
mlflow-demo/
│
├── feast_mlflow/              MLflow-side plugin (pip install, no fork needed)
│   ├── bridge.py              Monkey-patches Feast + MLflow at runtime
│   ├── artifacts.py           Generates HTML lineage, Markdown, run notes
│   ├── contract.py            FeatureContract (training schema snapshot)
│   ├── cli.py                 CLI: feast-mlflow validate / lineage
│   ├── dataset.py             MLflow dataset integration
│   ├── lineage.py             Bidirectional lineage tracker
│   ├── config.py              Config from YAML / env vars
│   └── providers/             Feature store provider interface
│
├── feast-patches/             Feast-side fork patches (for upstream PR)
│   ├── Dockerfile.mlflow      Multi-stage: builds UI from source + overlays Python
│   ├── sdk/python/feast/
│   │   ├── repo_config.py     Adds MlflowConfig (follows OpenLineage pattern)
│   │   ├── feature_store.py   Hooks in get_historical/online_features
│   │   ├── ui_server.py       Injects MLflow runs into /registry endpoint
│   │   ├── mlflow_integration/  Native emitter, contract, config
│   │   ├── infra/registry/    Training run storage in file registry
│   │   └── lineage/           mlflowRun entity type in lineage generator
│   ├── protos/                TrainingRunMetadata proto message
│   └── ui/src/                TypeScript: MLflow run nodes in lineage graph
│
├── demo/                      Fraud detection end-to-end demo
│   ├── docker-compose.yml     6-service stack
│   ├── feast_repo/            Client config (remote servers on localhost)
│   ├── feast_repo_server/     Server config (local stores, host Redis)
│   ├── feast_repo_docker/     Docker config (local stores, Docker Redis)
│   ├── training/train.py      FraudNet training (standard Feast + MLflow)
│   ├── inference/app.py       FastAPI serving (zero glue code)
│   └── scripts/demo.sh        Full pipeline: infra → data → train → validate → serve
│
├── tests/                     Unit tests for the plugin
├── pyproject.toml             Package metadata
└── feast-src/                 Full Feast clone for Docker builds (gitignored)
```

## Comparison

| | Databricks FeatureEngineeringClient | This Project |
|---|---|---|
| **Activation** | New API: `fe.create_training_set()` | Config block or `autolog()` |
| **Code changes** | Rewrite training + inference | Zero |
| **Training** | Auto-logs features | Auto-logs features + contract + HTML lineage |
| **Inference** | No contract validation | Auto-validates FeatureContract |
| **Feast UI** | N/A | MLflow runs in lineage graph |
| **MLflow UI** | Manual tags | Auto: HTML graph, Markdown, contract, notes |
| **Architecture** | Single process, Databricks only | Server/client, open source |

## Upstream Path

- **Feast PR**: native `mlflow:` config following the OpenLineage pattern, `FeastMlflowEmitter`, `TrainingRunMetadata` proto, UI lineage nodes
- **MLflow PR**: `mlflow.feast.autolog()` integration module, rich artifacts, no UI changes needed
