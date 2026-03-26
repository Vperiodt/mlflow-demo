# Feast + MLflow: Bidirectional Native Integration

**Zero new APIs. Zero code changes. Just config.**

Native, bidirectional integration between [Feast](https://feast.dev) feature stores and [MLflow](https://mlflow.org) experiment tracking — both at training and inference time.

- **Feast side**: Modified Feast SDK with a native `mlflow:` config block. Training calls auto-log to MLflow. Serving calls auto-validate the FeatureContract. MLflow training runs appear as nodes in the Feast UI lineage graph.
- **MLflow side**: `feast_mlflow` plugin package. `autolog()` patches Feast transparently. `load_model()` auto-attaches the FeatureContract. Rich HTML/Markdown artifacts render in the MLflow UI.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                     │
│                                                               │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Registry  │  │  Offline   │  │  Online    │  │  Feast UI  │ │
│  │  Server   │  │  Server    │  │  Server    │  │  + MLflow  │ │
│  │  :6570    │  │  :8815     │  │  :6566     │  │  :8888     │ │
│  └──────────┘  └───────────┘  └───────────┘  └───────────┘ │
│  ┌──────────┐  ┌───────────┐                                 │
│  │  Redis    │  │  MLflow    │                                │
│  │  :6379    │  │  :5000     │                                │
│  └──────────┘  └───────────┘                                 │
└─────────────────────────────────────────────────────────────┘

Training (client → servers):
  get_historical_features()  →  Arrow Flight to Offline Server
  auto-logs to MLflow:  tags, params, FeatureContract, HTML lineage, Markdown

Inference (client → servers):
  load_model()               →  auto-attaches FeatureContract from MLflow run
  get_online_features()      →  auto-validates contract (training-serving match)

Feast UI:
  Lineage graph shows MLflow training runs as blue nodes
  (ui_server injects training run data from MLflow into the registry proto)
```

## Quick Start

### 1. Build the custom Feast image (one-time)

```bash
cd feast-src
docker buildx build -t feast-mlflow:dev -f Dockerfile.mlflow .
```

### 2. Install the MLflow plugin

```bash
pip install -e ".[demo]"
```

### 3. Run the full demo

```bash
cd demo
bash scripts/demo.sh
```

### 4. Explore

| URL | What |
|-----|------|
| http://localhost:8888 | **Feast UI** — lineage graph with MLflow training run nodes |
| http://localhost:5000 | **MLflow UI** — runs with `feast_lineage.html`, contract, and notes |

## What Makes This Different

| | Databricks FeatureEngineeringClient | This Integration |
|---|---|---|
| **Activation** | New client API: `fe.create_training_set()` | Config in `feature_store.yaml` or `autolog()` |
| **Code changes** | Rewrite training and inference code | **Zero** — standard Feast + MLflow code |
| **Architecture** | Single process | **Server/client** (production-ready) |
| **Training** | Auto-logs features | Auto-logs features, contract, HTML lineage |
| **Inference** | No validation | **Auto-validates** FeatureContract at serving time |
| **Feast UI** | No MLflow lineage | **MLflow runs as nodes** in lineage graph |
| **MLflow UI** | Manual tags only | **Auto: HTML graph, Markdown, contract, notes** |
| **Portability** | Locked to Databricks | **Open source**, provider interface |

## How It Works

### Training (automatic)

1. Add `mlflow:` block to `feature_store.yaml`
2. `get_historical_features()` auto-logs feature metadata, schema, and FeatureContract to the active MLflow run
3. `log_model()` attaches deferred Feast context if the run started after the feature fetch

### Inference (automatic)

1. `mlflow.pytorch.load_model(uri)` — bridge auto-downloads FeatureContract and attaches it to the model
2. `get_online_features()` — Feast natively validates the contract against the current registry
3. **No validation code in app.py** — the inference app is pure business logic

### Feast UI lineage

The Feast UI server (`ui_server.py`) queries MLflow for runs tagged with `feast.feature_service` and injects them as `TrainingRunMetadata` into the registry proto before serving it to the browser. The UI TypeScript was modified to render these as blue clickable nodes in the lineage graph.

## Project Structure

```
mlflow-demo/
├── feast-patches/           Feast fork patches (tracked in git)
│   ├── sdk/python/feast/
│   │   ├── repo_config.py            MlflowConfig (follows OpenLineage pattern)
│   │   ├── feature_store.py          Hooks in get_historical/online_features
│   │   ├── ui_server.py              Injects MLflow runs into registry for UI
│   │   ├── mlflow_integration/       Emitter, contract, config
│   │   ├── infra/registry/           Training run storage
│   │   └── lineage/                  Extended with mlflowRun entity type
│   ├── protos/                       TrainingRunMetadata proto
│   ├── ui/src/                       MLflow run nodes in lineage graph
│   └── Dockerfile.mlflow             Multi-stage build (UI from source + Python overlay)
│
├── feast_mlflow/              MLflow plugin package (pip-installable)
│   ├── __init__.py            autolog(), enable(), disable()
│   ├── bridge.py              Patch engine: intercepts Feast + MLflow calls
│   ├── artifacts.py           HTML lineage graph, Markdown summary, run notes
│   ├── contract.py            FeatureContract schema snapshot
│   └── cli.py                 CLI: feast-mlflow validate / lineage
│
├── demo/                      Fraud detection demo
│   ├── docker-compose.yml     6 services: registry, offline, online, Feast UI, Redis, MLflow
│   ├── feast_repo/            Client config (points to remote servers)
│   ├── feast_repo_server/     Server config (local stores, localhost Redis)
│   ├── feast_repo_docker/     Docker config (local stores, docker Redis hostname)
│   ├── training/train.py      Standard Feast + MLflow code (zero glue)
│   ├── inference/app.py       FastAPI serving (zero glue — contract auto-validated)
│   └── scripts/demo.sh        Full pipeline runner
│
├── tests/                     Plugin unit tests
├── pyproject.toml
└── .gitignore
```

## Upstream Contribution Path

**Feast PR**: "Add native MLflow integration (following OpenLineage pattern)"
- `MlflowConfig` on `RepoConfig` — same pattern as `OpenLineageConfig`
- `FeastMlflowEmitter` — auto-logs at training, auto-validates at serving
- `TrainingRunMetadata` proto — training runs in registry
- `ui_server.py` — injects MLflow data into registry for lineage UI
- UI TypeScript — MLflow runs as first-class graph nodes

**MLflow PR**: "Add Feast autologging integration"
- `mlflow.feast.autolog()` — patches `get_historical_features` and `load_model`
- Rich artifacts: HTML lineage, Markdown summary, FeatureContract JSON
- No MLflow UI changes needed — uses existing artifact rendering
