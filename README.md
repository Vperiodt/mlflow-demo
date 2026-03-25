# Feast + MLflow: Bidirectional Native Integration

**Zero new APIs. Zero code changes. Just config.**

This project demonstrates **bidirectional** native integration between Feast feature stores and MLflow experiment tracking:

- **Feast side**: Modified Feast SDK with native `mlflow:` config block, auto-logging to MLflow, and MLflow training runs visible in the Feast UI lineage graph
- **MLflow side**: `feast_mlflow` plugin with `autolog()`, rich HTML/Markdown artifacts that render in MLflow's artifact viewer

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                       │
│                                                               │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Registry  │  │  Offline   │  │  Online    │  │  Feast UI  │ │
│  │  Server   │  │  Server    │  │  Server    │  │  :8888     │ │
│  │  :6570    │  │  :8815     │  │  :6566     │  │            │ │
│  └──────────┘  └───────────┘  └───────────┘  └───────────┘ │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐                 │
│  │  Redis    │  │  MLflow    │  │ Lineage UI │                │
│  │  :6379    │  │  :5000     │  │  :9090     │                │
│  └──────────┘  └───────────┘  └───────────┘                 │
└─────────────────────────────────────────────────────────────┘
         ▲                                  ▲
         │      Feast Client (training)     │
         │   get_historical_features ──────►│
         │   (Arrow Flight to offline srv)  │
         │                                  │
         │   Auto-logs to MLflow:           │
         │   - Feature tags & params        │
         │   - FeatureContract JSON         │
         │   - HTML lineage graph           │
         │   - Markdown summary             │
         └──────────────────────────────────┘
```

## Quick Start

### 1. Build the custom Feast image (one-time)

```bash
cd feast-src
docker buildx build -t feast-mlflow:dev -f Dockerfile.mlflow .
```

### 2. Run the full demo

```bash
cd demo
bash scripts/demo.sh
```

This starts the full infrastructure, generates data, trains a model, validates the feature contract, and opens all UIs.

### 3. Explore

| URL | What |
|-----|------|
| http://localhost:8888 | **Feast UI** — lineage graph with MLflow run nodes |
| http://localhost:9090 | **Lineage UI** — interactive Mermaid graph showing data flow |
| http://localhost:5000 | **MLflow UI** — runs with Feast metadata, artifacts, and notes |

## What Makes This Different

| | Databricks FeatureEngineeringClient | This Integration |
|---|---|---|
| **Activation** | New client API: `fe.create_training_set()` | Config in `feature_store.yaml` |
| **Code changes** | Rewrite training/inference | **Zero** — standard Feast + MLflow |
| **Architecture** | Single process | **Server/client** (production-ready) |
| **Feast UI** | No lineage | **MLflow runs in lineage graph** |
| **MLflow UI** | Manual tags | **Auto: HTML lineage, contract, notes** |
| **Validation** | None | **FeatureContract** — training-serving consistency |
| **Portability** | Locked to Databricks | **Open source**, plugin interface |

## How It Works

### Feast Side (native in SDK)

1. Add `mlflow:` block to `feature_store.yaml`
2. `FeatureStore.get_historical_features()` auto-logs feature metadata to MLflow
3. Training runs are recorded in the Feast registry for lineage
4. Feast UI lineage page shows MLflow runs as graph nodes

### MLflow Side (plugin package)

```python
import feast_mlflow
feast_mlflow.autolog()  # one-liner, MLflow convention
```

Auto-logs:
- `feast_lineage.html` — interactive lineage graph (renders in MLflow artifact viewer)
- `feast_lineage.md` — Markdown summary
- `feature_contract.json` — schema snapshot for validation
- `mlflow.note.content` — Feast summary on run overview page

## Project Structure

```
mlflow-demo/
├── feast-src/              Modified Feast fork (the upstream PR)
│   ├── sdk/python/feast/
│   │   ├── repo_config.py          MlflowConfig (follows OpenLineage pattern)
│   │   ├── feature_store.py        Native MLflow hooks
│   │   ├── mlflow_integration/     Emitter, contract, config
│   │   └── lineage/                Extended with mlflowRun nodes
│   ├── protos/                     TrainingRunMetadata proto
│   ├── ui/src/                     MLflow run nodes in lineage graph
│   └── Dockerfile.mlflow           Custom image build
├── feast_mlflow/           MLflow plugin package
│   ├── bridge.py           Patching engine (standalone mode)
│   ├── artifacts.py        HTML lineage, Markdown, run notes
│   ├── contract.py         FeatureContract
│   └── cli.py              validate, lineage commands
├── demo/                   Fraud detection demo
│   ├── docker-compose.yml  Full 7-service stack
│   ├── feast_repo/         Client config (remote servers)
│   ├── feast_repo_server/  Server config (local stores)
│   ├── training/           FraudNet (standard code)
│   ├── inference/          FastAPI serving
│   └── scripts/            Demo runner, data gen, materialize
└── pyproject.toml
```

## Upstream Contribution Path

- **Feast PR**: "Add native MLflow integration (following OpenLineage pattern)" — config, SDK hooks, registry proto, lineage, UI
- **MLflow PR**: "Add Feast autologging integration" — `mlflow.feast.autolog()` as a new integration module with rich artifacts
