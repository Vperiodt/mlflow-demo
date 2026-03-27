#!/usr/bin/env bash
# Full demo pipeline -- Feast server/client architecture with native MLflow integration.
#
# Prerequisites: Docker running, feast-mlflow:dev image built
# Run from demo/ directory:   bash scripts/demo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DEMO_DIR"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export FEAST_MLFLOW=1

PYTHON="$(command -v python3 2>/dev/null || command -v python)"
if [ -z "$PYTHON" ]; then
  echo "Neither python3 nor python found. Install Python 3." >&2
  exit 1
fi

echo "========================================================"
echo "  Feast + MLflow Native Integration Demo"
echo "  Server/Client Architecture"
echo "========================================================"

echo ""
echo "=== Step 1: Generate synthetic data ==="
"$PYTHON" scripts/generate_data.py

echo ""
echo "=== Step 2: Start infrastructure ==="
echo "Starting: Redis, MLflow, Feast servers, Feast UI..."
docker compose down 2>/dev/null || true
docker compose up -d
echo "Waiting for services to be ready..."
sleep 15

echo ""
echo "=== Step 3: Apply Feast definitions ==="
docker compose run --rm -T registry-server feast -c /feature_repo apply

echo ""
echo "=== Step 4: Train FraudNet ==="
echo "Zero glue code -- bridge auto-logs Feast metadata to MLflow."
"$PYTHON" training/train.py

echo ""
echo "=== Step 5: Materialize features to Redis ==="
DATES=$("$PYTHON" << 'PYEOF'
import pandas as pd
df = pd.read_parquet("data/parquet/transactions.parquet", columns=["event_timestamp"])
ts = pd.to_datetime(df["event_timestamp"])
print(f"{ts.min().strftime('%Y-%m-%dT%H:%M:%S')} {ts.max().strftime('%Y-%m-%dT%H:%M:%S')}")
PYEOF
)
START_DATE=$(echo "$DATES" | awk '{print $1}')
END_DATE=$(echo "$DATES" | awk '{print $2}')
echo "Materializing from $START_DATE to $END_DATE"
docker compose run --rm -T online-server feast -c /feature_repo materialize "$START_DATE" "$END_DATE"

echo ""
echo "=== Step 6: Validate FeatureContract ==="
RUN_ID=$("$PYTHON" -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
c = mlflow.MlflowClient()
exp = c.get_experiment_by_name('fraud-detection')
runs = c.search_runs([exp.experiment_id], order_by=['start_time DESC'], max_results=1)
print(runs[0].info.run_id)
")
echo "Latest run: $RUN_ID"
echo "FeatureContract artifact logged to MLflow (check Artifacts tab in UI)"
echo "Validating contract programmatically..."
"$PYTHON" -c "
import mlflow, json
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
c = mlflow.MlflowClient()
path = c.download_artifacts('$RUN_ID', 'feature_contract.json')
contract = json.load(open(path))
print(f\"  Feature Service: {contract['feature_service']}\")
print(f\"  Project: {contract['feast_project']}\")
print(f\"  Features: {len(contract['features'])}\")
for f in contract['features']:
    print(f\"    {f['name']} ({f['dtype']}) from {f['source_view']}\")
print(f\"  Entity Keys: {contract['entity_keys']}\")
print(f\"  Data Sources: {contract['data_sources']}\")
print('  Status: CONTRACT VALID')
"

echo ""
echo "=== Step 7: Query lineage ==="
echo "Querying MLflow for runs using Feast features..."
"$PYTHON" << 'PYEOF'
import mlflow, os
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = mlflow.MlflowClient()
for exp in c.search_experiments():
    runs = c.search_runs([exp.experiment_id], max_results=10)
    for r in runs:
        tags = r.data.tags
        fs = tags.get("feast.feature_service", "")
        if not fs:
            continue
        refs = tags.get("feast.feature_refs", "").split(",")
        print(f"  Run {r.info.run_id[:12]} | Experiment: {exp.name} | Feature Service: {fs} | Features: {len(refs)}")
PYEOF

echo ""
echo "=== Step 8: Restart Feast UI to pick up MLflow runs ==="
docker compose restart feast-ui
sleep 5

echo ""
echo "========================================================"
echo "  Demo Complete!"
echo ""
echo "  Feast UI:  http://localhost:8888  (lineage with MLflow runs)"
echo "  MLflow UI: http://localhost:5000  (artifacts, contract, lineage)"
echo "========================================================"

echo ""
echo "=== Step 9: Start inference API ==="
echo "Starting on port 9000 (Ctrl+C to stop)..."
uvicorn inference.app:app --port 9000
