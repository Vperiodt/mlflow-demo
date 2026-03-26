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
echo "=== Step 1: Start infrastructure ==="
echo "Starting: Redis, MLflow, Feast servers, Feast UI..."
docker compose up -d
echo "Waiting for services to be ready..."
sleep 15

echo ""
echo "=== Step 2: Generate synthetic data ==="
"$PYTHON" scripts/generate_data.py

echo ""
echo "=== Step 3: Apply Feast definitions ==="
docker compose run --rm -T registry-server feast -c /feature_repo apply

echo ""
echo "=== Step 4: Train FraudNet ==="
echo "Zero glue code -- bridge auto-logs Feast metadata to MLflow."
"$PYTHON" training/train.py

echo ""
echo "=== Step 5: Materialize features to Redis ==="
docker compose run --rm -T online-server feast -c /feature_repo materialize \
  "$(date -u -v-30d +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '30 days ago' +%Y-%m-%dT%H:%M:%S)" \
  "$(date -u +%Y-%m-%dT%H:%M:%S)"

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
feast-mlflow --feast-repo feast_repo_server --mlflow-uri "$MLFLOW_TRACKING_URI" validate --run-id "$RUN_ID"

echo ""
echo "=== Step 7: Query lineage ==="
feast-mlflow --feast-repo feast_repo_server --mlflow-uri "$MLFLOW_TRACKING_URI" lineage --feature-service fraud_feature_service

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
