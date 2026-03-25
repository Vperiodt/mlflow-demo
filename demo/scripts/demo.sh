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
echo "Starting: Redis, MLflow, Feast Registry/Offline/Online servers, Feast UI, Lineage UI..."
docker compose up -d
echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "=== Step 2: Generate synthetic data ==="
"$PYTHON" scripts/generate_data.py

echo ""
echo "=== Step 3: Apply Feast definitions ==="
echo "(Applying to local registry first, then server picks it up)"
cd feast_repo_server && feast apply && cd ..

echo ""
echo "=== Step 4: Train FraudNet (native Feast-MLflow integration) ==="
echo "Training uses the feast_mlflow bridge (autolog). Feast's native integration"
echo "will also log feature metadata when using server/client mode."
cd feast_repo && "$PYTHON" ../training/train.py && cd ..

echo ""
echo "=== Step 5: Materialize features to Redis ==="
"$PYTHON" scripts/materialize.py feast_repo_server

echo ""
echo "=== Step 6: Validate training-serving consistency ==="
RUN_ID=$("$PYTHON" -c "
import mlflow
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
c = mlflow.MlflowClient()
exp = c.get_experiment_by_name('fraud-detection')
runs = c.search_runs([exp.experiment_id], order_by=['start_time DESC'], max_results=1)
print(runs[0].info.run_id)
")
echo "Latest run: $RUN_ID"
feast-mlflow --feast-repo feast_repo --mlflow-uri "$MLFLOW_TRACKING_URI" validate --run-id "$RUN_ID"

echo ""
echo "=== Step 7: Query lineage ==="
feast-mlflow --feast-repo feast_repo --mlflow-uri "$MLFLOW_TRACKING_URI" lineage --feature-service fraud_feature_service

echo ""
echo "========================================================"
echo "  Demo Complete! Open these URLs:"
echo ""
echo "  Feast UI:        http://localhost:8888"
echo "  Lineage Graph:   http://localhost:9090"
echo "  MLflow UI:       http://localhost:5000"
echo ""
echo "  In MLflow, open the run and check artifacts for:"
echo "    - feast_lineage.html  (interactive lineage graph)"
echo "    - feast_lineage.md    (Markdown summary)"
echo "    - feature_contract.json (schema snapshot)"
echo "========================================================"

echo ""
echo "=== Step 8: Start inference API ==="
echo "Starting on port 9000 (Ctrl+C to stop)..."
uvicorn inference.app:app --port 9000
