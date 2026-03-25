"""
CLI for feast-mlflow: lineage, validate, ui commands.

Usage:
    feast-mlflow lineage --feature-service fraud_feature_service
    feast-mlflow lineage --run-id abc123
    feast-mlflow validate --run-id abc123
    feast-mlflow ui --feast-repo feast_repo --mlflow-uri http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _lineage_cmd(args):
    from feast import FeatureStore
    from feast_mlflow.lineage import LineageTracker

    store = FeatureStore(repo_path=args.feast_repo)
    tracker = LineageTracker(feast_store=store, mlflow_tracking_uri=args.mlflow_uri)

    if args.run_id:
        features = tracker.get_features_for_run(args.run_id)
        if not features.get("feature_service"):
            print(f"No Feast metadata found for run {args.run_id}", file=sys.stderr)
            sys.exit(1)
        print(f"Run: {args.run_id}")
        print(f"Feature Service: {features['feature_service']}")
        print(f"Project: {features['project']}")
        print(f"Entity Keys: {', '.join(features['entity_keys'])}")
        print(f"Features ({len(features['feature_refs'])}):")
        for ref in features["feature_refs"]:
            print(f"  - {ref}")
        print(f"Data Sources:")
        for ds in features["data_sources"]:
            print(f"  - {ds}")
    elif args.feature_service:
        models = tracker.get_models_for_feature_service(args.feature_service)
        print(f"Feature Service: {args.feature_service}")
        print(f"Models using this service ({len(models)}):")
        for m in models:
            metrics_str = ", ".join(f"{k}={v}" for k, v in list(m["metrics"].items())[:3])
            print(f"  Run {m['run_id'][:8]} | {m['experiment_name']} | {m['status']} | {metrics_str}")
    else:
        graph = tracker.get_full_lineage()
        print(json.dumps(graph.to_dict(), indent=2))


def _validate_cmd(args):
    import mlflow
    from feast import FeatureStore
    from feast_mlflow.contract import FeatureContract, validate_contract
    from feast_mlflow.providers.feast_provider import FeastProvider

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.MlflowClient()

    run_id = args.run_id
    if not run_id and args.model_uri:
        parts = args.model_uri.split("/")
        for i, p in enumerate(parts):
            if p == "runs:" and i + 1 < len(parts):
                run_id = parts[i + 1]
                break

    if not run_id:
        print("Provide --run-id or --model-uri", file=sys.stderr)
        sys.exit(1)

    artifacts = client.list_artifacts(run_id)
    contract_path = None
    for art in artifacts:
        if art.path == "feature_contract.json":
            contract_path = client.download_artifacts(run_id, art.path)
            break

    if contract_path is None:
        print(f"No feature_contract.json found for run {run_id}", file=sys.stderr)
        sys.exit(1)

    contract = FeatureContract.from_file(contract_path)
    store = FeatureStore(repo_path=args.feast_repo)
    provider = FeastProvider(store)
    results = validate_contract(contract, provider)

    print(f"Feature Contract: {contract.feature_service}")
    print(f"{'Feature':<30} {'Expected':<12} {'Actual':<12} {'Online':<10} {'Schema':<10}")
    print("-" * 74)
    for r in results:
        online = "yes" if r.online_available else "NO"
        schema = "match" if r.schema_match else "MISMATCH"
        print(f"  {r.feature_name:<28} {r.dtype_expected:<12} {r.dtype_actual:<12} {online:<10} {schema:<10}")

    all_valid = all(r.valid for r in results)
    print()
    if all_valid:
        print("Result: ALL FEATURES VALID -- safe to serve")
    else:
        print("Result: VALIDATION FAILED -- check mismatches above")
        sys.exit(1)


def _ui_cmd(args):
    import uvicorn

    os.environ["FEAST_REPO_PATH"] = args.feast_repo
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri

    print(f"Starting feast-mlflow lineage UI server on port {args.port}")
    print(f"  Feast repo: {args.feast_repo}")
    print(f"  MLflow URI: {args.mlflow_uri}")
    print(f"  Endpoints:")
    print(f"    GET http://localhost:{args.port}/api/lineage")
    print(f"    GET http://localhost:{args.port}/api/lineage/models?feature_service=...")
    print(f"    GET http://localhost:{args.port}/api/lineage/features?run_id=...")
    print(f"    GET http://localhost:{args.port}/api/lineage/validate?run_id=...")

    uvicorn.run("feast_mlflow.ui.server:app", host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        prog="feast-mlflow",
        description="Invisible bridge between Feast and MLflow",
    )
    parser.add_argument(
        "--feast-repo", default=os.environ.get("FEAST_REPO_PATH", "feast_repo"),
        help="Path to Feast repo (default: feast_repo)",
    )
    parser.add_argument(
        "--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI",
    )

    sub = parser.add_subparsers(dest="command")

    lineage_p = sub.add_parser("lineage", help="Query lineage")
    lineage_group = lineage_p.add_mutually_exclusive_group()
    lineage_group.add_argument("--feature-service", help="Show models for a feature service")
    lineage_group.add_argument("--run-id", help="Show features for an MLflow run")

    validate_p = sub.add_parser("validate", help="Validate training-serving consistency")
    validate_p.add_argument("--run-id", help="MLflow run ID")
    validate_p.add_argument("--model-uri", help="MLflow model URI")

    ui_p = sub.add_parser("ui", help="Start lineage API server")
    ui_p.add_argument("--host", default="0.0.0.0")
    ui_p.add_argument("--port", type=int, default=8889)

    args = parser.parse_args()

    if args.command == "lineage":
        _lineage_cmd(args)
    elif args.command == "validate":
        _validate_cmd(args)
    elif args.command == "ui":
        _ui_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
