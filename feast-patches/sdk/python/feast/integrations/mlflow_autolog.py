"""
Automatic MLflow logging for Feast — called internally by FeatureStore.

When ``mlflow:`` is configured in ``feature_store.yaml`` with ``auto_log: true``,
Feast's ``get_historical_features()`` automatically logs all feature metadata
to MLflow using only standard MLflow APIs.

No monkey-patching. No import tricks. No replacing functions in memory.
This is native Feast code that calls ``mlflow.set_tag()``, ``mlflow.log_param()``,
``mlflow.log_artifact()``, and ``mlflow.log_input()`` — all public, stable APIs.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("feast.integrations.mlflow_autolog")


def _is_mlflow_available() -> bool:
    try:
        import mlflow
        return True
    except ImportError:
        return False


def _get_active_run():
    try:
        import mlflow
        return mlflow.active_run()
    except Exception:
        return None


def auto_log_historical_features(
    *,
    feature_service_name: str,
    project: str,
    feature_refs: List[str],
    feature_views: List[Any],
    entity_keys: List[str],
    data_sources: List[str],
    duration_seconds: float = 0.0,
    entity_count: int = 0,
    tracking_uri: str = "",
) -> None:
    """Auto-log everything to the active MLflow run after get_historical_features.

    Called internally by FeatureStore. Logs:
    - Tags: feast.feature_service, feast.feature_refs, feast.entity_keys, etc.
    - Params: feast.feature_service, feast.num_features, feast.num_entities
    - Metric: feast.retrieval_duration_sec
    - Artifact: feature_contract.json (schema snapshot)
    - Artifact: feast_lineage.html (interactive graph)
    - Artifact: feast_lineage.md (Markdown summary)
    - Artifact: feast/schema.json (feature schema)
    - Tag: mlflow.note.content (run description with Feast context)
    - Dataset: feast:{feature_service_name} via mlflow.log_input()
    """
    if not _is_mlflow_available():
        return

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    active_run = mlflow.active_run()
    if active_run is None:
        return

    run_id = active_run.info.run_id

    # --- Tags ---
    try:
        mlflow.set_tag("feast.feature_service", feature_service_name)
        mlflow.set_tag("feast.project", project)
        mlflow.set_tag("feast.feature_refs", ",".join(feature_refs))
        mlflow.set_tag("feast.entity_keys", ",".join(entity_keys))
        mlflow.set_tag("feast.data_sources", ",".join(data_sources))
        mlflow.set_tag("feast.retrieval_type", "historical")
    except Exception:
        logger.debug("Could not log Feast tags", exc_info=True)

    # --- Params ---
    try:
        mlflow.log_params({
            "feast.feature_service": feature_service_name,
            "feast.num_features": len(feature_refs),
            "feast.num_entities": len(entity_keys),
            "feast.entity_count": entity_count,
        })
    except Exception:
        logger.debug("Could not log Feast params", exc_info=True)

    # --- Metric ---
    if duration_seconds > 0:
        try:
            mlflow.log_metric("feast.retrieval_duration_sec", round(duration_seconds, 3))
        except Exception:
            pass

    # --- Feature Contract artifact ---
    try:
        contract = _build_contract(
            feature_service_name=feature_service_name,
            project=project,
            run_id=run_id,
            feature_refs=feature_refs,
            entity_keys=entity_keys,
            data_sources=data_sources,
            feature_views=feature_views,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feature_contract.json"
            path.write_text(json.dumps(contract, indent=2))
            mlflow.log_artifact(str(path))
    except Exception:
        logger.debug("Could not log FeatureContract", exc_info=True)

    # --- HTML lineage artifact ---
    try:
        html = _build_lineage_html(
            feature_service_name=feature_service_name,
            feature_refs=feature_refs,
            data_sources=data_sources,
            entity_keys=entity_keys,
            run_id=run_id,
            project=project,
            feature_views=feature_views,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feast_lineage.html"
            path.write_text(html)
            mlflow.log_artifact(str(path))
    except Exception:
        logger.debug("Could not log lineage HTML", exc_info=True)

    # --- Markdown summary artifact ---
    try:
        md = _build_lineage_markdown(
            feature_service_name=feature_service_name,
            feature_refs=feature_refs,
            data_sources=data_sources,
            entity_keys=entity_keys,
            run_id=run_id,
            project=project,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feast_lineage.md"
            path.write_text(md)
            mlflow.log_artifact(str(path))
    except Exception:
        logger.debug("Could not log lineage Markdown", exc_info=True)

    # --- Feature schema artifact ---
    try:
        schema = _build_schema(feature_views)
        if schema:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="feast_schema_"
            ) as f:
                json.dump(schema, f, indent=2)
                schema_path = f.name
            mlflow.log_artifact(schema_path, artifact_path="feast")
            Path(schema_path).unlink(missing_ok=True)
    except Exception:
        logger.debug("Could not log schema artifact", exc_info=True)

    # --- Run description note ---
    try:
        note = (
            f'<div style="padding:12px;background:#161b22;border:1px solid #30363d;'
            f'border-radius:8px;font-family:sans-serif">'
            f'<b style="color:#58a6ff">Feast Features</b><br>'
            f'<span style="color:#8b949e">Service:</span> {feature_service_name} | '
            f'<span style="color:#8b949e">Project:</span> {project} | '
            f'{len(feature_refs)} features, {len(entity_keys)} entities'
            f"</div>"
        )
        mlflow.set_tag("mlflow.note.content", note)
    except Exception:
        pass

    # --- Dataset logging ---
    try:
        from mlflow.data.pandas_dataset import from_pandas
        import pandas as pd
        meta_df = pd.DataFrame({"feature_ref": feature_refs})
        dataset = from_pandas(meta_df, name=f"feast:{feature_service_name}")
        mlflow.log_input(dataset, context="training")
    except Exception:
        logger.debug("Could not log MLflow dataset", exc_info=True)


def load_feast_contract_for_model(
    model_uri: str,
    tracking_uri: str = "",
) -> Optional[Dict[str, Any]]:
    """Load the FeatureContract from an MLflow model's run artifacts.

    Call after loading a model to get the feature schema it was trained on.

    Args:
        model_uri: MLflow model URI (e.g. "runs:/<run_id>/model")
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        FeatureContract dict, or None if not found.

    Example::

        model = mlflow.pytorch.load_model(model_uri)
        contract = load_feast_contract_for_model(model_uri)
        if contract:
            print(f"Model expects {len(contract['features'])} features from {contract['feature_service']}")
    """
    if not _is_mlflow_available():
        return None

    import mlflow
    import re

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    match = re.search(r"runs:/([a-f0-9]+)/", model_uri)
    if not match:
        return None
    run_id = match.group(1)

    try:
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        for art in artifacts:
            if art.path == "feature_contract.json":
                local = client.download_artifacts(run_id, art.path)
                return json.loads(Path(local).read_text())
    except Exception:
        logger.debug("Could not load FeatureContract for model", exc_info=True)

    return None


# --- Internal builders --------------------------------------------------------

def _build_contract(
    *,
    feature_service_name: str,
    project: str,
    run_id: str,
    feature_refs: List[str],
    entity_keys: List[str],
    data_sources: List[str],
    feature_views: List[Any],
) -> dict:
    from datetime import datetime, timezone

    features = []
    for fv in feature_views:
        fv_name = fv.name if hasattr(fv, "name") else str(fv)
        if hasattr(fv, "features"):
            for feat in fv.features:
                features.append({
                    "name": feat.name,
                    "dtype": str(feat.dtype),
                    "source_view": fv_name,
                })

    return {
        "contract_version": "1.0",
        "feature_service": feature_service_name,
        "feast_project": project,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
        "features": features,
        "entity_keys": entity_keys,
        "data_sources": data_sources,
        "statistics": {},
    }


def _build_schema(feature_views: List[Any]) -> list:
    schema = []
    for fv in feature_views:
        fv_name = fv.name if hasattr(fv, "name") else str(fv)
        if hasattr(fv, "features"):
            for feat in fv.features:
                schema.append({
                    "name": feat.name,
                    "dtype": str(feat.dtype),
                    "source_view": fv_name,
                })
    return schema


def _build_lineage_html(
    *,
    feature_service_name: str,
    feature_refs: List[str],
    data_sources: List[str],
    entity_keys: List[str],
    run_id: str,
    project: str,
    feature_views: List[Any],
) -> str:
    fv_names = set()
    for fv in feature_views:
        fv_names.add(fv.name if hasattr(fv, "name") else str(fv))

    lines = ["graph LR"]
    for ds in data_sources:
        ds_id = ds.replace("/", "_").replace(".", "_").replace("-", "_")
        lines.append(f'  ds_{ds_id}["{ds}"]')
    for fv in fv_names:
        fv_id = fv.replace("-", "_")
        lines.append(f'  fv_{fv_id}["{fv}"]')
        for ds in data_sources:
            ds_id = ds.replace("/", "_").replace(".", "_").replace("-", "_")
            lines.append(f"  ds_{ds_id} --> fv_{fv_id}")
    fs_id = feature_service_name.replace("-", "_")
    lines.append(f'  fs_{fs_id}["{feature_service_name}"]:::fsNode')
    for fv in fv_names:
        lines.append(f"  fv_{fv.replace('-','_')} --> fs_{fs_id}")
    short = run_id[:8]
    lines.append(f'  run_{short}["MLflow Run {short}"]:::runNode')
    lines.append(f"  fs_{fs_id} --> run_{short}")
    lines.append("  classDef fsNode fill:#238636,stroke:#2ea043,color:#fff")
    lines.append("  classDef runNode fill:#1565c0,stroke:#1976d2,color:#fff")
    graph = "\n".join(lines)

    features_html = "".join(f"<li><code>{r}</code></li>" for r in feature_refs)
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Feast Lineage</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>body{{font-family:sans-serif;background:#0d1117;color:#e6edf3;padding:24px}}
h1{{color:#58a6ff;font-size:20px}}h2{{color:#8b949e;font-size:16px;margin-top:24px}}
code{{background:#21262d;padding:2px 6px;border-radius:4px;font-size:13px}}
ul{{padding-left:24px}}li{{margin:4px 0}}.meta{{color:#8b949e;font-size:13px}}</style>
</head><body>
<h1>Feast Feature Lineage</h1>
<p class="meta">Project: {project} | Service: {feature_service_name} | Run: {run_id[:12]}...</p>
<div class="mermaid">{graph}</div>
<h2>Features ({len(feature_refs)})</h2><ul>{features_html}</ul>
<h2>Data Sources</h2><ul>{"".join(f"<li>{ds}</li>" for ds in data_sources)}</ul>
<h2>Entities</h2><ul>{"".join(f"<li>{e}</li>" for e in entity_keys)}</ul>
<script>mermaid.initialize({{theme:'dark',startOnLoad:true}})</script>
</body></html>"""


def _build_lineage_markdown(
    *,
    feature_service_name: str,
    feature_refs: List[str],
    data_sources: List[str],
    entity_keys: List[str],
    run_id: str,
    project: str,
) -> str:
    return f"""# Feast Feature Lineage

| Field | Value |
|-------|-------|
| **Project** | {project} |
| **Feature Service** | {feature_service_name} |
| **MLflow Run** | `{run_id}` |
| **Features** | {len(feature_refs)} |
| **Entities** | {len(entity_keys)} |
| **Data Sources** | {len(data_sources)} |

## Features

{chr(10).join(f"- `{r}`" for r in feature_refs)}

## Data Sources

{chr(10).join(f"- {ds}" for ds in data_sources)}

## Entity Keys

{chr(10).join(f"- {e}" for e in entity_keys)}

---
*Auto-generated by Feast MLflow integration*
"""
