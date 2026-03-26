import json
import logging
import os
import threading
from importlib import resources as importlib_resources
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import feast

logger = logging.getLogger("feast.ui_server")


def _inject_mlflow_training_runs(registry_proto):
    """Query MLflow for runs tagged with feast metadata and inject them into the registry proto."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return

    try:
        import mlflow
        from google.protobuf.timestamp_pb2 import Timestamp
        from feast.protos.feast.core.Registry_pb2 import TrainingRunMetadata

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

        existing_run_ids = {r.run_id for r in registry_proto.training_runs}

        for exp in client.search_experiments():
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.`feast.feature_service` != ''",
                max_results=50,
            )
            for r in runs:
                if r.info.run_id in existing_run_ids:
                    continue
                tags = r.data.tags
                refs_str = tags.get("feast.feature_refs", "")
                refs = [x.strip() for x in refs_str.split(",") if x.strip()]

                created = Timestamp()
                created.FromMilliseconds(r.info.start_time)

                training_run = TrainingRunMetadata(
                    run_id=r.info.run_id,
                    experiment_name=exp.name,
                    tracking_uri=tracking_uri.replace("mlflow", "localhost"),
                    feature_service_name=tags.get("feast.feature_service", ""),
                    project=tags.get("feast.project", ""),
                    feature_refs=refs,
                    created_at=created,
                )
                registry_proto.training_runs.append(training_run)
    except Exception as e:
        logger.debug(f"Could not inject MLflow training runs: {e}")


def get_app(
    store: "feast.FeatureStore",
    project_id: str,
    registry_ttl_secs: int,
    root_path: str = "",
):
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    registry_proto = None
    shutting_down = False
    active_timer: Optional[threading.Timer] = None

    def async_refresh():
        store.refresh_registry()
        nonlocal registry_proto
        registry_proto = store.registry.proto()
        _inject_mlflow_training_runs(registry_proto)
        if shutting_down:
            return
        nonlocal active_timer
        active_timer = threading.Timer(registry_ttl_secs, async_refresh)
        active_timer.start()

    @app.on_event("shutdown")
    def shutdown_event():
        nonlocal shutting_down
        shutting_down = True
        if active_timer:
            active_timer.cancel()

    async_refresh()

    ui_dir_ref = importlib_resources.files(__spec__.parent) / "ui/build/"  # type: ignore[name-defined, arg-type]
    with importlib_resources.as_file(ui_dir_ref) as ui_dir:
        with ui_dir.joinpath("projects-list.json").open(mode="w") as f:
            discovered_projects = []
            registry = store.registry.proto()

            if registry and registry.projects and len(registry.projects) > 0:
                for proj in registry.projects:
                    if proj.spec and proj.spec.name:
                        discovered_projects.append(
                            {
                                "name": proj.spec.name.replace("_", " ").title(),
                                "description": proj.spec.description
                                or f"Project: {proj.spec.name}",
                                "id": proj.spec.name,
                                "registryPath": f"{root_path}/registry",
                            }
                        )
            else:
                discovered_projects.append(
                    {
                        "name": "Project",
                        "description": "Test project",
                        "id": project_id,
                        "registryPath": f"{root_path}/registry",
                    }
                )

            if len(discovered_projects) > 1:
                all_projects_entry = {
                    "name": "All Projects",
                    "description": "View data across all projects",
                    "id": "all",
                    "registryPath": f"{root_path}/registry",
                }
                discovered_projects.insert(0, all_projects_entry)

            projects_dict = {"projects": discovered_projects}
            f.write(json.dumps(projects_dict))

    @app.get("/registry")
    def read_registry():
        if registry_proto is None:
            return Response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        return Response(
            content=registry_proto.SerializeToString(),
            media_type="application/octet-stream",
        )

    @app.get("/health")
    def health():
        return (
            Response(status_code=status.HTTP_200_OK)
            if registry_proto
            else Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        )

    @app.api_route("/p/{path_name:path}", methods=["GET"])
    def catch_all():
        return Response(
            content=open(
                os.path.join(
                    str(ui_dir),
                    "index.html",
                )
            ).read(),
            media_type="text/html",
        )

    app.mount(
        "/",
        StaticFiles(directory=str(ui_dir), html=True),
        name="site",
    )

    return app


def start_server(
    store: "feast.FeatureStore",
    host: str,
    port: int,
    get_registry_dump: Callable,
    project_id: str,
    registry_ttl_sec: int,
    root_path: str = "",
    tls_key_path: str = "",
    tls_cert_path: str = "",
):
    app = get_app(
        store,
        project_id,
        registry_ttl_sec,
        root_path,
    )
    if tls_key_path and tls_cert_path:
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=tls_key_path,
            ssl_certfile=tls_cert_path,
        )
    else:
        uvicorn.run(app, host=host, port=port)
