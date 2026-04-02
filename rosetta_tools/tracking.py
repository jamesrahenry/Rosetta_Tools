"""
tracking — MLflow integration for Rosetta experiment tracking.

Thin wrapper that keeps MLflow concerns out of extraction scripts.
All functions are safe to call even if MLflow is not installed or the
tracking server is unreachable — they degrade to no-ops with a warning.

Usage in extraction scripts
---------------------------
    from rosetta_tools.tracking import start_run, log_concept, end_run

    run = start_run("caz_scaling", model_id, {"dtype": "bfloat16", ...})
    # ... extraction loop ...
    log_concept(concept, summary_dict)
    end_run(run, out_dir)

Central tracking
----------------
The gpu_runner sets MLFLOW_TRACKING_URI before launching jobs so all
experiments land in one place. If not set, MLflow falls back to
./mlruns/ in the working directory (local-only mode).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

# Default central store: ~/mlflow (shared across all repos on the machine)
DEFAULT_STORE = Path.home() / "mlflow"
DEFAULT_PORT = 5111


def ensure_server(store: Path | None = None, port: int = DEFAULT_PORT) -> str:
    """Start an MLflow tracking server if one isn't already running.

    Returns the tracking URI (http://127.0.0.1:<port>).
    Safe to call multiple times — skips if the port is already bound.
    """
    import subprocess
    import socket

    store = store or DEFAULT_STORE
    store.mkdir(parents=True, exist_ok=True)
    uri = f"http://127.0.0.1:{port}"

    # Check if something is already listening on the port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            log.info("MLflow server already running at %s", uri)
            return uri

    # Start the server in the background
    log_file = store / "server.log"
    subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", f"sqlite:///{store}/mlflow.db",
            "--default-artifact-root", str(store / "artifacts"),
            "--host", "0.0.0.0",
            "--port", str(port),
        ],
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from parent process
    )

    # Wait briefly for it to come up
    import time
    for _ in range(10):
        time.sleep(0.5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                log.info("MLflow server started at %s (log: %s)", uri, log_file)
                return uri

    log.warning("MLflow server may not have started — check %s", log_file)
    return uri


def configure(tracking_uri: str | None = None) -> str | None:
    """Set the MLflow tracking URI for this process.

    Priority:
    1. Explicit tracking_uri argument
    2. MLFLOW_TRACKING_URI environment variable
    3. None (MLflow default — local ./mlruns/)

    Returns the URI that was set, or None.
    """
    if not _HAS_MLFLOW:
        return None

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        log.info("MLflow tracking URI: %s", uri)
    return uri


def start_run(
    experiment: str,
    model_id: str,
    params: dict[str, Any],
    run_name: str | None = None,
) -> Any | None:
    """Begin an MLflow run. Returns the run object, or None on failure."""
    if not _HAS_MLFLOW:
        log.warning("mlflow not installed — skipping experiment tracking")
        return None
    try:
        # Auto-configure from env if not already set
        if not mlflow.get_tracking_uri().startswith("http"):
            configure()

        mlflow.set_experiment(experiment)
        name = run_name or f"{model_id.split('/')[-1]}"
        run = mlflow.start_run(run_name=name)
        mlflow.log_params({
            "model_id": model_id,
            **{k: str(v) for k, v in params.items()},
        })
        return run
    except Exception as e:
        log.warning("MLflow start_run failed (tracking disabled): %s", e)
        return None


def log_concept(concept: str, summary: dict[str, Any]) -> None:
    """Log per-concept metrics to the active MLflow run."""
    if not _HAS_MLFLOW or mlflow.active_run() is None:
        return
    try:
        mlflow.log_metrics({
            f"{concept}/peak_layer": summary["peak_layer"],
            f"{concept}/peak_separation": summary["peak_separation"],
            f"{concept}/peak_depth_pct": summary["peak_depth_pct"],
            f"{concept}/extraction_seconds": summary["extraction_seconds"],
        })
    except Exception as e:
        log.warning("MLflow log_metrics failed for %s: %s", concept, e)


def end_run(run: Any | None, out_dir: Path | None = None) -> None:
    """Log artifacts from out_dir and close the MLflow run."""
    if run is None or not _HAS_MLFLOW:
        return
    try:
        if out_dir and out_dir.exists():
            # Log the run_summary.json and per-concept JSONs (skip large .npy files)
            for f in sorted(out_dir.glob("*.json")):
                mlflow.log_artifact(str(f))
        mlflow.end_run()
    except Exception as e:
        log.warning("MLflow end_run failed: %s", e)
        try:
            mlflow.end_run()
        except Exception:
            pass
