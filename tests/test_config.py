"""Tests for feast_mlflow.config."""

import os
import tempfile
from pathlib import Path

import pytest

from feast_mlflow.config import BridgeConfig


def test_from_yaml_with_mlflow_block():
    yaml_content = """
project: test
provider: local
mlflow:
  tracking_uri: http://mlflow:5000
  auto_log: true
  lineage: true
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "feature_store.yaml").write_text(yaml_content)
        cfg = BridgeConfig.from_yaml(tmpdir)

    assert cfg.enabled is True
    assert cfg.tracking_uri == "http://mlflow:5000"
    assert cfg.auto_log is True
    assert cfg.lineage is True


def test_from_yaml_without_mlflow_block():
    yaml_content = """
project: test
provider: local
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "feature_store.yaml").write_text(yaml_content)
        cfg = BridgeConfig.from_yaml(tmpdir)

    assert cfg.enabled is False


def test_from_yaml_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = BridgeConfig.from_yaml(tmpdir)
    assert cfg.enabled is False


def test_from_env_enabled(monkeypatch):
    monkeypatch.setenv("FEAST_MLFLOW", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://test:5000")
    cfg = BridgeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.tracking_uri == "http://test:5000"
    assert cfg.auto_log is True


def test_from_env_disabled(monkeypatch):
    monkeypatch.delenv("FEAST_MLFLOW", raising=False)
    cfg = BridgeConfig.from_env()
    assert cfg.enabled is False


def test_resolve_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("FEAST_MLFLOW", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")

    yaml_content = """
project: test
mlflow:
  tracking_uri: http://yaml:5000
  auto_log: true
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "feature_store.yaml").write_text(yaml_content)
        cfg = BridgeConfig.resolve(feast_repo_path=tmpdir)

    assert cfg.enabled is True
    assert cfg.tracking_uri == "http://env:5000"
