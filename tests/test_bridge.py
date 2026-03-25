"""Tests for feast_mlflow.bridge -- verifies patching mechanics."""

from unittest.mock import MagicMock, patch

from feast_mlflow.bridge import _get_feast_context, _set_feast_context, _clear_feast_context


def test_feast_context_thread_local():
    _clear_feast_context()
    assert _get_feast_context() == {}

    _set_feast_context({"store": "mock", "logged": False})
    ctx = _get_feast_context()
    assert ctx["store"] == "mock"
    assert ctx["logged"] is False

    _clear_feast_context()
    assert _get_feast_context() == {}


def test_enable_disable():
    import feast_mlflow.bridge as bridge

    was_active = bridge.is_active()

    if was_active:
        bridge.disable()

    assert bridge.is_active() is False

    bridge.enable()
    assert bridge.is_active() is True

    bridge.enable()
    assert bridge.is_active() is True

    bridge.disable()
    assert bridge.is_active() is False
