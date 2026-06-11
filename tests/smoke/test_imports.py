"""Smoke: all core modules import without error."""

from __future__ import annotations


def test_import_core() -> None:
    import m6
    import m6.config
    import m6.cv
    import m6.data
    import m6.evaluation
    import m6.features
    import m6.metrics
    import m6.scoring

    assert m6


def test_import_models() -> None:
    import m6.models
    import m6.models.gaussian
    import m6.models.historical
    import m6.models.naive

    assert m6.models


def test_import_cli() -> None:
    from m6.cli import app

    assert app
