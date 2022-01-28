#!/usr/bin/env python
"""Tests `nineturn.core.config` package."""

from nineturn.core.backends import supported_backends, TENSORFLOW, PYTORCH


def test_supported_backends():
    """Dummy test, make sure the supported_backends return the desired list."""
    assert supported_backends() == [TENSORFLOW, PYTORCH]
