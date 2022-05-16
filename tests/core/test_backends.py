#!/usr/bin/env python
"""Tests `spiro.core.config` package."""

from spiro.core.backends import supported_backends, TENSORFLOW, PYTORCH


def test_supported_backends():
    """Dummy test, make sure the supported_backends return the desired list."""
    assert supported_backends() == [TENSORFLOW, PYTORCH]
