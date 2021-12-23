#!/usr/bin/env python
"""Tests `nineturn.core.config` package."""

from nineturn.core.backends import supported_backends, TENSORFLOW, PYTORCH


def test_supported_backends():
    assert supported_backends() == [TENSORFLOW, PYTORCH]
