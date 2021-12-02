#!/usr/bin/env python
# flake8: noqa
"""Tests `nineturn.core.config` package."""

import os
from tests.core.common_functions import *


def test_first_time_get_backend():
    set_background()
    from nineturn.core import config

    assert os.environ[DGL_BACKEND] == TENSORFLOW
    assert config.get_backend() == TENSORFLOW


def test_invalid_backend(capsys):
    set_invalid_background()
    from nineturn.core import config

    out, err = capsys.readouterr()
    assert " ".join(err.split()) == " ".join(BACKEND_NOT_FOUND.split())
