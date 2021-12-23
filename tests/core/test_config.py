#!/usr/bin/env python
# flake8: noqa
"""Tests `nineturn.core.config` package."""

import os
from nineturn.core.backends import TENSORFLOW, PYTORCH
from nineturn.core.config import set_backend, get_logger
from tests.core.common_functions import *

logger = get_logger()


def test_first_time_get_backend():
    set_background(TENSORFLOW)
    set_backend()
    assert os.environ[DGL_BACKEND] == TENSORFLOW


def test_first_time_set_backend():
    clear_background()
    set_backend(TENSORFLOW)
    assert os.environ[DGL_BACKEND] == TENSORFLOW


def test_invalid_backend():
    set_invalid_background()
    set_backend()
    assert os.environ[DGL_BACKEND] == PYTORCH


def test_first_time_get_backend_torch():
    set_background(PYTORCH)
    set_backend()
    assert os.environ[DGL_BACKEND] == PYTORCH


def test_first_time_set_backend_torch():
    clear_background()
    set_backend(PYTORCH)
    assert os.environ[DGL_BACKEND] == PYTORCH
