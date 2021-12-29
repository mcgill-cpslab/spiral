#!/usr/bin/env python
# flake8: noqa
"""Tests `nineturn.core.config` package."""

import os
from nineturn.core.backends import TENSORFLOW, PYTORCH
from nineturn.core.config import set_backend, get_logger, get_backend
from tests.core.common_functions import *

logger = get_logger()

logger.info("Testing backends setting functions.")


def test_set_backend_tf_by_env():
    clear_background()
    set_background(TENSORFLOW)
    set_backend()
    assert os.environ[DGL_BACKEND] == TENSORFLOW


def test_set_backend_tf_by_code():
    clear_background()
    set_backend(TENSORFLOW)
    assert os.environ[DGL_BACKEND] == TENSORFLOW


def test_invalid_backend():
    clear_background()
    set_invalid_background()
    set_backend()
    assert os.environ[DGL_BACKEND] == PYTORCH


def test_set_backend_torch_by_env():
    clear_background()
    set_background(PYTORCH)
    set_backend()
    assert os.environ[DGL_BACKEND] == PYTORCH


def test_set_backend_torch_by_code():
    clear_background()
    set_backend(PYTORCH)
    assert os.environ[DGL_BACKEND] == PYTORCH


def test_get_backend_tf():
    clear_background()
    set_backend(TENSORFLOW)
    current_backend = get_backend()
    assert current_backend == TENSORFLOW


def test_get_backend_torch():
    clear_background()
    set_backend(PYTORCH)
    current_backend = get_backend()
    assert current_backend == PYTORCH
