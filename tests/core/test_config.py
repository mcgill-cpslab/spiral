#!/usr/bin/env python
"""Tests `nineturn.core.config` package."""

import os
from nineturn.core.backends import TENSORFLOW, PYTORCH
from nineturn.core.config import set_backend
from tests.core.common_functions import *


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
