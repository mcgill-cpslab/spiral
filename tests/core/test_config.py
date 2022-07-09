#!/usr/bin/env python
"""Tests `spiro.core.config` package."""

import os
from tests.core.common_functions import *
from spiro.core.config import set_backend

invalid_backend = "bbb"

def test_set_backend_tf_by_code():
    clear_background()
    from spiro.core.backend_config import tensorflow
    assert os.environ[DGL_BACKEND] == TENSORFLOW


def test_invalid_backend():
    clear_background()
    set_backend(invalid_backend)
    assert os.environ[DGL_BACKEND] == PYTORCH


def test_set_backend_torch_by_code():
    clear_background()
    from spiro.core.backend_config import torch
    assert os.environ[DGL_BACKEND] == PYTORCH
