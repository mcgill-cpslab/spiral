#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.core.config` package."""

import os
import numpy as np
from spiro.core.backends import TENSORFLOW, PYTORCH
from tests.core.common_functions import *
from spiro.core.logger import get_logger


logger = get_logger()


def test_get_backend_tf():
    clear_background()
    from spiro.core.config import _BACKEND, set_backend

    set_backend(TENSORFLOW)
    from spiro.core.utils import _get_backend

    current_backend = _get_backend()
    assert current_backend == TENSORFLOW


def test_get_backend_torch():
    clear_background()
    from spiro.core.config import _BACKEND, set_backend

    set_backend(PYTORCH)
    from spiro.core.utils import _get_backend

    current_backend = _get_backend()
    assert current_backend == PYTORCH


def test_is_sorted():
    from spiro.core.utils import is_sorted

    sorted_arr = np.arange(10)
    unsorted_arr = np.arange(10)
    np.random.shuffle(unsorted_arr)
    assert is_sorted(sorted_arr)
    assert not is_sorted(unsorted_arr)


def test_anchor():
    from spiro.core.utils import get_anchor_position

    a = np.array([0, 0, 1, 1, 2, 3, 3, 3, 4, 5])
    b = np.unique(a)
    c = get_anchor_position(a, b)
    assert (c == np.array([2, 4, 5, 8, 9, 10])).all()
