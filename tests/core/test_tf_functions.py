#!/usr/bin/env python
# flake8: noqa
"""Tests `nineturn.core.tf_functions` package."""
import numpy as np
import tensorflow as tf
from nineturn.core.config import set_backend
from nineturn.core.backends import TENSORFLOW
from tests.core.common_functions import *

arr1 = np.random.rand(3, 4)


def test_to_tensor():
    """Test _to_tensor"""
    clear_background()
    set_background(TENSORFLOW)
    set_backend(TENSORFLOW)
    from nineturn.core.tf_functions import _to_tensor

    assert np.all(tf.equal(_to_tensor(arr1), tf.convert_to_tensor(arr1)))


def test_nt_layers_list():
    """Test nt_layers_list"""
    clear_background()
    set_background(TENSORFLOW)
    set_backend(TENSORFLOW)
    from nineturn.core.tf_functions import nt_layers_list

    assert type(nt_layers_list()) == type([])
    assert len(nt_layers_list()) == 0


def test_reshape_tensor():
    """Test reshape_tensor"""
    clear_background()
    set_background(TENSORFLOW)
    set_backend(TENSORFLOW)
    from nineturn.core.tf_functions import reshape_tensor

    shape = -1

    assert np.all(tf.equal(reshape_tensor(arr1, shape), tf.reshape(arr1, shape)))