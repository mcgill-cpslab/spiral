#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.core.config` package."""

import os
import numpy as np
import tensorflow as tf
from spiro.core.backends import TENSORFLOW, PYTORCH
from tests.core.common_functions import *
from spiro.core.logger import get_logger

def test_conv1d_tf():
    clear_background()
    from spiro.core.config import _BACKEND, set_backend

    set_backend(TENSORFLOW)
    from spiro.core.layers import Conv1d
    
    features = 5
    output_dimension = 2
    kernel = 4
    window = 9
    n_nodes = 20
    time_dimension = tf.convert_to_tensor(np.random.rand(n_nodes,window,features), dtype=tf.float32)
     
    l = Conv1d(features,output_dimension,window)
    b = l(time_dimension)
    assert(b.shape==[n_nodes,window,output_dimension])


