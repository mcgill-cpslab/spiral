#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.core.config` package."""

import os
import numpy as np
import tensorflow as tf
from spiro.core.backends import TENSORFLOW, PYTORCH
from tests.core.common_functions import *
from spiro.core.logger import get_logger

logger = get_logger()


def test_time2vec_tf():
    clear_background()
    from spiro.core.config import _BACKEND, set_backend

    set_backend(TENSORFLOW)
    from spiro.core.layers import Time2Vec
    
    features = 5
    kernel = 4
    window = 9
    time_dimension = tf.convert_to_tensor([np.arange(window)], dtype=tf.float32)
     
    t2v = Time2Vec(kernel,features)
    b = t2v(time_dimension)
    assert(b.shape==[1,window,features,kernel])


