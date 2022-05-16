#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels` package."""
import numpy as np
import tensorflow as tf
from tests.core.common_functions import *


def test_node_memory_tf():
    """Test NodeMemory"""
    clear_background()
    from spiro.core.config import set_backend
    from spiro.core.backends import TENSORFLOW
    set_backend(TENSORFLOW)
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import NodeMemory

    n_nodes = 5
    hidden_d = 2
    n_layers = 3
    this_memory = NodeMemory(n_nodes, hidden_d, n_layers)
    nodes_to_change = [2,3]
    new_memory = tf.convert_to_tensor(np.random.randn(2, n_layers, hidden_d))
    this_memory.update_memory(new_memory, nodes_to_change)
    new_memory = tf.convert_to_tensor(np.random.randn(n_layers, 2, hidden_d))
    assert not np.all(tf.equal(this_memory.get_memory(nodes_to_change), new_memory))
    old_memory = this_memory.memory.copy()
    this_memory.reset_state()
    assert not np.all(tf.equal(old_memory, this_memory.memory))
