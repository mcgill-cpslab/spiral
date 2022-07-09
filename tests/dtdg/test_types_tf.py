#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.dtdg.types` package."""

import random
import tensorflow as tf
import numpy as np
from tests.core.common_functions import *
from spiro.core.logger import get_logger


edges = np.array([[0, 0, 1], [0, 0, 2], [1, 3, 0], [2, 4, 1]])
nodes = np.array([[0, 3], [0, 2], [0, 5], [1, 4], [2, 9]])
times = np.array([2006, 2007, 2008])

logger = get_logger()

def test_snapshot_tf():
    clear_background()
    from spiro.core.config import set_backend
    from spiro.core.backends import TENSORFLOW, PYTORCH

    set_backend(TENSORFLOW)
    import dgl
    from spiro.dtdg.types import Snapshot

    src_ids = tf.constant([2, 3, 4], dtype=tf.int32)
    dst_ids = tf.constant([1, 2, 3], dtype=tf.int32)

    g = dgl.graph((src_ids, dst_ids))
    sn = Snapshot(g, 1)


def test_citation_graph_tf():
    clear_background()
    """Test that citation graph could support different backend."""
    from spiro.core.config import set_backend
    from spiro.core.backends import TENSORFLOW

    set_backend(TENSORFLOW)
    from spiro.dtdg.types import CitationGraph

    this_graph = CitationGraph(edges, nodes, times)
    assert len(this_graph) == len(times)
    ob_0,_ = this_graph.dispatcher(0)
    assert ob_0.t == 0
    assert ob_0.observation.num_edges() == 2
    assert ob_0.observation.num_nodes() == 3
    assert np.sum(ob_0.observation.ndata['h'].numpy()[:, -1]) == 0
    ob_1, _ = this_graph.dispatcher(1)
    assert ob_1.t == 1
    assert ob_1.observation.num_edges() == 3
    assert ob_1.observation.num_nodes() == 4
    assert np.sum(ob_1.observation.ndata['h'].numpy()[:, -1]) == 2
