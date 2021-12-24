#!/usr/bin/env python
# flake8: noqa
"""Tests `nineturn.dtdg.types` package."""

import random
import torch
import tensorflow as tf
from nineturn.core.config import set_backend
from nineturn.core.backends import TENSORFLOW, PYTORCH
from nineturn.dtdg.types import Snapshot, DiscreteGraph


def test_snapshot_tf():
    set_backend(TENSORFLOW)
    src_ids = tf.constant([2, 3, 4], dtype=tf.int32)
    dst_ids = tf.constant([1, 2, 3], dtype=tf.int32)
    from dgl import graph

    g = graph((src_ids, dst_ids))
    sn = Snapshot(g, 1)


def test_snapshot_torch():
    """Test that Snapshot could support different backend."""
    set_backend(PYTORCH)
    src_ids = torch.tensor([2, 3, 4])
    dst_ids = torch.tensor([1, 2, 3])
    from dgl import graph

    g = graph((src_ids, dst_ids))
    sn = Snapshot(g, 1)


def test_discrete_graph_torch():
    """Test that discrete graph could support different backend."""
    set_backend(PYTORCH)
    src_ids = torch.tensor([2, 3, 4])
    dst_ids = torch.tensor([1, 2, 3])
    from dgl import graph

    g = graph((src_ids, dst_ids))
    sns = [Snapshot(g, t) for t in range(10)]
    random.shuffle(sns)
    dg = DiscreteGraph(sns)
    assert dg.snapshots[2].t == 2
    assert dg.snapshots[3].t == 3
