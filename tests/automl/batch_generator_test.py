#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.dtdg.models.decoder.torch.sequentialDecoder.implicitTimeModels` package."""
import tensorflow as tf
import numpy as np
from tests.core.common_functions import *


edges = np.array([[0, 0, 1], [0, 0, 2], [1, 3, 0], [2, 4, 1], [2, 4, 2]])
nodes = np.array([[0, 0], [0, 1], [0, 2], [1, 3], [2, 4], [3, 9]])
times = np.array([2006, 2007, 2008,2009])

def test_batch_generator():
    """Test nodebatchgenerator"""
    clear_background()
    from spiro.core.config import set_backend
    from spiro.core.backends import TENSORFLOW
    set_backend(TENSORFLOW)
    from spiro.dtdg.types import CitationGraph
    from spiro.automl.tf.prepare_dataset import prepare_citation_task, TARGET, LABEL
    from spiro.automl.tf.batch_training import NodeBatchGenerator
    this_graph = CitationGraph(edges, nodes, times)
    prepare_citation_task(this_graph, validating_snapshots=1,minimum_citation=0)
    gen = NodeBatchGenerator(this_graph, 2,1,3,False)
    tt = [i for i in gen]
    number_of_batch=len(gen)
    assert len(tt) ==number_of_batch 
