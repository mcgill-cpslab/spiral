#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.automl.prepare_dataset package."""

from tests.core.common_functions import *
from spiro.core.logger import get_logger
import numpy as np


logger = get_logger()

def test_prepare_edge_task():
    """Test LSTM"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import LSTM
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
    from spiro.automl.tf.prepare_dataset import prepare_edge_task, TARGET, LABEL
    from spiro.dtdg.models.decoder.simpleDecoders import MLP
    from spiro.automl.model_assembler import assembler
    from spiro.core.commonF import to_tensor

    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    n_snapshot = len(this_graph)
    eval_t = n_snapshot -1
    prepare_edge_task(this_graph, 2, num_negative=1000, start_t=eval_t)
    eval_snapshot,_ = this_graph.dispatcher(eval_t)
    eval_target_links = this_graph.time_data[TARGET][eval_t]
    edge_to_remove = this_graph.time_data['positive_edges'][eval_t]
    small_array = edge_to_remove.numpy()
    big_array = eval_snapshot.observation.edges(form='eid').numpy()
  
    assert all( k in big_array for k in  small_array)     
    assert len(this_graph.time_data[TARGET][eval_t]) == len(this_graph.time_data[LABEL][eval_t])

