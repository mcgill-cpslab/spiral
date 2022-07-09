#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels` package."""

from tests.core.common_functions import *
from spiro.core.logger import get_logger
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pytest

logger = get_logger()
hidden_dim = 32
num_GNN_layers = 1
num_RNN_layers = 3
output_dim = 10
layer_dims = [64,output_dim]
activation_f = None
iteration = 5

def test_gcn():
    """Test GCN"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import GRU
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GCN
    from spiro.automl.tf.prepare_dataset import prepare_citation_task, TARGET, LABEL
    from spiro.dtdg.models.decoder.simpleDecoders import MLP
    from spiro.automl.model_assembler import assembler
    from spiro.core.commonF import to_tensor

    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    prepare_citation_task(this_graph, start_t=9)
    this_snapshot,_ = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    n_snapshot = len(this_graph)
    n_nodes = this_graph.dispatcher(n_snapshot -1)[0].observation.num_nodes()

    gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=None,norm='none', allow_zero_in_degree=True, dropout=0.2)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
    this_model = assembler(gnn, sa)
    save_path = "model_lstm"
    loss_fn = keras.losses.MeanAbsolutePercentageError()
    lr = 1e-3
    optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
    this_model.decoder.training()
    t = 11
    this_snapshot, node_ids = this_graph.dispatcher(t)
    label = this_graph.time_data[LABEL][t]
    loss1 = 0
    for i in range(iteration):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 1:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=None,norm='none', allow_zero_in_degree=True, dropout=0.2)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder2)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3



def test_gat():
    """Test GAT"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import GRU
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GAT
    from spiro.automl.tf.prepare_dataset import prepare_citation_task, TARGET, LABEL
    from spiro.dtdg.models.decoder.simpleDecoders import MLP
    from spiro.automl.model_assembler import assembler
    from spiro.core.commonF import to_tensor
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    prepare_citation_task(this_graph, start_t=9)
    this_snapshot,_ = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    n_snapshot = len(this_graph)
    n_nodes = this_graph.dispatcher(n_snapshot -1)[0].observation.num_nodes()

    gnn = GAT([1], in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
    this_model = assembler(gnn, sa)
    save_path = "model_lstm"
    loss_fn = keras.losses.MeanAbsolutePercentageError()
    lr = 1e-3
    optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
    this_model.decoder.training()
    t = 11
    this_snapshot, node_ids = this_graph.dispatcher(t)
    label = this_graph.time_data[LABEL][t]
    loss1 = 0
    for i in range(iteration):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 1:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = GAT([1], in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder2)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3


def test_dysat():
    """Test dysat"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import GRU
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import DysatGat
    from spiro.automl.tf.prepare_dataset import prepare_citation_task, TARGET, LABEL
    from spiro.dtdg.models.decoder.simpleDecoders import MLP
    from spiro.automl.model_assembler import assembler
    from spiro.core.commonF import to_tensor
    from spiro.core.errors import ValueError
    
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    prepare_citation_task(this_graph, start_t=9)
    this_snapshot,_ = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    n_snapshot = len(this_graph)
    n_nodes = this_graph.dispatcher(n_snapshot -1)[0].observation.num_nodes()

    gnn = DysatGat(1, in_dim, hidden_dim,1,0)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
    this_model = assembler(gnn, sa)
    save_path = "model_lstm"
    loss_fn = keras.losses.MeanAbsolutePercentageError()
    lr = 1e-3
    optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
    this_model.decoder.training()
    t = 11
    this_snapshot, node_ids = this_graph.dispatcher(t)
    label = this_graph.time_data[LABEL][t]
    loss1 = 0
    for i in range(iteration):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 1:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1

    with pytest.raises(ValueError, match="Save and load model is not supported for Dysat."):
        this_model.save_model(save_path)


def test_graphsage():
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import GRU
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GraphSage
    from spiro.automl.tf.prepare_dataset import prepare_citation_task, TARGET, LABEL
    from spiro.dtdg.models.decoder.simpleDecoders import MLP
    from spiro.automl.model_assembler import assembler
    from spiro.core.commonF import to_tensor
    from spiro.core.errors import ValueError
    
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    prepare_citation_task(this_graph, start_t=9)
    this_snapshot,_ = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    n_snapshot = len(this_graph)
    n_nodes = this_graph.dispatcher(n_snapshot -1)[0].observation.num_nodes()

    gnn = GraphSage('gcn', in_dim, hidden_dim,  activation=activation_f)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
    this_model = assembler(gnn, sa)
    save_path = "model_tem"
    loss_fn = keras.losses.MeanAbsolutePercentageError()
    lr = 1e-3
    optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
    this_model.decoder.training()
    t = 11
    this_snapshot, node_ids = this_graph.dispatcher(t)
    label = this_graph.time_data[LABEL][t]
    loss1 = 0
    for i in range(iteration):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 1:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)
    gnn2 = GraphSage('gcn', in_dim, hidden_dim,  activation=activation_f)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder2)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

