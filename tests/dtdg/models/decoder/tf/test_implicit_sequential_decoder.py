#!/usr/bin/env python
# flake8: noqa
"""Tests `spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels` package."""

from tests.core.common_functions import *
from spiro.core.logger import get_logger
import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = get_logger()
hidden_dim = 32
num_GNN_layers = 1
num_RNN_layers = 3
output_dim = 10
layer_dims = [64,output_dim]
activation_f = None


def test_lstm():
    """Test LSTM"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import LSTM
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = LSTM( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = LSTM( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder2)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

def test_lstm_n():
    """Test LSTM"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import LSTM_N
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = LSTM_N( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = LSTM_N( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder2)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

def test_gru():
    """Test GRU"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import GRU
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder2)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

def test_rnn():
    """Test GRU"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import RNN
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = RNN( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = RNN( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder2)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

def test_sa():
    """Test GRU"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import SelfAttention
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = SelfAttention( 2, hidden_dim, [2,output_dim], n_nodes, 2, output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = SelfAttention( 2, hidden_dim, [2,output_dim], n_nodes, 2, output_decoder)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3


def test_ptsa():
    """Test PTSA"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import PTSA
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = PTSA( 2, hidden_dim, [2,output_dim], n_nodes, 2, output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = PTSA( 2, hidden_dim, [2,output_dim], n_nodes, 2, output_decoder)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3


def test_ptsa_n():
    """Test ptsa with node memory"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import PTSA
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = PTSA( 2, hidden_dim, [2,output_dim], n_nodes, 2, output_decoder, node_tracking=True)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = PTSA( 2, hidden_dim, [2,output_dim], n_nodes, 2, output_decoder, node_tracking=True)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3


def test_conv1d():
    """Test ptsa with node memory"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import Conv1D
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = Conv1D(hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = Conv1D(hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

def test_conv1d_n():
    """Test ptsa with node memory"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import Conv1D_N
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = Conv1D_N(hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = Conv1D_N(hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

def test_ftsa_sum():
    """Test ptsa with node memory"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import FTSA
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'sum', output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'sum', output_decoder)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3

def test_ftsa_con():
    """Test ptsa with node memory"""
    clear_background()
    from spiro.core.backend_config import tensorflow
    from spiro.dtdg.models.decoder.tf.sequentialDecoder.implicitTimeModels import FTSA
    from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
    from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import SGCN
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

    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'concate', output_decoder)
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
    for i in range(3):
        this_model.decoder.reset_memory_state()
        with tf.GradientTape() as tape:
            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                        training=True))
            loss = loss_fn(label,predict)
        if i == 0:
            loss1 = loss.numpy()
        grads = tape.gradient(loss, this_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
    this_model.decoder.reset_memory_state()
    predict2 = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t])))
    loss2 = loss_fn(label,predict2).numpy()
    assert loss2 < loss1
    this_model.save_model(save_path)

    gnn2 = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    output_decoder2 = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
    sa2 = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'concate', output_decoder)
    this_model2 = assembler(gnn2, sa2)
    this_model2((this_snapshot,this_graph.time_data[TARGET][t]))
    this_model2.load_model(save_path)
    this_model2.decoder.reset_memory_state()
    predict3 = tf.squeeze(this_model2((this_snapshot,this_graph.time_data[TARGET][t])))
    loss3 = loss_fn(label,predict3).numpy()
    assert loss2 == loss3
