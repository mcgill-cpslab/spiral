"""
Code to repeat the experiment in the paper: introducing Node Memory into Discrete Time Dynamic Graph Learning
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import datetime
from spiro.core.backend_config import tensorflow
from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GCN, SGCN, GAT, GraphSage
from spiro.dtdg.models.decoder.sequentialDecoders import SelfAttention, PTSA, FTSA, Conv1D, Conv1D_N
from spiro.dtdg.models.decoder.sequentialDecoders import LSTM_N, LSTM
from spiro.dtdg.models.decoder.simpleDecoders import MLP
from spiro.core.commonF import to_tensor
from spiro.automl.model_assembler import assembler
from spiro.automl.tf.prepare_dataset import prepare_citation_task, TARGET, LABEL
from spiro.core.utils import printProgressBar


if __name__ == '__main__':
    #----------------------------------------------------------------
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    n_snapshot = len(this_graph)
    print(f"experiment on dataset: {data_to_test}")
    print(f"number of snapshots {n_snapshot}")
    n_nodes = this_graph.dispatcher(n_snapshot -1)[0].observation.num_nodes()
    this_snapshot,_ = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    hidden_dim = 32
    num_GNN_layers = 2
    num_RNN_layers = 3
    output_dim = 10
    layer_dims = [64,output_dim]
    activation_f = None
    encoders = ['gcn', 'sgcn', 'gat', 'sage']
    decoders = ['sa', 'ptsa', 'node_tracking_ptsa', 'tsa_sum', 'node_tracking_tsa', 'conv1d', 'convid_n', 'lstm',
    'lstm_n']
    epochs = 1000
    n_training = 23
    eval_t = n_snapshot - 2
    start_t = eval_t - n_training
    prepare_citation_task(this_graph, start_t=start_t)
    for g in range(4):
        for r in range(8,9):
            #set up logger
            this_logger = logging.getLogger('citation_predictoin_pipeline')
            this_logger.setLevel(logging.INFO)
            # create file handler which logs even debug messages
            log_path = f"data_{data_to_test}_model_{encoders[g]}_{decoders[r]}.log"
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            for hdlr in this_logger.handlers[:]:  # remove all old handlers
                this_logger.removeHandler(hdlr)
            this_logger.addHandler(fh)
            this_logger.info("--------------------------------------------------------")
            for trial in range(10):
                this_logger.info("--------------------------------------------------------")
                this_logger.info(f"start trial {trial}")
                if g == 0:
                    gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=None,norm='none', allow_zero_in_degree=True, dropout=0.2)
                elif g == 1:
                    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
                elif g == 2:
                    gnn = GAT([1], in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True)
                else: 
                    gnn = GraphSage('gcn', in_dim, hidden_dim,  activation=activation_f)

                output_decoder = MLP(output_dim, [hidden_dim,20,10,5], activation="linear")
                

                if r == 0:
                    sa = SelfAttention( 3, hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
                elif r == 1:
                    sa = PTSA( 3, hidden_dim,layer_dims , n_nodes, 5, output_decoder)
                elif r == 2:
                    sa = PTSA( 3, hidden_dim,layer_dims , n_nodes, 5, output_decoder, node_tracking=True)
                elif r == 3:
                    sa = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'sum', output_decoder)
                elif r == 4:
                    sa = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'sum', output_decoder, node_tracking=True)
                elif r == 5:
                    sa = Conv1D(hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
                elif r == 6:
                    sa = Conv1D_N(hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
                elif r == 7:
                    sa = LSTM( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
                elif r == 8:
                    sa = LSTM_N( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
                else:
                    pass


                this_model = assembler(gnn, sa)
                save_path = f"model_{encoders[g]}_{decoders[r]}_{trial}"
                loss_fn_eval = keras.losses.MeanSquaredError()
                loss_fn = keras.losses.MeanAbsolutePercentageError()
                loss_list=[]
                all_predictions=[]
                eval_loss = []
                eval_predictions = []
                eval_loss2 = []
                eval_predictions2= []
                lr = 1e-3
                optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
                total_batches = eval_t - start_t +1

                for epoch in range(epochs):
                    this_model.decoder.training()
                    this_model.decoder.reset_memory_state()
                    progress = 0
                    printProgressBar(progress, total_batches, prefix = 'Progress:', suffix = 'Complete', length = 50)
                    for t in range(start_t, eval_t):
                        this_snapshot, node_ids = this_graph.dispatcher(t)
                        with tf.GradientTape() as tape:
                            predict = tf.squeeze(this_model((this_snapshot,this_graph.time_data[TARGET][t]),
                                training=True))
                            label = this_graph.time_data[LABEL][t]
                            loss = loss_fn(label,predict)
                        all_predictions.append(predict.numpy())
                        grads = tape.gradient(loss, this_model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
                        progress+=1
                        printProgressBar(progress, total_batches, prefix = 'Progress:', suffix = 'Complete', length = 50)
                    loss_list.append(loss.numpy())
                    print(f"train loss: {loss.numpy()}")
                    this_snapshot, node_samples = this_graph.dispatcher(eval_t)
                    predict = this_model((this_snapshot,this_graph.time_data[TARGET][eval_t]))
                    label = this_graph.time_data[LABEL][eval_t]
                    eval_predictions.append(tf.squeeze(predict).numpy())
                    print(eval_predictions[-1][:20])
                    print(label[:20])
                    loss = loss_fn(label, tf.squeeze(predict)).numpy()
                    print(f"test loss:{loss}")
                    eval_loss.append(loss)
                    loss = loss_fn_eval(label, tf.squeeze(predict)).numpy()
                    eval_loss2.append(loss)
                    print(f"eval loss: {loss}")
                    mini = min(eval_loss)
                    if eval_loss[-1] == mini:
                        print(f"save best model for loss {mini}")
                        this_model.save_model(save_path)
                    if epoch > 10:
                        if all(eval_loss[-40:] > mini) or np.isnan( eval_loss[-1] ):
                            print(mini)
                            break

                this_logger.info(loss_list)
                this_logger.info(eval_loss)
                this_logger.info(eval_loss2)
                this_logger.info(f"best loss {mini}")
