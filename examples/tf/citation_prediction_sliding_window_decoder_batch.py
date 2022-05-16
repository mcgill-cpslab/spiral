import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import datetime
from spiro.core.backends import TENSORFLOW
from spiro.core.config import  set_backend
set_backend(TENSORFLOW)
from spiro.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GCN, SGCN, GAT, GraphSage
from spiro.dtdg.models.decoder.sequentialDecoders import SelfAttention, PTSA, FTSA, Conv1D, NodeTrackingPTSA
from spiro.dtdg.models.decoder.simpleDecoders import MLP
from spiro.core.commonF import to_tensor
from spiro.core.utils import printProgressBar
from spiro.automl.model_assembler import assembler
from spiro.automl.tf.prepare_dataset import prepare_citation_task, TARGET, LABEL
from spiro.automl.tf.batch_training import NodeBatchGenerator


"""
def loss_fn(predict, label):
    return torch.sqrt(torch.mean(torch.abs(torch.log1p(predict) - torch.log1p(label))))
"""
if __name__ == '__main__':
    #----------------------------------------------------------------
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    n_snapshot = len(this_graph)
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
    decoders = ['sa', 'ptsa', 'node_tracking_ptsa', 'tsa_sum', 'node_tracking_tsa', 'conv1d']
    epochs = 1000
    prepare_citation_task(this_graph)
    for g in range(1):
        for r in range(6):
            #set up logger
            this_logger = logging.getLogger('citation_predictoin_pipeline')
            this_logger.setLevel(logging.INFO)
            # create file handler which logs even debug messages
            log_path = f"model_{encoders[g]}_{decoders[r]}.log"
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
                    sa = NodeTrackingPTSA( 3, hidden_dim,layer_dims , n_nodes, 5, output_decoder)
                elif r == 3:
                    sa = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'sum', output_decoder)
                elif r == 4:
                    sa = FTSA( 3, hidden_dim, layer_dims, n_nodes, 5,3,'sum', output_decoder, node_tracking=True)
                elif r == 5:
                    sa = Conv1D(hidden_dim, [8,output_dim], n_nodes, 5, output_decoder)
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
                batch_size = 5000
                batchs = NodeBatchGenerator(this_graph, 40, 10, n_snapshot-2)
                eval_batchs = NodeBatchGenerator(this_graph, 40, n_snapshot-2, n_snapshot-1)
                total_batches = len(batchs)
                for epoch in range(epochs):
                    this_model.decoder.training()
                    this_model.decoder.memory.reset_state()
                    progress = 0
                    printProgressBar(progress, total_batches, prefix = 'Progress:', suffix = 'Complete', length = 50)
                    for snapshot, target, label in batchs:
                        with tf.GradientTape() as tape:
                            predict = tf.reshape(this_model((snapshot,target), training=True), (-1))
                            all_predictions.append(predict.numpy())
                            loss = loss_fn(label,predict)
                        grads = tape.gradient(loss, this_model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
                        loss_list.append(loss.numpy())
                        progress+=1
                        printProgressBar(progress, total_batches, prefix = 'Progress:', suffix = 'Complete', length = 50)
                    print(f"train loss: {loss_list[-1]}")
                    eval_t = n_snapshot - 2
                    predicts = np.array([])
                    labels = np.array([])
                    for snapshot, target, label in eval_batchs:
                        predicts = tf.experimental.numpy.hstack((predicts,tf.reshape(this_model((snapshot,target)),(-1))))
                        labels = tf.experimental.numpy.hstack((labels, label))
                    eval_predictions.append(predicts)
                    print(eval_predictions[-1][:20])
                    print(labels[:20])
                    test_loss = loss_fn(labels, predicts).numpy()
                    print(f"test loss:{test_loss}")
                    eval_loss.append(test_loss)
                    test_loss_2 = loss_fn_eval(labels, predicts).numpy()
                    eval_loss2.append(test_loss_2)
                    print(f"eval loss: {test_loss_2}")
                    mini = min(eval_loss)
                    batchs.on_epoch_end()
                    eval_batchs.on_epoch_end()
                    if eval_loss[-1] == mini:
                        print(f"save best model for loss {mini}")
                        this_model.save_model(save_path)
                    if epoch > 10:
                        if all(eval_loss[-40:] > mini):
                            print(mini)
                            break

                this_logger.info(loss_list)
                this_logger.info(eval_loss)
                this_logger.info(eval_loss2)
                this_logger.info(f"best loss {mini}")

"""
    gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=activation_f, allow_zero_in_degree=True, dropout=0.2)
    output_decoder = MLP(output_dim, [hidden_dim,20,10,5])
    decoder = FTSA( 3, hidden_dim, [8], n_nodes, 7,3,'sum', output_decoder)
    new_model = assembler(gnn, decoder)
    new_snapshot = this_graph.dispatcher(n_snapshot-2)
    next_snapshot = this_graph.dispatcher(n_snapshot-1)
    node_samples = np.arange(this_snapshot.num_nodes())
    new_predict = new_model((this_snapshot, node_samples))
    new_model.load_model(save_path)
    new_model.decoder.memory.reset_state()
    for t in range(1,n_snapshot-2):
        this_snapshot = this_graph.dispatcher(t)
        next_snapshot = this_graph.dispatcher(t+1)
        node_samples = np.arange(this_snapshot.num_nodes())
        predict = new_model((this_snapshot,node_samples))
        label = next_snapshot.node_feature()[:this_snapshot.num_nodes(), -1]
        all_predictions.append(tf.squeeze(predict).numpy())
        loss = loss_fn(tf.squeeze(predict), label)
    loss_list.append(loss.numpy())
    print(loss_list[-1])

    this_snapshot = this_graph.dispatcher(n_snapshot-2)
    next_snapshot = this_graph.dispatcher(n_snapshot-1)
    node_samples = np.arange(this_snapshot.num_nodes())
    predict = new_model((this_snapshot,node_samples))
    label = next_snapshot.node_feature()[:this_snapshot.num_nodes(), -1]
    eval_predictions.append(tf.squeeze(predict).numpy())
    loss = loss_fn(tf.squeeze(predict), label)
    eval_loss.append(loss.numpy())
    print(eval_loss[-1])
"""
