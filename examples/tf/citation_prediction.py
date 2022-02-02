import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import datetime
from nineturn.core.backends import TENSORFLOW
from nineturn.core.config import  set_backend
set_backend(TENSORFLOW)
from nineturn.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from nineturn.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GCN, SGCN, GAT, GraphSage
from nineturn.dtdg.models.decoder.tf.sequentialDecoder.rnnFamily import LSTM, GRU,RNN
from nineturn.dtdg.models.decoder.simpleDecoders import MLP
from nineturn.core.commonF import to_tensor
from nineturn.automl.model_assembler import assembler


"""
def loss_fn(predict, label):
    return torch.sqrt(torch.mean(torch.abs(torch.log1p(predict) - torch.log1p(label))))
"""
if __name__ == '__main__':
    #gpus = tf.config.list_physical_devices('GPU')

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #----------------------------------------------------------------
    #set up logger
    this_logger = logging.getLogger('citation_predictoin_pipeline')
    this_logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('test2.log')
    fh.setLevel(logging.DEBUG)
    this_logger.addHandler(fh)
    #--------------------------------------------------------
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    n_snapshot = len(this_graph)
    this_logger.info(f"number of snapshots: {n_snapshot}")
    n_nodes = this_graph.dispatcher(n_snapshot -1).observation.num_nodes()
    this_logger.info(f"number of nodes: {n_nodes}")
    this_snapshot = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    hidden_dim = 32
    num_GNN_layers = 2
    num_RNN_layers = 2
    output_dim = 10
    activation_f = tf.nn.relu
    #gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=activation_f, allow_zero_in_degree=True, dropout=0.2)
    #gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
    #gnn = GAT([1], in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True)
    gnn = GraphSage('gcn', in_dim, hidden_dim,  activation=activation_f)
    output_decoder = MLP(output_dim, [10,20,10,5])
    #decoder = LSTM( hidden_dim, 10,n_nodes,3,output_decoder, device)
    #decoder = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
    decoder = RNN( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
    this_model = assembler(gnn, decoder)
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8)
    loss_list=[]
    all_predictions=[]
    for epoch in range(20):
        #this_model[1].reset_memory_state()
        for t in range(5,n_snapshot-2):
            with tf.GradientTape() as tape:
                this_snapshot = this_graph.dispatcher(t)
                next_snapshot = this_graph.dispatcher(t+1)
                node_samples = np.arange(this_snapshot.num_nodes())
                predict = this_model((this_snapshot,node_samples))
                label = next_snapshot.node_feature()[:this_snapshot.num_nodes(), -1]
                all_predictions.append(tf.squeeze(predict).numpy())
                loss = loss_fn(tf.squeeze(predict), label)
                grads = tape.gradient(loss, this_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, this_model.trainable_weights))
                loss_list.append(loss.numpy())
        print(loss_list[-1])
        print(all_predictions[-1][:20])
        print(label[:20])


