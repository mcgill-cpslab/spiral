import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from nineturn.core.backends import PYTORCH
from nineturn.core.config import get_logger, set_backend
logger = get_logger()
set_backend(PYTORCH)
from nineturn.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from nineturn.dtdg.models.encoder.implicitTimeEncoder.torch.staticGraphEncoder import GCN, GAT
from nineturn.dtdg.models.decoder.torch.sequentialDecoder import LSTM


def assembler(encoder, decoder):
    return nn.Sequential(encoder,decoder)



if __name__ == '__main__':
    device = 'cpu'
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    n_snapshot = len(this_graph)
    print(n_snapshot)
    n_nodes = this_graph.dispatcher(n_snapshot -1).observation.num_nodes()
    print(n_nodes)
    this_snapshot = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    hidden_dim = 32
    num_GNN_layers = 2
    num_RNN_layers = 2
    #gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=F.leaky_relu,allow_zero_in_degree=True, dropout=0.2).to(device)
    gnn = GAT([2,3,2], in_dim, hidden_dim,  activation=F.leaky_relu,allow_zero_in_degree=True).to(device)
    decoder = LSTM(2 * hidden_dim, 10,n_nodes,device)
    this_model = assembler(gnn, decoder)
    n_feat = this_snapshot.node_feature().float()
    predict = this_model.forward((this_snapshot.observation,n_feat , np.array(range(200))))
    print(predict)
    this_snapshot = this_graph.dispatcher(21)
    n_feat = this_snapshot.node_feature()
    predict = this_model.forward((this_snapshot.observation,n_feat , np.array(range(200))))
    print(predict)
