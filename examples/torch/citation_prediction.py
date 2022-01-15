import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from nineturn.core.backends import PYTORCH
from nineturn.core.config import  set_backend
set_backend(PYTORCH)
from nineturn.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from nineturn.dtdg.models.encoder.implicitTimeEncoder.torch.staticGraphEncoder import GCN, GAT, SGCN, GraphSage
from nineturn.dtdg.models.decoder.torch.sequentialDecoder import LSTM


def assembler(encoder, decoder):
    return nn.Sequential(encoder,decoder).to(device)

loss_fn = torch.nn.MSELoss()
"""
def loss_fn(predict, label):
    return torch.sqrt(torch.mean(torch.abs(torch.log1p(predict) - torch.log1p(label))))
"""
if __name__ == '__main__':
    device = 'cpu'
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
    #gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=F.leaky_relu,allow_zero_in_degree=True, dropout=0.2).to(device)
    #gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True).to(device)
    #gnn = GAT([1], in_dim, hidden_dim,  activation=F.leaky_relu,allow_zero_in_degree=True).to(device)
    gnn = GraphSage('gcn', in_dim, hidden_dim,  activation=F.leaky_relu).to(device)
    decoder = LSTM( hidden_dim, 10,n_nodes).to(device)
    #this_model = LSTM( in_dim, 10,n_nodes,device).to(device)
    this_model = assembler(gnn, decoder)

    
    optimizer = torch.optim.Adam(
        [{"params": this_model.parameters()}], lr=1e-3
    )
    loss_list=[]
    all_predictions=[]
    for epoch in range(2):
        this_model[1].memory.reset_state()
        for t in range(5,7):
            this_model.train()
            optimizer.zero_grad()
            this_snapshot = this_graph.dispatcher(t)
            next_snapshot = this_graph.dispatcher(t+1)
            node_samples = torch.arange(this_snapshot.num_nodes())
            predict = this_model.forward((this_snapshot, node_samples)).to(device)
            #predict, (h,c) = this_model.forward((this_snapshot.node_feature()[node_samples].float(), this_state))
            label = next_snapshot.node_feature()[:this_snapshot.num_nodes(), -1].float()
            all_predictions.append(predict.squeeze())
            loss = loss_fn(predict.squeeze(), label).to(device)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(loss_list[-1])
        print(all_predictions[-1][:20])
        print(label[:20])


