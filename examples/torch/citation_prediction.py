import numpy as np
import torch
import torch.nn.functional as F
import logging
import datetime

from nineturn.core.backends import PYTORCH
from nineturn.core.config import  set_backend
set_backend(PYTORCH)
from nineturn.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from nineturn.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import GCN, GAT, SGCN, GraphSage
from nineturn.dtdg.models.decoder.sequentialDecoders import LSTM, GRU,RNN
from nineturn.dtdg.models.decoder.simpleDecoders import MLP
from nineturn.automl.model_assembler import assembler


"""
def loss_fn(predict, label):
    return torch.sqrt(torch.mean(torch.abs(torch.log1p(predict) - torch.log1p(label))))
"""
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    activation_f = F.leaky_relu
    #gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True, dropout=0.2).to(device)
    #gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True).to(device)
    #gnn = GAT([1], in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True).to(device)
    gnn = GraphSage('gcn', in_dim, hidden_dim,  activation=activation_f)
    output_decoder = MLP(10, [10,20,10,5])
    decoder = LSTM( hidden_dim, 10,n_nodes,3,output_decoder)
    #decoder = GRU( hidden_dim, 10,n_nodes,3,output_decoder)
    #decoder = RNN( hidden_dim, 10,n_nodes,3,output_decoder)
    #this_model = LSTM( in_dim, 10,n_nodes,device).to(device)
    this_model = assembler(gnn, decoder).to(device)
    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        [{"params": this_model.parameters()}], lr=1e-3
    )
    loss_list=[]
    all_predictions=[]
    for epoch in range(20):
        this_model.decoder.reset_memory_state()
        for t in range(5,n_snapshot-2):
            this_model.train()
            optimizer.zero_grad()
            this_snapshot = this_graph.dispatcher(t)
            next_snapshot = this_graph.dispatcher(t+1)
            node_samples = torch.arange(this_snapshot.num_nodes())
            predict = this_model.forward((this_snapshot, node_samples))
            label = next_snapshot.node_feature()[:this_snapshot.num_nodes(), -1].float()
            all_predictions.append(predict.squeeze().clone())
            loss = loss_fn(predict.squeeze(), label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(loss_list[-1])
        print(all_predictions[-1][:20])
        print(label[:20])


