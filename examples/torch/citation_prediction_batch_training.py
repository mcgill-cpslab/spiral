import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import json
import logging
import logging.config
from nineturn.core.backends import PYTORCH
from nineturn.core.config import  set_backend
set_backend(PYTORCH)
from nineturn.dtdg.types import BatchedSnapshot
from nineturn.dtdg.dataloader import ogb_dataset, supported_ogb_datasets
from nineturn.dtdg.models.encoder.implicitTimeEncoder.torch.staticGraphEncoder import GCN, GAT, SGCN, GraphSage
from nineturn.dtdg.models.decoder.torch.sequentialDecoder import LSTM
from nineturn.automl.torch.model_assembler import assembler
import dgl


#def loss_fn(predict, label):
#    return torch.mean(torch.abs(predict - label))

loss_fn = torch.nn.MSELoss()

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)


if __name__ == '__main__':
    device = 'cpu'
    #----------------------------------------------------------------
    #set up logger
    this_logger = logging.getLogger('citation_predictoin_pipeline')
    this_logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('node_wise_batch.log')
    fh.setLevel(logging.DEBUG)
    this_logger.addHandler(fh)
    #--------------------------------------------------------

    #-------------------------------------------------------
    #load data
    data_to_test = supported_ogb_datasets()[1]
    this_graph = ogb_dataset(data_to_test)
    #-------------------------------------------------------

    #-------------------------------------------------------
    #create learning model
    n_snapshot = len(this_graph)
    n_nodes = this_graph.dispatcher(n_snapshot -1).observation.num_nodes()
    this_snapshot = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    hidden_dim = 200
    num_GNN_layers = 3
    num_RNN_layers = 2
    gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=F.leaky_relu,allow_zero_in_degree=True, dropout=0.2).to(device)
    #gnn = GAT([1], in_dim, hidden_dim,  activation=F.leaky_relu,allow_zero_in_degree=True).to(device)
    decoder = LSTM( hidden_dim, 20,n_nodes)
    #this_model = LSTM( in_dim, 20,n_nodes,device).to(device)
    this_model = assembler(gnn, decoder).to(device)
    #---------------------------------------------------------- 

    #---------------------------------------------------------
    #configure training
    optimizer = torch.optim.Adam(
        [{"params": this_model.parameters()}], lr=1e-3
    )
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    #---------------------------------------------------------
    
    test_loss_list =[]
    eval_loss_list = []
    all_predictions = []
    for epoch in range(400):
        this_model[1].memory.reset_state()
        #this_model[0].set_mini_batch(True)
        this_model[1].set_mini_batch(True)
        for t in range(2,n_snapshot-3):
            this_snapshot = this_graph.dispatcher(t, True)
            next_snapshot = this_graph.dispatcher(t+1, True)
            node_samples = torch.arange(this_snapshot.num_nodes())
            #------------------------
            # batch creation
            #------------------------
            collator = dgl.dataloading.NodeCollator(this_snapshot.observation, node_samples, sampler)
            dataloader = dgl.dataloading.NodeDataLoader(
                this_snapshot.observation, node_samples, sampler,
                batch_size=500,
                shuffle=True,
                drop_last=False,
                num_workers=1)
            for in_nodes, out_nodes, blocks in dataloader:
                this_model.train()
                optimizer.zero_grad()
                sample = BatchedSnapshot(blocks, this_snapshot.node_feature()[in_nodes],this_snapshot.t)
            #---------------------
                _in = (sample, out_nodes)
                predict = this_model.forward(_in).to(device)
                label = next_snapshot.node_feature()[out_nodes, -1].float()
                loss = loss_fn(predict.squeeze(), label).to(device)
                loss.backward()
                optimizer.step()
        
        #---------------------------------------------------------
        #turn model to inference mode
        this_model[0].set_mini_batch(False)
        this_model[1].set_mini_batch(False)
        this_model.eval()
        #---------------------------------------------------------

        this_snapshot = this_graph.dispatcher(n_snapshot-3, True)
        next_snapshot = this_graph.dispatcher(n_snapshot-2, True)
        node_samples = torch.arange(this_snapshot.num_nodes())
        predict = this_model.forward((this_snapshot, node_samples)).to(device)
        label = next_snapshot.node_feature()[:this_snapshot.num_nodes(), -1].float()
        loss = loss_fn(predict.squeeze(), label).to(device)
        #scheduler.step(loss)
        test_loss_list.append(loss.item())
        this_snapshot = this_graph.dispatcher(n_snapshot-2, True)
        next_snapshot = this_graph.dispatcher(n_snapshot-1, True)
        node_samples = torch.arange(this_snapshot.num_nodes())
        predict = this_model.forward((this_snapshot, node_samples)).to(device)
        label = next_snapshot.node_feature()[:this_snapshot.num_nodes(), -1].float()
        loss = loss_fn(predict.squeeze(), label).to(device)
        eval_loss_list.append(loss.item())
        print(test_loss_list[-1])
        print(eval_loss_list[-1])
        this_logger.info(predict.squeeze()[:20])
        this_logger.info(label[:20])

    this_logger.info(test_loss_list)
    this_logger.info(eval_loss_list)


