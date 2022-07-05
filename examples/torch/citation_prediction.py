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
    n_nodes = this_graph.dispatcher(n_snapshot -1)[0].observation.num_nodes()
    this_logger.info(f"number of nodes: {n_nodes}")
    this_snapshot,_ = this_graph.dispatcher(20)
    in_dim = this_snapshot.num_node_features()
    hidden_dim = 32
    num_GNN_layers = 2
    num_RNN_layers = 2
    output_dim = 10
    activation_f = F.leaky_relu
    encoders = ['gcn', 'sgcn', 'gat', 'sage']
    decoders = ['lstm', 'gru', 'rnn']
    epochs = 2
    for g in range(4):
        for r in range(3):
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
            for trial in range(1):
                this_logger.info("--------------------------------------------------------")
                this_logger.info(f"start trial {trial}")
                if g == 0:
                    gnn = GCN(num_GNN_layers, in_dim, hidden_dim,  activation=activation_f, allow_zero_in_degree=True, dropout=0.2)
                elif g == 1:
                    gnn = SGCN(num_GNN_layers, in_dim, hidden_dim ,allow_zero_in_degree=True)
                elif g == 2:
                    gnn = GAT([1], in_dim, hidden_dim,  activation=activation_f,allow_zero_in_degree=True)
                else:
                    gnn = GraphSage('gcn', in_dim, hidden_dim,  activation=activation_f)

                output_decoder = MLP(output_dim, [10,20,10,5])
                
                if r == 0:
                    rnn = LSTM( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
                elif r == 1:
                    rnn = GRU( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)
                else:
                    rnn = RNN( hidden_dim, output_dim,n_nodes,num_RNN_layers,output_decoder)


                this_model = assembler(gnn, rnn)
                loss_fn = torch.nn.MSELoss().to(device)
                optimizer = torch.optim.Adam([{"params": this_model.parameters()}], lr=1e-3)
                loss_list=[]
                all_predictions=[]
                for epoch in range(2):
                    this_model.decoder.reset_memory_state()
                    for t in range(5,n_snapshot-2):
                        this_model.train()
                        optimizer.zero_grad()
                        this_snapshot, node_samples = this_graph.dispatcher(t)
                        next_snapshot,_ = this_graph.dispatcher(t+1)
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


