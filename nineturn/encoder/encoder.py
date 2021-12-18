import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, gnn_model, sequential_model):
        self.gnn_model = gnn_model
        self.sequential_model = sequential_model
        
    def forward_v2(self, graphs, Xs):
        for index, (g, X0) in enumerate(zip(graphs, Xs)):
            # The sequential model takes effects on X (the input to GNN)
            if index == 0:
                X = X0
            else:
                X = self.sequential_model(embs, X0)
            
            embs = self.model(g, X)
        
        return embs

        
    def forward_v1(self, graphs, Xs):
        embs = []
        for g, X in zip(graphs, Xs):
            embs.append(self.model(g, X))
        
        # the sequential model processes the output of all graphs
        embs = nn.utils.rnn.pad_sequence(embs, batch_first = True)
        res, (_, _) = self.sequential_model(embs)
        return res
        

if __name__ == '__main__':
    device = 'cpu'
    in_dim = 32
    hidden_dim = 32
    num_GNN_layers = 2
    num_RNN_layers = 2
    from .model import GCN

    gnn = GCN(in_dim, hidden_dim, num_GNN_layers, activation=F.leaky_relu, dropout=0.2).to(device)
    rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers = num_RNN_layers, batch_first = True).to(device)
    encoder = Encoder(gnn, rnn)
    