import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_d,hidden_d,n_nodes,device):
        super().__init__()
        self.input_d = input_d
        self.lstm = nn.LSTM(
            input_size=input_d,
            hidden_size=hidden_d,
            batch_first=True,
            num_layers=1
        ).to(device)
        self.linear = nn.Linear(in_features=hidden_d, out_features=1)
        self.memory = [(torch.randn(1, 1, hidden_d),torch.randn(1, 1, hidden_d)) for i in range(n_nodes)]


    def forward(self, in_state):
        node_embs, dst_node_ids = in_state
        out = np.zeros(len(dst_node_ids))
        #node_embs: [|V|, |hidden_dim|]
        #sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        for i in range(len(dst_node_ids)):
            in_feature = node_embs[i].view(1,1,self.input_d)
            tem_out, self.memory[dst_node_ids[i]] = self.lstm(in_feature, self.memory[dst_node_ids[i]])
            out[i] = self.linear(tem_out).flatten().item()
        return out
