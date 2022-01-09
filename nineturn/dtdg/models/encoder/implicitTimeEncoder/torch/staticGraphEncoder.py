# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The models are adopted from https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn.py
"""

from typing import Tuple, List
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv, GATConv
from dgl import DGLGraph, add_self_loop
from nineturn.core.config import get_logger

logger = get_logger()


from abc import ABC, abstractmethod

class StaticGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    @abstractmethod
    def forward(self, in_sample: Tuple[DGLGraph, List])->Tuple[torch.Tensor, List]:
        pass


class GCN(StaticGraphEncoder):
    def __init__(self,
                 n_layers,
                 in_feats,
                 n_hidden,
                 dropout=0,
                 **kwargs):
        super().__init__()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, **kwargs))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, **kwargs))
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    # to fulfill sequential.forward it only accepts one input
    def forward(self, in_sample):
        g, h,dst_node_ids = in_sample
        g = add_self_loop(g)
        h = h.float()
        h = self.layers[0](g, h)
        for layer in self.layers[1:]:
            h = self.dropout(h)
            h = layer(g, h)
        return (h, dst_node_ids)




class GAT(StaticGraphEncoder):
    def __init__(self,
                 heads,
                 in_feat,
                 n_hidden,
                 **kwargs):
        super().__init__()
        n_layers = len(heads)
        if heads[-1] > 1:
            logger.warning(
                f"""The head of attention for the last layer is {heads[-1]} which is greater than 1, the output
                dimension will be {heads[-1] * n_hidden}"""
            )
        # input projection (no residual)
        self.layers.append(GATConv(
            in_feat, n_hidden, heads[0],
            **kwargs))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                n_hidden * heads[l-1], n_hidden, heads[l],
                **kwargs))
     

    def forward(self, in_sample):
        g, h,dst_node_ids = in_sample
        h = h.float()
        for layer in self.layers:
            logger.info(h.shape)
            h = layer(g, h)
            logger.info(h.shape)
            h = h.view(-1, h.size(1) * h.size(2))
        logger.info(h.shape)
        return (h, dst_node_ids)

