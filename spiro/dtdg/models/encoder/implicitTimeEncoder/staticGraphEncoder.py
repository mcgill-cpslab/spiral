# Copyright 2022 The Nine Turn Authors. All Rights Reserved.
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
"""The models are adopted from https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn.py."""

from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from dgl import add_self_loop

from spiro.core.backends import TENSORFLOW
from spiro.core.commonF import reshape_tensor
from spiro.core.logger import get_logger
from spiro.core.types import Dropout, GATConv, GraphConv, MLBaseModel, SAGEConv, SGConv, Tensor, nt_layers_list
from spiro.core.utils import _get_backend
from spiro.dtdg.types import Snapshot

logger = get_logger()

this_backend = _get_backend()


class StaticGraphEncoder(MLBaseModel):
    """Prototype of static graph encoder."""

    def __init__(self):
        """All static graph encoder should follow this contract.

        Must have layers to hold multiple layer of the same type.
        Must have internal state mini_batch to decide which forward function to use.
        """
        super().__init__()
        self.layers = nt_layers_list()
        self.mini_batch = False

    @abstractmethod
    def forward(self, in_sample: Tuple[Snapshot, List], training=False) -> Tuple[Tensor, List]:
        """All StaticGraphEncoder subclass should have a forward function.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.
            training: boolean, is it for training

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        pass

    def call(self, in_sample: Tuple[Snapshot, List], training=False) -> Tuple[Tensor, List]:
        """All StaticGraphEncoder subclass should have a forward function.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.
            training: boolean, is it for training

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        return self.forward(in_sample, training)

    def has_weights(self) -> bool:
        return True


class GCN(StaticGraphEncoder):
    """A wrapper of DGL GraphConv."""

    def __init__(self, n_layers: int, in_feats: int, n_hidden: int, dropout: float = 0, **kwargs):
        """Create a multiplayer GCN.

        Args:
            n_layers: int, number of GCN layers.
            in_feats: number of input features
            n_hidden: number of hidden units. This would also be the second dimention of the output embedding.
            dropout: probability to dropout in training, default to 0, no dropout.
            **kwargs: all other keyword arguments supported by dgl.GraphConv
        """
        super().__init__()
        self.n_layers = n_layers
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, **kwargs))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, **kwargs))
        # output layer
        self.dropout = Dropout(dropout)

    def forward(self, in_sample: Tuple[Snapshot, List], training=False) -> Tuple[Tensor, List]:
        """Forward function in normal mode.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.
            training: boolean, is it for training

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        g = snapshot.observation
        h = snapshot.node_feature()
        h = self.layers[0](g, h)
        for layer in self.layers[1:]:
            h = self.dropout(h, training=training)
            h = layer(g, h)
        return (h, dst_node_ids)


class SGCN(StaticGraphEncoder):
    """A wrapper of DGL SGConv. https://arxiv.org/pdf/1902.07153.pdf ."""

    def __init__(self, n_layers: int, in_feats: int, n_hidden: int, **kwargs):
        """Create a multiplayer GCN.

        Args:
            n_layers: int, number of GCN layers.
            in_feats: number of input features
            n_hidden: number of hidden units. This would also be the second dimention of the output embedding.
            **kwargs: all other keyword arguments supported by dgl.GraphConv
        """
        super().__init__()
        # input layer
        # in SGConv, k representing the hops for neiborhood information  aggregation, is the same as number of layers.
        self.layers.append(SGConv(in_feats, n_hidden, k=n_layers, **kwargs))

    # to fulfill sequential.forward it only accepts one input
    def forward(self, in_sample: Tuple[Snapshot, List], training: bool = False) -> Tuple[Tensor, List]:
        """Forward function.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.
            training: boolean, training mode or not

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        g = snapshot.observation
        h = snapshot.node_feature()
        h = self.layers[0](g, h)
        return (h, dst_node_ids)


class GAT(StaticGraphEncoder):
    """A wrapper of DGL GATConv."""

    def __init__(self, heads: List[int], in_feat: int, n_hidden: int, **kwargs):
        """Create a multiplayer GAT based on GATConv."""
        super().__init__()
        self.n_layers = len(heads)
        self.heads = heads
        self.n_hidden = n_hidden
        if heads[-1] > 1:
            logger.warning(
                f"""The head of attention for the last layer is {heads[-1]} which is greater than 1, the output
                dimension will be {heads[-1] * n_hidden}"""
            )
        # input projection (no residual)
        self.layers.append(GATConv(in_feat, n_hidden, heads[0], **kwargs))
        # hidden layers
        for i in range(1, self.n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(n_hidden * heads[i - 1], n_hidden, heads[i], **kwargs))

    def forward(self, in_sample: Tuple[Snapshot, List], training=False) -> Tuple[Tensor, List]:
        """Forward function in normal mode.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.
            training: boolean, training mode or not

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        g = add_self_loop(snapshot.observation)
        h = snapshot.node_feature()
        for i in range(self.n_layers):
            h = self.layers[i](g, h)
            h = reshape_tensor(h, [-1, self.heads[i] * self.n_hidden])
        return (h, dst_node_ids)


class GraphSage(StaticGraphEncoder):
    """A wrapper of DGL SAGEConv."""

    def __init__(self, aggregator: str, in_feat: int, n_hidden: int, depth: int = 50, **kwargs):
        """Create GraphSage based on SAGEConv."""
        super().__init__()
        self.depth = depth
        self.layers.append(SAGEConv(in_feat, n_hidden, aggregator, **kwargs))
        for i in range(1, depth):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator, **kwargs))

    def forward(self, in_sample: Tuple[Snapshot, List], training: bool = False) -> Tuple[Tensor, List]:
        """Forward function.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.
            training: boolean, training mode or not

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        g = snapshot.observation
        h = snapshot.node_feature()
        for layer in self.layers:
            h = layer(g, h)
        return (h, dst_node_ids)

    def has_weights(self) -> bool:
        logger.info("GraphSage has not trainable weights.")
        return True


# flake8: noqa
# import backend specific models
if this_backend == TENSORFLOW:
    from spiro.dtdg.models.encoder.implicitTimeEncoder.tf.dysat_gat import DysatGat as DysatGat
