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
from typing import List, Tuple, Union

from dgl import add_self_loop

from nineturn.core.commonF import reshape_tensor
from nineturn.core.errors import DimensionError
from nineturn.core.logger import get_logger
from nineturn.core.types import Dropout, GATConv, GraphConv, MLBaseModel, SAGEConv, SGConv, Tensor, nt_layers_list
from nineturn.dtdg.types import BatchedSnapshot, Snapshot

logger = get_logger()


def _check_mini_batch_mode(mini_batch_mode: bool, input_snapshot: Union[Snapshot, BatchedSnapshot]):

    if (not mini_batch_mode) and isinstance(input_snapshot, BatchedSnapshot):
        error_message = """
                Please set mini batch mode to True to perform DGLBlock based batch training.
            """
        logger.error(error_message)
        raise DimensionError(error_message)
    elif mini_batch_mode and isinstance(input_snapshot, Snapshot):
        error_message = """
                Please set mini batch mode to False to perform non DGLBlock training.
            """
        logger.error(error_message)
        raise DimensionError(error_message)
    else:
        return True


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


        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        pass

    def call(self, in_sample: Tuple[Snapshot, List], training=False) -> Tuple[Tensor, List]:
        """All StaticGraphEncoder subclass should have a forward function.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.


        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        return self.forward(in_sample, training)

    def set_mini_batch(self, mini_batch: bool = True):
        """Turn on batch training mode.

        Since mini batch training with DGL requires specific support to read in DGLBlock instead of a DGLGraph,
        we need this to control which mode the encoder is used.

        Args:
            mini_batch: bool, default True to turn on batch training mode.
        """
        self.mini_batch = mini_batch


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

    def single_graph_forward(self, in_sample: Tuple[Snapshot, List],training=False) -> Tuple[Tensor, List]:
        """Forward function in normal mode.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        g = snapshot.observation
        h = snapshot.node_feature()
        h = self.layers[0](g, h)
        for layer in self.layers[1:]:
            h = self.dropout(h,training=training)
            h = layer(g, h)
        return (h, dst_node_ids)

    def mini_batch_forward(self, in_sample: Tuple[BatchedSnapshot, List],training=False) -> Tuple[Tensor, List]:
        """Forward function in batch mode.

        Args:
            in_sample: tuple, first entry is a BatchedSnapshot, second entry is the list[int] of targeted node ids.

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        if snapshot.num_blocks() != self.n_layers:
            error_message = f"""
                The input blocked sample has {snapshot.num_blocks()},
                but we are expecting {self.n_layers} which is the same as the number of GCN layers
            """
            logger.error(error_message)
            raise DimensionError(error_message)
        blocks = snapshot.observation
        h = snapshot.feature
        h = self.layers[0](blocks[0], h)
        for i in range(1, self.n_layers):
            h = self.dropout(h)
            h = self.layers[i](blocks[i], h)
        return (h, dst_node_ids)

    def forward(self, _input,training=False):
        """Forward function.

        It checks the self.mini_batch to see which mode the encoder is and apply the corresponding forward function.
        If self.mini_batch is true, apply mini_batch_forward(). Otherwise, apply single_graph_forward().
        """
        _check_mini_batch_mode(self.mini_batch, _input[0])
        if self.mini_batch:
            return self.mini_batch_forward(_input)
        else:
            return self.single_graph_forward(_input, training=training)


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
    def forward(self, in_sample: Tuple[Snapshot, List],training=False) -> Tuple[Tensor, List]:
        """Forward function.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        if isinstance(snapshot, BatchedSnapshot):
            error_message = "SGCN does not support DGLBlock based batch training."
            logger.error(error_message)
            raise DimensionError(error_message)
        else:
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

    def single_graph_forward(self, in_sample: Tuple[Snapshot, List],training=False) -> Tuple[Tensor, List]:
        """Forward function in normal mode.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        g = snapshot.observation
        h = snapshot.node_feature()
        for i in range(self.n_layers):
            h = self.layers[i](g, h)
            h = reshape_tensor(h, [-1, self.heads[i] * self.n_hidden])
        return (h, dst_node_ids)

    def mini_batch_forward(self, in_sample: Tuple[BatchedSnapshot, List], training=False) -> Tuple[Tensor, List]:
        """Forward function in batch mode.

        Args:
            in_sample: tuple, first entry is a BatchedSnapshot, second entry is the list[int] of targeted node ids.

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        if snapshot.num_blocks() != self.n_layers:
            error_message = f"""
                The input blocked sample has {snapshot.num_blocks()},
                but we are expecting {self.n_layers} which is the same as the number of GCN layers
            """
            logger.error(error_message)
            raise DimensionError(error_message)
        blocks = snapshot.observation
        h = snapshot.feature
        for i in range(self.n_layers):
            h = self.layers[i](blocks[i], h)
        return (h, dst_node_ids)

    def forward(self, _input,training=False):
        """Forward function.

        It checks the self.mini_batch to see which mode the encoder is and apply the corresponding forward function.
        If self.mini_batch is true, apply mini_batch_forward(). Otherwise, apply single_graph_forward().
        """
        _check_mini_batch_mode(self.mini_batch, _input[0])
        if self.mini_batch:
            return self.mini_batch_forward(_input, training=training)
        else:
            return self.single_graph_forward(_input, training=training)


class GraphSage(StaticGraphEncoder):
    """A wrapper of DGL SAGEConv."""

    def __init__(self, aggregator: str, in_feat: int, n_hidden: int, **kwargs):
        """Create GraphSage based on SAGEConv."""
        super().__init__()
        self.layers.append(SAGEConv(in_feat, n_hidden, aggregator, **kwargs))

    def forward(self, in_sample: Tuple[Snapshot, List], training=False) -> Tuple[Tensor, List]:
        """Forward function.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.

        Return:
            tuple of node-wise embedding for the targeted ids and the list of targeted node ids.
        """
        snapshot, dst_node_ids = in_sample
        if isinstance(snapshot, BatchedSnapshot):
            error_message = "GraphSage does not support DGLBlock based batch training."
            logger.error(error_message)
            raise DimensionError(error_message)
        else:
            g = snapshot.observation
            h = snapshot.node_feature()
        h = self.layers[0](g, h)
        return (h, dst_node_ids)
