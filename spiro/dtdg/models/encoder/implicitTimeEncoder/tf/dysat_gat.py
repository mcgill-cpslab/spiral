from abc import abstractmethod
from typing import List, Tuple, Union

from dgl import add_self_loop
import tensorflow as tf
from spiro.core.commonF import reshape_tensor
from spiro.core.errors import DimensionError
from spiro.core.logger import get_logger
from spiro.core.tf.structualAttention import StructuralAttentionLayer
from spiro.core.types import Dropout, GATConv, GraphConv, MLBaseModel, SAGEConv, SGConv, Tensor, nt_layers_list
from spiro.dtdg.types import BatchedSnapshot, Snapshot
from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import StaticGraphEncoder



class DysatGat(StaticGraphEncoder):
    def __init__(self, n_layers: int, in_feats: int, n_hidden: int,n_heads:int =3, dropout: float = 0, **kwargs):
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
        self.layers.append(StructuralAttentionLayer(in_feats, n_hidden,n_heads,dropout,0, **kwargs))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(StructuralAttentionLayer(n_hidden, n_hidden,n_heads,dropout,0, **kwargs))

    def forward(self, in_sample: Tuple[Snapshot, List],training=False) -> Tuple[Tensor, List]:
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
            h = layer(g, h)
        return (h, dst_node_ids)
