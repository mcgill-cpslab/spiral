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
"""DySat.

Sankar, Aravind, et al. "Dysat: Deep neural representation learning on dynamic graphs via self-attention
networks." Proceedings of the 13th international conference on web search and data mining. 2020.
"""

from typing import List, Tuple

from spiro.core.errors import ValueError
from spiro.core.tf.structualAttention import StructuralAttentionLayer
from spiro.core.types import Tensor
from spiro.dtdg.models.encoder.implicitTimeEncoder.staticGraphEncoder import StaticGraphEncoder
from spiro.dtdg.types import Snapshot


class DysatGat(StaticGraphEncoder):
    """DySat."""

    def __init__(self, n_layers: int, in_feats: int, n_hidden: int, n_heads: int = 3, dropout: float = 0, **kwargs):
        """Create a multiplayer GCN.

        Args:
            n_layers: int, number of GCN layers.
            in_feats: number of input features
            n_hidden: number of hidden units. This would also be the second dimention of the output embedding.
            n_heads: number of attention heads
            dropout: probability to dropout in training, default to 0, no dropout.
            **kwargs: all other keyword arguments supported by dgl.GraphConv
        """
        super().__init__()
        self.n_layers = n_layers
        # input layer
        self.layers.append(StructuralAttentionLayer(in_feats, n_hidden, n_heads, dropout, 0, **kwargs))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(StructuralAttentionLayer(n_hidden, n_hidden, n_heads, dropout, 0, **kwargs))

    def forward(self, in_sample: Tuple[Snapshot, List], training=False) -> Tuple[Tensor, List]:
        """Forward function in normal mode.

        Args:
            in_sample: tuple, first entry is a snapshot, second entry is the list[int] of targeted node ids.
            training: boolearn, training mode or not

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

    def get_weights(self):
        """We don't support model saving for dysat yet."""
        raise ValueError("Save and load model is not supported for Dysat.")
