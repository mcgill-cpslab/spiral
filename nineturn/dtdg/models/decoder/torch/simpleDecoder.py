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
"""Pytorch based simple decoders. Designed specially for dynamic graph learning."""

from abc import abstractmethod
from typing import List, Tuple, Union
import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    """Prototype of simple decoder"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, in_sample: Tuple[torch.Tensor, List]) -> torch.Tensor:
        """All SimpleDecoder subclass should have a forward function.

        Args:
            in_sample: tuple, first entry is either a nodes embedding or the hidden representation from a sequential
            decoder, second entry is the list[int] of targeted node ids.


        Return: 
            prediction: torch.Tensor
        """
        pass


class MLP(SimpleDecoder):

    def __init__(self, input_dim:int, embed_dims:List[int], dropout:float=0.5, output_dim:int=1):
        super().__init__()
        layers = list()
        dim_last_layer = input_dim
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(dim_last_layer, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            dim_last_layer = embed_dim
        layers.append(torch.nn.Linear(dim_last_layer, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, in_state):
        """Implementation of forward.

        Args:
            in_sample: tuple, first entry is either a nodes embedding or the hidden representation from a sequential
            decoder, second entry is the list[int] of targeted node ids.


        Return: 
            prediction: torch.Tensor
        """
        node_embs, ids = in_state
        return self.mlp(node_embs)

